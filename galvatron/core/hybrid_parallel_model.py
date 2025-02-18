import torch
from torch import nn
import numpy as np
from galvatron.core import check_hp_config, hp_config_whole_model, get_enc_groups, mixed_precision_dtype, layer_shapes_dtypes_whole_model, get_chunks
from galvatron.core import gen_comm_groups, wrap_modules_relocation
from galvatron.core.initialize import init_empty_weights

class GalvatronModel(nn.Module):
    def __init__(self, hp_model):
        super().__init__()
        from galvatron.core import get_args
        self.args = get_args()
        self.model = hp_model
        self.iter = 0
        
    def forward_backward(self, batch, iter=None, profiler=None, loss_func=None):
        args, model = self.args, self.model
        self.iter = iter if iter is not None else self.iter
        if loss_func is not None:
            assert isinstance(batch, (tuple, list)) and isinstance(batch[0], (tuple, list)) and isinstance(batch[1], (tuple, list))
        elif not args.use_packing:
            loss_func = self.fake_loss_func
            assert isinstance(batch, (tuple, list))
            batch = [batch, [self.fake_tensor(batch[0])]]
        elif not args.use_flexSP:
            # batch: [input_ids, cu_seqlens]
            loss_func = self.fake_loss_func
            assert isinstance(batch, (tuple, list))
            batch = [batch, [self.fake_tensor(batch[1])]]
        else:
            loss_func = self.fake_loss_func
            for mbatch in batch: #[[seqs, cu_seqlens]]
                mbatch.append([self.fake_tensor(mbatch[0][1])])
        if args.pp_deg > 1:
            if args.pipeline_type == "gpipe":
                loss = model.gpipe_forward(batch, loss_func)
                if profiler is not None:
                    profiler.profile_memory(self.iter, "After Forward")
                model.gpipe_backward()
            elif args.pipeline_type == "pipedream_flush":
                loss = model.pipedream_flush_forward_backward(batch, loss_func)
        else:
            loss = model.no_pipeline_forward_backward(batch, loss_func, forward_only = args.profile_forward, profiler = profiler, iter = self.iter)
        self.iter += 1
        return self.loss_to_cpu(loss)
    
    def fake_tensor(self, x):
        if not self.args.use_packing:
            if isinstance(x,torch.Tensor):
                return torch.zeros([x.shape[0], 1], dtype=x.dtype, device=x.device)
        else:
            return torch.zeros([len(x) - 1 , 1], dtype = torch.int32, device="cuda")
    
    def fake_loss_func(self, labels, outputs):
        return outputs[0]
    
    def loss_to_cpu(self, loss):
        if isinstance(loss, (list, tuple)): # Average loss of each microbatch
            if len(loss) == 0:
                return None
            loss = np.mean([l.item() for l in loss])
        else:
            loss = loss.item()
        return loss

class GalvatronModelWrapper():
    def __init__(self, args, wrap_block_names=[]):
        self.args = args
        self.wrap_block_names = wrap_block_names
    
    # Wrap Galvatron Hybrid Parallel Model, need to be called after Galvatron is initialized
    def wrap_model_hybrid_parallel(self, model, model_config, hybrid_parallel_configs, model_info, construct_sequential_model, construct_tensor_parallel_model):
        return construct_hybrid_parallel_model_api(
            model,
            model_config,
            self.args,
            hybrid_parallel_configs,
            model_info,
            construct_sequential_model,
            construct_tensor_parallel_model,
            self.wrap_block_names
        )

    # Wrap Data Parallel Model, can be called on any PyTorch Model even when Galvatron is not initilized
    def wrap_model_data_parallel(self, model, device, dp_type='ddp', mixed_precision='bf16', comm_group=None, initialize_on_meta=False, backward_prefetch=True):
        from galvatron.core.parallel import wrap_model_data_parallel
        mixed_precision = mixed_precision_dtype(mixed_precision)
        return wrap_model_data_parallel(model, device, self.wrap_block_names, dp_type, mixed_precision, comm_group, initialize_on_meta, backward_prefetch)

    # Wrap Activation Checkpoint Model, can be called on any PyTorch Model even when Galvatron is not initilized
    def wrap_model_checkpoint(self, model):
        from galvatron.core.parallel import wrap_model_checkpoint
        return wrap_model_checkpoint(model, self.wrap_block_names)

def construct_hybrid_parallel_model_api(
    model,
    model_config,
    training_args,
    hybrid_parallel_configs,
    model_info,
    construct_sequential_model,
    construct_tensor_parallel_model,
    wrap_block_name=None,
    tied_wte_attr_names=None,
    sp_layernorm_attr_names=None,
):
    config, args, hp_configs = model_config, training_args, hybrid_parallel_configs

    # Get model-specific model info: module_types, layernum_list, layer_shapes_list, layer_dtypes_list
    model_info = model_info(config, args)
    module_types = model_info.module_types()
    layernum_list = model_info.layernums()
    layer_shapes_list = model_info.shapes()
    layer_dtypes_list = model_info.dtypes()
    
    # Check the validity of hp_configs (encoders only)
    check_hp_config(hp_configs, layernum_list)
    
    # Calculate shapes and dtypes for whole model (including embed/cls/... layers)
    shapes_whole, dtypes_whole = layer_shapes_dtypes_whole_model(module_types, layernum_list, layer_shapes_list, layer_dtypes_list)
    
    # Get hp_configs_whole for the whole model (including embed/cls/... layers)
    hp_configs_whole = hp_config_whole_model(module_types, hp_configs, embed_sdp=args.embed_sdp, embed_ckpt=0, vocab_tp = args.vocab_tp)
    # hp_configs_whole['tp_sizes_whole'] = [1,1,1,1,1]
    if args.use_ulysses:
        hp_configs_whole['sp_sizes_whole'] = hp_configs_whole['tp_sizes_whole']
        hp_configs_whole['tp_sizes_whole'] = [1] * len(hp_configs_whole['tp_sizes_whole'])
    else:
        hp_configs_whole['sp_sizes_whole'] = [1] * len(hp_configs_whole['tp_sizes_whole'])
    # hp_configs_whole['dp_sizes_whole'] = [2,2,2,2,2]
    # [Step 0] Generate communication groups
    pp_group, tp_groups_whole, sp_groups_whole, dp_groups_whole, seq_data_groups_whole ,allgather_groups_whole, split_groups_whole, fused_allgather_groups_whole, fused_split_groups_whole, embedding_group = \
        gen_comm_groups(hp_configs_whole['tp_sizes_whole'], hp_configs_whole['sp_sizes_whole'] ,hp_configs_whole['pp_deg'], hp_configs_whole['tp_consec_whole'], show_rank = 0)
    
    # Note: We don't need redistribute, set the groups to None
    split_groups_whole = [None] * len(split_groups_whole)
    allgather_groups_whole = [None] * len(allgather_groups_whole)
    fused_allgather_groups_whole = [None] * len(fused_allgather_groups_whole)
    fused_split_groups_whole = [None] * len(fused_split_groups_whole)
    
    # [Step 1] Construct Tensor Parallel Model based on tp_groups using model-specific TP function 
    # Note, reuse ParallelAttention And ParallelMLP to construct sequence parallel
    if args.initialize_on_meta and args.shape_order == "SBH":
        with init_empty_weights(True):
            model = construct_tensor_parallel_model(model, config, tp_groups_whole, sp_groups_whole, seq_data_groups_whole[-1])
    else:
        model = construct_tensor_parallel_model(model, config, tp_groups_whole, sp_groups_whole, seq_data_groups_whole[-1])
    # [Step 2] Construct Sequantial model using model-specific sequential function
    if args.initialize_on_meta and args.shape_order == "SBH":
        with init_empty_weights(True):
            model = construct_sequential_model(model, config)
    else:
        model = construct_sequential_model(model, config)

    # [Step 3] Wrap Relocation modules if necessary
    model = wrap_modules_relocation(model, allgather_groups_whole, split_groups_whole, fused_allgather_groups_whole, fused_split_groups_whole)
    hp_configs_whole['sp_sizes_whole'] = [torch.distributed.get_world_size(sp_groups_whole[0].group)] * len(hp_configs_whole['sp_sizes_whole'])
    # [Step 4] Construct Pipeline Module and place the layers on corresponding devices
    from galvatron.core.pipeline import PipelineParallel
    hp_model = PipelineParallel(
        model=model,
        model_ranks=hp_configs_whole['pp_ranks_whole'],
        layer_output_tensor_shapes=shapes_whole,
        layer_output_tensor_dtypes=dtypes_whole,
        layer_tp_sizes=hp_configs_whole['tp_sizes_whole'],
        layer_sp_sizes=hp_configs_whole['sp_sizes_whole'],
        chunks=get_chunks(args),
        process_group=pp_group.ranks,
        embedding_group=embedding_group,
        nproc_per_node=8,
        info=False,
        tied_wte_attr_names=tied_wte_attr_names,
    )

    # [Step 5] Wrap Data Parallel modules based on dp_types & dp_groups
    hp_model.wrap_pipeline_modules_data_parallel(
        hp_configs_whole['dp_types_whole'],
        seq_data_groups_whole,
        module_types=module_types,
        mixed_precision=mixed_precision_dtype(args.mixed_precision),
        wrap_block_name=wrap_block_name,
        tied_wte_attr_names=tied_wte_attr_names,
    )
    
    hp_model.gen_sp_layernorm_info(
        layer_module_types=module_types,
        layer_tp_groups=tp_groups_whole,
        sp_layernorm_attr_names=sp_layernorm_attr_names,
    )
    
    # [Step 6] Wrap checkpoint based on checkpoint_flags
    hp_model.wrap_pipeline_modules_checkpoint(hp_configs_whole['checkpoint_flags_whole'], wrap_block_name=wrap_block_name)
    
    model = GalvatronModel(hp_model)
    
    model.dp_groups_whole = dp_groups_whole # weight partit through seq_data_group, while data partit through dp_groups
    
    return model