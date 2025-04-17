import numpy as np
from tqdm import trange
from .cost_model import pipeline_costmodel
from .cost_model import OtherTimeCostModel
from .cost_model_args import ModelArgs, TrainArgs, ParallelArgs, ProfileModelArgs, ProfileHardwareArgs

class DPAlg():
    def __init__(self, max_mem=8200, other_mem_cost=None, other_time_cost = None, layer_num=24, strategy_num=4, strategy_set=None, fine_grained_mode=True, use_cpp_core=True) -> None:
        assert(other_mem_cost != None)
        self.max_mem = max_mem + 1
        self.layer_num = layer_num
        self.strategy_num = strategy_num
        self.other_mem_cost = other_mem_cost
        self.other_time_cost = other_time_cost

        self._f = np.full((self.max_mem, strategy_num), 0, dtype=np.float64)
        
        self.v_data = None
        self.inter_cost = None
        self.intra_cost = None

        self._mark = np.full((layer_num, self.max_mem, strategy_num), -1, dtype=np.int32)
        self.use_cpp_core = use_cpp_core
        self.strategy_set = strategy_set
        self.fine_grained_mode = fine_grained_mode
    
    def set_v_and_cost(self, v: np.ndarray, intra_layer_cost: np.ndarray, inter_layer_cost: np.ndarray):
        assert v.ndim == 2
        assert inter_layer_cost.ndim == 3
        assert intra_layer_cost.ndim == 2

        assert v.shape[0] == self.layer_num
        assert v.shape[1] == self.strategy_num

        assert inter_layer_cost.shape[0] == self.layer_num
        assert inter_layer_cost.shape[1] == self.strategy_num and inter_layer_cost.shape[2] == self.strategy_num

        assert intra_layer_cost.shape[0] == self.layer_num
        assert intra_layer_cost.shape[1] == self.strategy_num

        self.v_data = v.astype(np.int32)
        self.inter_cost = inter_layer_cost
        self.intra_cost = intra_layer_cost

    def fit(self):
        # if self.strategy_num == 1:
        #     total_v = np.sum(self.v_data[:,0])
        #     total_cost = np.sum(self.intra_cost[:,0])
        #     min_cost = np.inf
        #     min_tp = -1
        #     min_mem = -1
        #     for tp in other_mem_cost.keys():
        #         if total_v <= self.max_mem - 1 - other_mem_cost[tp]:
        #             if min_cost > total_cost + other_time_cost[tp]:
        #                 min_cost = total_cost + other_time_cost[tp]
        #                 min_tp = tp
        #                 min_mem = self.max_mem - 1 - other_mem_cost[tp] - total_v
        #     if min_tp != -1:
        #         return total_cost, [0] * self.layer_num, min_mem, min_tp
        #     return np.inf, None, -1, -1
        # print(self.other_mem_cost, self.other_time_cost)
        if not self.fine_grained_mode:
            res_list = {k:np.full((self.layer_num), -1, dtype=np.int32) for k,v in self.other_mem_cost.items()}
            total_cost = {k:np.inf for k,v in self.other_mem_cost.items()}
            remaining_mem = {k:-1 for k,v in self.other_mem_cost.items()}
            for k,v in self.other_mem_cost.items():
                for i in range(self.strategy_num):
                    if self.strategy_set[i][1]==k:
                        time_cost = sum(self.intra_cost[:,i]) + sum(self.inter_cost[:,i,i]) + self.other_time_cost[k]
                        mem_cost = sum(self.v_data[:,i]) + self.other_mem_cost[k]
                        if self.max_mem - 1 - mem_cost >= 0 and total_cost[k] > time_cost:
                            remaining_mem[k] = self.max_mem - 1 - mem_cost
                            total_cost[k] = time_cost
                            res_list[k] = np.full((self.layer_num), i, dtype=np.int32)
            return total_cost, res_list, remaining_mem       
        if self.use_cpp_core:
            import galvatron_dp_core
            res_list = {k:np.full((self.layer_num), -1, dtype=np.int32) for k,v in self.other_mem_cost.items()}
            total_cost, remaining_mem = galvatron_dp_core.dynamic_programming_core(
                self.layer_num, 
                self.max_mem, 
                self.strategy_num, 
                self.v_data, 
                self._mark, 
                self._f, 
                self.inter_cost, 
                self.intra_cost,
                self.other_mem_cost,
                self.other_time_cost,
                res_list,
                )
            res_list = {k:list(v) for k,v in res_list.items()}

            return total_cost, res_list, remaining_mem

        for i in range(self.layer_num):
            for v in range(self.max_mem - 1, -1, -1):
                for s in range(self.strategy_num):

                    if v < self.v_data[i, s]:
                        self._mark[i, v, s] = -1
                        self._f[v, s] = np.inf
                        continue

                    candidates = [self._f[v - self.v_data[i, s], si] + self.inter_cost[i, si, s] for si in range(self.strategy_num)]
                    candidates = np.array(candidates) + self.intra_cost[i, s]

                    min_index = np.argmin(candidates)

                    self._mark[i, v, s] = min_index
                    self._f[v, s] = candidates[min_index]
        
        next_index, next_v = np.argmin(self._f[-1, :]), self.max_mem - 1
        total_cost = self._f[-1, next_index]

        if not total_cost < np.inf:
            return np.inf, None, -1

        res_list = [-1] * self.layer_num
        res_list[-1] = next_index

        for i in range(self.layer_num - 1, 0, -1):
            next_index, next_v = self._mark[i, next_v, next_index], next_v - self.v_data[i, next_index]
            res_list[i - 1] = next_index

        return total_cost, res_list, next_v - self.v_data[0, next_index]

class DpOnModel:
    def __init__(   self, 
                    strategies_set, 
                    memcost_model, 
                    timecost_model, 
                    model_args_list=None,
                    train_args_list=None,
                    parallel_args_list=None,
                    profile_model_args_list=None,
                    profile_hardware_args_list=None,
                    max_mem=8192, 
                    layer_num=24,
                    sequence_len = [512],
                    multi_layer_type=False,
                    pp_stage_dict=None,
                    search_history=None,
                    comm_coe_dict={},
                    gpu_num=8,
                    mem_cache=True,
                    model_microbatch_after_dp=False,
                    pipeline_type='gpipe',
                    config = None,
                    logger = None):
        self.strategies_set = strategies_set
        self.memcost_model = memcost_model
        self.timecost_model = timecost_model
        self.model_args_list = model_args_list
        self.train_args_list = train_args_list
        self.parallel_args_list = parallel_args_list
        self.profile_model_args_list = profile_model_args_list
        self.profile_hardware_args_list = profile_hardware_args_list
        self.max_mem = max_mem
        self.layer_num = layer_num
        self.sequence_len = sequence_len
        self.n_gpu = strategies_set[0][0] * strategies_set[0][1] * strategies_set[0][2]
        self.ppdeg_set = np.unique(np.array([s[0] for s in strategies_set], dtype=np.int32))
        self.multi_layer_type = multi_layer_type
        self.search_history = search_history
        self.comm_coe_dict = comm_coe_dict
        self.gpu_num = gpu_num
        self.config = config
        self.logger = logger
        if multi_layer_type:
            # If multi_layer_type == True, layer_num/model_args_list/train_args_list/parallel_args_list/profile_model_args_list/profile_hardware_args_list should be list.
            # e.g. for T5, layer_num = [12, 12], model_args_list = [model_args_list_enc, model_args_list_dec]
            # train_args_list = [train_args_list_enc, train_args_list_dec]
            # pp_stage_dict = {1:[24], 2: [15, 9], 4: [7, 7, 5, 5], 8:[4, 4, 4, 4, 2, 2, 2, 2]}
            assert(isinstance(layer_num, list))
            self.total_layer_num = sum(layer_num)
            assert(isinstance(model_args_list, list) and len(layer_num) == len(model_args_list))
            assert(isinstance(train_args_list, list) and len(layer_num) == len(train_args_list))
            assert(isinstance(parallel_args_list, list) and len(layer_num) == len(parallel_args_list))
            assert(isinstance(profile_model_args_list, list) and len(layer_num) == len(profile_model_args_list))
            assert(isinstance(profile_hardware_args_list, list) and len(layer_num) == len(profile_hardware_args_list))
            assert(isinstance(pp_stage_dict, dict))
            for ppdeg in self.ppdeg_set:
                if ppdeg > 1:
                    assert(ppdeg in pp_stage_dict.keys())
                    assert(sum(pp_stage_dict[ppdeg])==self.total_layer_num)
            self.pp_stage_dict = pp_stage_dict
            if 1 not in self.pp_stage_dict.keys():
                self.pp_stage_dict[1] = [self.total_layer_num]
        self.mem_cache = 0
        if max_mem // 1024 > 20 and mem_cache:
            self.mem_cache = int(max_mem * 0.2) # reserved memory for pytorch memory cache
            self.max_mem -= self.mem_cache
        self.model_microbatch_after_dp = model_microbatch_after_dp
        self.pipeline_type = pipeline_type

    def match_strategy(self, s1, s2, except_keys=[]):
        # print(s1, s2)
        if not np.array_equal(s1[:3], s2[:3]):
            return False
        s1, s2 = s1[-1], s2[-1]
        for key in s1.keys():
            if key not in except_keys:
                if key not in s2.keys() or s1[key] != s2[key]:
                    return False
        for key in s2.keys():
            if key not in except_keys:
                if key not in s1.keys() or s1[key] != s2[key]:
                    return False
        return True

    def _build_dp_and_run_multi_layer_type(self, pp_deg, bsz, mbsz, min_tp, max_tp, vsp, embed_sdp, sp_search):
        # Look for results in search history
        # history_results = []
        # for i in range(pp_deg):
        #     key = (bsz, mbsz, pp_deg, min_tp, i)
        #     if self.search_history is not None and key in self.search_history.keys() and self.search_history[key]['mem_cost'] <= self.max_mem:
        #         history_results.append(self.search_history[key])
        #     else:
        #         history_results.append(None)

        if self.model_microbatch_after_dp:
            dp_size = self.gpu_num // pp_deg
            chunks = [parallel_args.optimal_chunk_func(bsz * min_tp // dp_size, [pp_deg, min_tp, dp_size], mbsz, min_tp) for parallel_args in self.parallel_args_list]
        strategy_set = list(filter(lambda s: s[0] == pp_deg, self.strategies_set))
        strategy_num = len(strategy_set)

        intra_layer_cost_list = []

        for i in range(len(self.layer_num)):
            if self.model_microbatch_after_dp:
                intra_layer_cost = [self.timecost_model(strategy, bsz/chunks[i],
                                                        model_args=self.model_args_list[i], train_args=self.train_args_list[i],
                                                        parallel_args=self.parallel_args_list[i], profile_model_args=self.profile_model_args_list[i],
                                                        profile_hardware_args=self.profile_hardware_args_list[i], logger=self.logger).gen_result() for strategy in strategy_set]
            else:
                intra_layer_cost = [self.timecost_model(strategy, bsz, 
                                                        model_args=self.model_args_list[i], train_args=self.train_args_list[i],
                                                        parallel_args=self.parallel_args_list[i], profile_model_args=self.profile_model_args_list[i],
                                                        profile_hardware_args=self.profile_hardware_args_list[i], logger=self.logger).gen_result() for strategy in strategy_set]
            intra_layer_cost = np.array(intra_layer_cost, dtype=np.float64).reshape(1, -1).repeat(self.layer_num[i], axis=0)
            intra_layer_cost_list.append(intra_layer_cost)

        intra_layer_cost = np.concatenate(intra_layer_cost_list, axis = 0)
        min_cost_strategy_ids = np.argmin(intra_layer_cost, axis=1)

        other_mem_cost = {}
        other_time_cost = OtherTimeCostModel(mbsz, pp_deg, self.n_gpu, vsp, embed_sdp, min_tp, max_tp, self.sequence_len, 
                                            model_args=self.model_args_list[0],
                                            train_args=self.train_args_list[0],
                                            parallel_args=self.parallel_args_list[0],
                                            profile_model_args=self.profile_model_args_list[0],
                                            profile_hardware_args=self.profile_hardware_args_list[0], logger=self.logger).gen_result()
        if self.pipeline_type == "gpipe":
            v_list = []
            for i in range(len(self.layer_num)):
                mem_cost_list = [self.memcost_model(strategy, bsz, mbsz = mbsz, min_tp = min_tp, max_tp = max_tp, vsp = vsp, embed_sdp = embed_sdp, 
                                                    model_args=self.model_args_list[i], train_args=self.train_args_list[i], parallel_args=self.parallel_args_list[i], 
                                                    profile_model_args=self.profile_model_args_list[i], logger=self.logger).get_memory_cost() for strategy in strategy_set]
                # TODO: mulitple layer type
                if i == 0:
                    for k, v in mem_cost_list[0]['other'].items():
                        other_mem_cost[k] = np.ceil(v).astype(int)
                v = [cost['enc_total'] for cost in mem_cost_list]
                v = np.ceil(np.array(v)).astype(np.int32)
                v = v.reshape(1, -1).repeat(self.layer_num[i], axis=0)
                v_list.append(v)
            v = np.concatenate(v_list, axis = 0)
            v_list_stage_idx = v
        elif self.pipeline_type == "pipedream_flush":
            v_list_stage_idx = []
            for stage_idx in range(pp_deg):
                v_list = []
                for i in range(len(self.layer_num)):
                    mem_cost_list = [self.memcost_model(strategy, bsz, mbsz = mbsz, min_tp = min_tp, max_tp = max_tp, stage_idx = stage_idx, vsp = vsp, embed_sdp = embed_sdp,
                                                        model_args=self.model_args_list[i], train_args=self.train_args_list[i], parallel_args=self.parallel_args_list[i], 
                                                        profile_model_args=self.profile_model_args_list[i], logger=self.logger).get_memory_cost() for strategy in strategy_set]
                    # TODO: mulitple layer type
                    if stage_idx == 0 and i == 0:
                        for k, v in mem_cost_list[0]['other'].items():
                            other_mem_cost[k] = np.ceil(v).astype(int)
                    # other_mem_cost = np.ceil(mem_cost_list[0]['other']).astype(int)
                    v = [cost['enc_total'] for cost in mem_cost_list]
                    v = np.ceil(np.array(v)).astype(np.int32)
                    v = v.reshape(1, -1).repeat(self.layer_num[i], axis=0)
                    v_list.append(v)
                v = np.concatenate(v_list, axis = 0)
                v_list_stage_idx.append(v)

        # NEW VERSION: inter-layer timecost model
        # print(other_time_cost, other_mem_cost, v[0], strategy_set)
        inter_layer_cost_list = []
        for idx in range(len(self.layer_num)):
            inter_layer_cost = np.zeros((strategy_num, strategy_num))
            for i in range(strategy_num):
                for j in range(strategy_num):
                    case1 = strategy_set[j][1] > strategy_set[i][1]
                    case2 = False
                    case3 = False
                    if 'tp' in strategy_set[j][-1].keys() and 'tp' in strategy_set[i][-1].keys():
                        case2 = (strategy_set[j][1] == strategy_set[i][1] and strategy_set[j][-1]['tp'] != strategy_set[i][-1]['tp'])
                        world_size = strategy_set[i][1] * strategy_set[i][2]
                        case3 = (world_size == 8 and strategy_set[i][1] == 4 and strategy_set[j][1] == 2 \
                            and strategy_set[j][-1]['tp'] != strategy_set[i][-1]['tp'])
                    sample_num = self.sequence_len[idx] * self.config.hidden_size * (4 if self.config.mixed_precision == "fp32" else 2)
                    case4 = self.config.sequence_parallel and strategy_set[j][1] != strategy_set[i][1]
                    if case1 or case2 or case3 or case4:
                        nw_max_tp = max(strategy_set[j][1], strategy_set[i][1])
                        inter_layer_cost[i, j] = (nw_max_tp-1) / nw_max_tp * mbsz * (nw_max_tp // min_tp) * sample_num
                    # if self.config.sequence_parallel:
                    #     if strategy_set[j][1] != strategy_set[i][1]:
                    #         inter_layer_cost[i, j] += (strategy_set[j][1]-1) / strategy_set[j][1] * mbsz * sample_num / strategy_set[j][1] / 2

                # if case1 or case2 or case3:
                #      ratio = strategy_set[j][1]
                #      activation = 2 * bsz / strategy_set[j][2]
                #      inter_layer_cost[i, j] = (ratio - 1) * activation / ratio

            # find corresponding communication coefficient
            for i in range(strategy_num):
                for j in range(strategy_num):
                    tp_size, dp_size = max(strategy_set[j][1], strategy_set[i][1]), min(strategy_set[j][2], strategy_set[i][2])
                    if tp_size == 1 or dp_size == 1:
                        coe = self.comm_coe_dict['%d'%tp_size] if '%d'%tp_size in self.comm_coe_dict.keys() else self.comm_coe_dict['%d_1'%tp_size]
                    else:
                        # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                        if strategy_set[j][1] > strategy_set[i][1]:
                            info = strategy_set[j][-1]
                        else:
                            info = strategy_set[i][-1]
                        assert 'tp' in info.keys() and info['tp'] in [0, 1]
                        if info['tp']:
                            coe = self.comm_coe_dict['%d_1'%tp_size]
                        else:
                            coe = self.comm_coe_dict['%d_0'%tp_size]
                    # Expand communication cost
                    inter_layer_cost[i, j] = inter_layer_cost[i, j] * coe * 1e-7

                    # add a small bias to sort fsdp and dp
                    strategy0, strategy1 = strategy_set[i], strategy_set[j]
                    # if i != j and np.array_equal(strategy0[:3], strategy1[:3]):
                    #     case1 = 'tp' not in strategy0[-1] and 'fsdp' in strategy0[-1] and strategy0[-1]['fsdp']!=strategy1[-1]['fsdp']
                    #     case2 = 'tp' in strategy0[-1] and strategy0[-1]['tp']==strategy1[-1]['tp'] and strategy0[-1]['fsdp']!=strategy1[-1]['fsdp']
                    #     if (case1 or case2) and strategy0[-1]['fsdp']:
                    #         inter_layer_cost[i, j] = 1e-4
                    # tp -> sp
                    if i != j and self.match_strategy(strategy0, strategy1, except_keys=['sp']):
                        if 'sp' in strategy1[-1] and strategy1[-1]['sp']:
                            inter_layer_cost[i, j] = 1e-10
                    # ->f     c -> fc
                    if i != j and self.match_strategy(strategy0, strategy1, except_keys=['fsdp']):
                        if 'fsdp' in strategy1[-1] and strategy1[-1]['fsdp']:
                            inter_layer_cost[i, j] = 1e-9
                    # ->c  f -> cf
                    if i != j and self.match_strategy(strategy0, strategy1, except_keys=['cpt']):
                        if 'cpt' in strategy1[-1] and strategy1[-1]['cpt']:
                            inter_layer_cost[i, j] = 2e-9
                    # ->fc
                    if i != j and self.match_strategy(strategy0, strategy1, except_keys=['fsdp','cpt']):
                        if 'fsdp' in strategy1[-1] and strategy1[-1]['fsdp'] and 'cpt' in strategy1[-1] and strategy1[-1]['cpt']:
                            inter_layer_cost[i, j] = 3e-9
                    # f->c
                    if i != j and self.match_strategy(strategy0, strategy1, except_keys=['fsdp','cpt']) \
                            and not self.match_strategy(strategy0, strategy1, except_keys=['fsdp']) \
                            and not self.match_strategy(strategy0, strategy1, except_keys=['cpt']):
                        if 'fsdp' in strategy0[-1] and strategy0[-1]['fsdp'] and 'cpt' in strategy1[-1] and strategy1[-1]['cpt']:
                            inter_layer_cost[i, j] = 1e-9
            inter_layer_cost_list.append(inter_layer_cost)
        for i in range(len(self.layer_num)):
            inter_layer_cost_list[i] = np.expand_dims(inter_layer_cost_list[i], axis=0).repeat(self.layer_num[i], axis=0)
        inter_layer_cost = np.concatenate(inter_layer_cost_list, axis=0)
        inter_layer_cost[0, :, :] = 0 # no inter-layer communication cost in first layer

        pp_stage_list = self.pp_stage_dict[pp_deg]
        start_layer = 0
        comm_cost_list, res_list_list, mem_remain_list, mem_cost_list = [], [], [], []
        best_strategy_flag = {k:[False for i in range(pp_deg)] for k,v in other_mem_cost.items()}
        from_history = None

        if self.config.fine_grained_mode==0:
            final_comm_cost = np.inf
            vtp = -1
            final_comm_cost_list, final_res_list_list, final_mem_remain_list, final_mem_cost_list = [], [], [], []

            for si in range(len(strategy_set)):
                s = strategy_set[si]
                local_strategy_set = [s]
                start_layer = 0
                comm_cost_list, res_list_list, mem_remain_list, mem_cost_list = [], [], [], []
                for i in range(pp_deg):
                    if self.config.sequence_parallel:
                        global_memory = mbsz / min_tp * max_tp * self.config.hidden_size * max(self.sequence_len) * 4 / 1024 / 1024
                        if self.config.mixed_precision:
                            global_memory = global_memory / 2
                    else:
                        global_memory = 0
                    nw_other_mem_cost = {k:v[i] + int(global_memory) for k,v in other_mem_cost.items()}
                    nw_other_time_cost = {k:v[i] for k,v in other_time_cost[0].items()}
                    mem_cost = {k:0 for k,v in other_time_cost[0].items()}
                    dp = DPAlg(self.max_mem, nw_other_mem_cost, nw_other_time_cost, pp_stage_list[i], 1, local_strategy_set, self.config.fine_grained_mode)
                    if self.pipeline_type == "pipedream_flush":
                        v = v_list_stage_idx[i]
                    dp.set_v_and_cost(v[start_layer:start_layer+pp_stage_list[i],si:si+1], 
                                        intra_layer_cost[start_layer:start_layer+pp_stage_list[i],si:si+1], 
                                        inter_layer_cost[start_layer:start_layer+pp_stage_list[i],si:si+1,si:si+1])
                    comm_cost, res_list, mem_remain = dp.fit()
                    for k,v in comm_cost.items():
                        if mem_remain[k] == -1:
                            res_list[k] = None
                        
                        best_strategy_flag[k][i] = res_list[k] is not None and (np.array(res_list[k]) == min_cost_strategy_ids[start_layer:start_layer+pp_stage_list[i]]).all()
                        if res_list[k] is not None:
                            res_list[k] = list(map(lambda x: local_strategy_set[x], res_list[k]))
                        mem_cost[k] = self.max_mem - mem_remain[k] if mem_remain[k] >= 0 else np.inf
                        
                    comm_cost_list.append(comm_cost)
                    res_list_list.append(res_list)
                    mem_remain_list.append(mem_remain)
                    mem_cost_list.append(mem_cost)
                    start_layer += pp_stage_list[i]
                
                for k in other_time_cost[0].keys():
                    nw_res_list_list = [v2[k] for v2 in res_list_list]
                    nw_comm_cost_list = [v2[k] for v2 in comm_cost_list]
                    if self.model_microbatch_after_dp:
                        if None not in nw_res_list_list:
                            res_list = []
                            for res in nw_res_list_list:
                                res_list += res
                            pipeline_cost = pipeline_costmodel(self.timecost_model, self.layer_num, self.model_args_list, self.train_args_list, self.parallel_args_list, self.profile_model_args_list, self.profile_hardware_args_list, res_list, pp_stage_list, chunks, bsz, min_tp, other_time_cost[1][k], self.logger)
                            # print(sum(comm_cost_list),pipeline_cost)
                            # print(pp_stage_list, res_list_list)
                            if final_comm_cost > pipeline_cost:
                                final_comm_cost = pipeline_cost
                                vtp = k
                                final_res_list_list = [v2[vtp] for v2 in res_list_list]
                                final_mem_remain_list = [v2[vtp] for v2 in mem_remain_list]
                                final_mem_cost_list = [v2[vtp] for v2 in mem_cost_list]
                    else:
                        final_comm_cost = sum(nw_comm_cost_list)
                        
                if vtp == -1:
                    res_list_list, mem_remain_list, mem_cost_list = None, [-1 for v2 in mem_remain_list], [-1 for v2 in mem_cost_list]
            return final_comm_cost, final_res_list_list, final_mem_remain_list, final_mem_cost_list, vtp, best_strategy_flag, from_history
        
        for i in range(pp_deg):
            if self.config.sequence_parallel and self.config.global_memory_buffer and sp_search != 2:
                global_memory = mbsz / min_tp * max_tp * self.config.hidden_size * max(self.sequence_len) * 4 / 1024 / 1024
                if self.config.mixed_precision:
                    global_memory = global_memory / 2
            else:
                global_memory = 0
            # if sp_search != 1:
            #     global_memory += 8192 # reserved memory for efficient all2all communication
            nw_other_mem_cost = {k:v[i] + int(global_memory) for k,v in other_mem_cost.items()}
            nw_other_time_cost = {k:v[i] for k,v in other_time_cost[0].items()}
            mem_cost = {k:0 for k,v in other_time_cost[0].items()}
            dp = DPAlg(self.max_mem, nw_other_mem_cost, nw_other_time_cost, pp_stage_list[i], strategy_num, strategy_set, self.config.fine_grained_mode)
            if self.pipeline_type == "pipedream_flush":
                v = v_list_stage_idx[i]
            else:
                v = v_list_stage_idx
            dp.set_v_and_cost(v[start_layer:start_layer+pp_stage_list[i]], 
                                intra_layer_cost[start_layer:start_layer+pp_stage_list[i]], 
                                inter_layer_cost[start_layer:start_layer+pp_stage_list[i]])
            comm_cost, res_list, mem_remain = dp.fit()
            for k,v in comm_cost.items():
                if mem_remain[k] == -1:
                    res_list[k] = None
                
                best_strategy_flag[k][i] = res_list[k] is not None and (np.array(res_list[k]) == min_cost_strategy_ids[start_layer:start_layer+pp_stage_list[i]]).all()
                if res_list[k] is not None:
                    res_list[k] = list(map(lambda x: strategy_set[x], res_list[k]))
                mem_cost[k] = self.max_mem - mem_remain[k] if mem_remain[k] >= 0 else np.inf
                
                # if res_list[k] is not None:
                #     if self.config.sequence_parallel and res_list[k][0][1] != k: # vsp != first layer tp
                #         sample_num = self.sequence_len[0] * self.config.hidden_size * (4 if self.config.mixed_precision == "fp32" else 2)
                #         nw_max_tp = max(res_list[k][0][1], k)
                #         comm_num = (nw_max_tp-1) / nw_max_tp * mbsz * (nw_max_tp // min_tp) * sample_num
                #         coe = self.comm_coe_dict['%d'%nw_max_tp] if '%d'%nw_max_tp in self.comm_coe_dict.keys() else self.comm_coe_dict['%d_1'%nw_max_tp]
                #         comm_cost[k] += comm_num * coe * 1e-7
                        
            comm_cost_list.append(comm_cost)
            res_list_list.append(res_list)
            mem_remain_list.append(mem_remain)
            mem_cost_list.append(mem_cost)
            start_layer += pp_stage_list[i]
        comm_cost = np.inf
        vtp = -1
        for k in other_time_cost[0].keys():
            nw_res_list_list = [v2[k] for v2 in res_list_list]
            nw_comm_cost_list = [v2[k] for v2 in comm_cost_list]
            if self.model_microbatch_after_dp:
                if None not in nw_res_list_list:
                    res_list = []
                    for res in nw_res_list_list:
                        res_list += res
                    pipeline_cost = pipeline_costmodel(self.timecost_model, self.layer_num, self.model_args_list, self.train_args_list, self.parallel_args_list, self.profile_model_args_list, self.profile_hardware_args_list, res_list, pp_stage_list, chunks, bsz, min_tp, other_time_cost[1][k], self.logger)
                    # print(sum(comm_cost_list),pipeline_cost)
                    # print(pp_stage_list, res_list_list)
                    if comm_cost > pipeline_cost:
                        comm_cost = pipeline_cost
                        vtp = k
            else:
                comm_cost = sum(nw_comm_cost_list)
        if vtp != -1:
            res_list_list = [v2[vtp] for v2 in res_list_list]
            mem_remain_list = [v2[vtp] for v2 in mem_remain_list]
            mem_cost_list = [v2[vtp] for v2 in mem_cost_list]
        else:
            res_list_list, mem_remain_list, mem_cost_list = None, [-1 for v2 in mem_remain_list], [-1 for v2 in mem_cost_list]
        return comm_cost, res_list_list, mem_remain_list, mem_cost_list, vtp, best_strategy_flag, from_history

    def fit(self, bsz, min_tp, max_tp, vsp, embed_sdp, sp_search, print_=True, mbsz_dict=None):
        min_comm_cost = np.inf
        min_res_list = None
        min_pp_deg = -1
        min_mem_remain = -1
        min_mem_cost = -1
        min_vtp = -1
        if mbsz_dict == None:
            mbsz_dict = {}
            for pp_deg in self.ppdeg_set:
                mbsz_dict[pp_deg] = 8

        for pp_deg in self.ppdeg_set:
            if pp_deg * min_tp > self.gpu_num:
                continue
            if print_:
                if self.logger is not None:
                    self.logger.info(f'bsz={bsz}, pp_deg={pp_deg}, min_tp={min_tp}, max_tp={max_tp}, vsp={vsp}, embed_sdp={embed_sdp}, sp_search={sp_search}:')
                else:
                    print(f'bsz={bsz}, pp_deg={pp_deg}, min_tp={min_tp}, max_tp={max_tp}, vsp={vsp}, embed_sdp={embed_sdp}, sp_search={sp_search}:', flush=True)
            if bsz % (self.gpu_num//(pp_deg*min_tp)):
                comm_cost, res_list, mem_remain, mem_cost, best_strategy_flag, from_history = np.inf, None, -1, np.inf, False, False
                if min_res_list is None:
                    min_res_list = '[current bsz is not divisible by bsz_scale]'
                if print_:
                    if self.logger is not None:
                        self.logger.info(f'Best strategy: {best_strategy_flag} \nFrom history: {from_history}')
                        self.logger.info(f'time cost: {comm_cost}, memory remaining: {mem_remain}, memory cost: {mem_cost}')
                    else:
                        print(f'Best strategy: {best_strategy_flag} \nFrom history: {from_history}')
                        print(f'time cost: {comm_cost}, memory remaining: {mem_remain}, memory cost: {mem_cost}')
                continue
            assert self.multi_layer_type
            comm_cost, res_list, mem_remain, mem_cost, vtp, best_strategy_flag, from_history = self._build_dp_and_run_multi_layer_type(pp_deg, bsz, mbsz_dict[pp_deg], min_tp, max_tp, vsp, embed_sdp, sp_search)
            mem_cost = [m + self.mem_cache for m in mem_cost] if isinstance(mem_cost, list) else mem_cost + self.mem_cache
            if print_:
                if self.logger is not None:
                    self.logger.info(f'Best strategy: {best_strategy_flag} \nFrom history: {from_history}')
                    self.logger.info(f'time cost: {comm_cost}, memory remaining: {mem_remain}, memory cost: {mem_cost}')
                else:
                    print(f'Best strategy: {best_strategy_flag} \nFrom history: {from_history}')
                    print(f'time cost: {comm_cost}, memory remaining: {mem_remain}, memory cost: {mem_cost}')
            if min_comm_cost > comm_cost:
                min_res_list = res_list
                min_comm_cost = comm_cost
                min_pp_deg = pp_deg
                min_mem_remain = mem_remain
                min_mem_cost = mem_cost
                min_vtp = vtp

        return min_comm_cost, min_res_list, min_pp_deg, min_mem_remain, min_mem_cost, min_vtp