import copy
import os
from collections import defaultdict
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from galvatron.utils.config_utils import array2str, num2str, read_json_config, str2array, write_json_config

from .base_profiler import BaseProfiler


class ModelProfiler(BaseProfiler):
    """Model profiler for analyzing model performance characteristics including computation and memory usage"""

    def __init__(self, args):
        """Initialize model profiler

        Args:
            args: Arguments containing profiling configuration including:
                - profile_mode: Profiling mode ('static', 'batch', or 'sequence')
                - profile_type: Type of profiling ('computation' or 'memory')
                - profile_batch_size: Batch size for static profiling
                - profile_min/max_batch_size: Range for batch size profiling
                - profile_min/max_seq_length: Range for sequence length profiling
                - profile_batch/seq_length_step: Step size for profiling
        """
        super().__init__(args)

    def set_profiler_launcher(
        self,
        path: str,
        layernum_arg_names: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        seqlen_arg_names: Optional[List[str]] = None,
        layernum_listed: bool = False,
    ) -> None:
        """Set up profiler launcher configuration

        Args:
            path: Path to profiling scripts
            layernum_arg_names: Names of arguments specifying number of layers
            model_name: Name of the model being profiled
            seqlen_arg_names: Names of arguments specifying sequence lengths
            layernum_listed: Whether layer numbers are provided as a list
        """
        self.set_path(path)
        self.set_model_name(model_name)
        self.set_layernum_list(layernum_arg_names, layernum_listed)
        self.set_seqlen_list()
        self.set_seqlen_arg_names(seqlen_arg_names)

    def set_layernum_list(self, layernum_arg_names: Optional[List[str]], layernum_listed: bool) -> None:
        """Set layer number argument names and process layer numbers

        Args:
            layernum_arg_names: Names of arguments specifying number of layers
            layernum_listed: Whether layer numbers are provided as a list

        Note:
            For models like Swin Transformer, layer numbers might be provided as a list
        """
        self.layernum_listed = layernum_listed
        self.layernum_arg_names = layernum_arg_names

        if not self.layernum_listed:
            # Regular case: each argument specifies one layer type
            self.num_layertype = len(layernum_arg_names)
            self.layernum_list = [getattr(self.args, name) for name in layernum_arg_names]
        else:
            # Special case: list-format layernum args like Swin `depths`
            assert len(layernum_arg_names) == 1
            self.num_layertype = sum([len(getattr(self.args, name)) for name in layernum_arg_names])
            self.layernum_list = [layernum for name in layernum_arg_names for layernum in getattr(self.args, name)]

    def set_seqlen_list(self) -> None:
        """Set sequence length list based on profiling configuration

        This method handles different profiling modes:
        - static: Fixed sequence length
        - batch: Batch size variation
        - sequence: Sequence length variation

        Note:
            For memory profiling in sequence mode, sequence lengths must be powers of 2
        """
        args = self.args
        self.sequence_length_list = []

        if self.args.profile_seq_length_list is not None:
            self.profile_seq_length_list = str2array(self.args.profile_seq_length_list)
            assert len(self.profile_seq_length_list) == self.num_layertype
        else:
            # for swin model
            if self.layernum_listed:
                self.profile_seq_length_list = []
                for i in range(len(self.args.depths)):
                    seq_len = (self.args.image_size // self.args.patch_size // (2 ** i)) ** 2
                    self.profile_seq_length_list.append(seq_len)

        for i in range(self.num_layertype):
            if args.profile_mode == "static":
                self.sequence_length_list.append([self.profile_seq_length_list[i]])
            elif args.profile_mode == "batch":
                self.sequence_length_list.append([self.profile_seq_length_list[i]])
            elif args.profile_mode == "sequence":
                if self.num_layertype > 1:
                    assert False, "Sequence profiling only support single layertype!"
                if args.profile_type == "computation":
                    assert args.profile_min_seq_length is not None and args.profile_max_seq_length is not None
                    self.sequence_length_list.append(
                        list(
                            range(
                                args.profile_min_seq_length,
                                args.profile_max_seq_length + 1,
                                args.profile_seq_length_step,
                            )
                        )
                    )
                elif args.profile_type == "memory":
                    assert args.profile_min_seq_length is not None and args.profile_max_seq_length is not None
                    # For memory profiling, sequence lengths must be powers of 2
                    assert (
                        1 << (args.profile_min_seq_length.bit_length() - 1)
                    ) == args.profile_min_seq_length, "profile_min_seq_length must be a power of 2"
                    assert (
                        1 << (args.profile_max_seq_length.bit_length() - 1)
                    ) == args.profile_max_seq_length, "profile_max_seq_length must be a power of 2"
                    self.sequence_length_list.append(
                        [
                            (1 << j)
                            for j in range(
                                args.profile_min_seq_length.bit_length() - 1, args.profile_max_seq_length.bit_length()
                            )
                        ]
                    )

    def set_seqlen_arg_names(self, seqlen_arg_names: Optional[List[str]]) -> None:
        """Set sequence length argument names

        Args:
            seqlen_arg_names: Names of arguments specifying sequence lengths.
                            If None, defaults to ['seq_length']
        """
        if seqlen_arg_names is None:
            self.seqlen_arg_names = ["seq_length"]
        else:
            self.seqlen_arg_names = seqlen_arg_names

    # =============== For Launching Profiling Scripts ===============
    def get_bsz_list(self) -> List[int]:
        """Get list of batch sizes for profiling

        Returns:
            List[int]: List of batch sizes based on profiling mode:
                - static: Single batch size
                - batch: Range of batch sizes
                - sequence: Fixed batch size with sequence length variation

        Raises:
            AssertionError: If required batch size parameters are not set
        """
        args = self.args
        if hasattr(self, "batch_size_list"):
            return self.batch_size_list

        if args.profile_mode == "static":
            assert args.profile_batch_size is not None
            self.batch_size_list = [args.profile_batch_size]
        elif args.profile_mode == "batch":
            assert args.profile_min_batch_size is not None and args.profile_max_batch_size is not None
            self.batch_size_list = list(
                range(args.profile_min_batch_size, args.profile_max_batch_size + 1, args.profile_batch_size_step)
            )
        elif args.profile_mode == "sequence":
            self.batch_size_list = [args.profile_batch_size]

        return self.batch_size_list

    def launch_profiling_scripts(self) -> None:
        """Launch profiling scripts for memory or computation profiling

        This method handles:
        1. Memory profiling with different tensor parallelism and pipeline parallelism settings
        2. Computation profiling with different batch sizes and sequence lengths

        Note:
            Memory profiling only supports sequence or static profile modes
        """
        args = self.args
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        MODEL_ARGS, PROFILE_ARGS, LAUNCH_SCRIPTS, world_size, layernum_lists = self.prepare_launch_args()

        if args.profile_type == "memory":
            self._launch_memory_profiling(MODEL_ARGS, PROFILE_ARGS, LAUNCH_SCRIPTS, world_size, layernum_lists)
        elif args.profile_type == "computation":
            self._launch_computation_profiling(MODEL_ARGS, PROFILE_ARGS, LAUNCH_SCRIPTS, layernum_lists)

    def _launch_memory_profiling(
        self, MODEL_ARGS: str, PROFILE_ARGS: str, LAUNCH_SCRIPTS: str, world_size: int, layernum_lists: List[List[int]]
    ) -> None:
        """Launch memory profiling scripts

        Args:
            MODEL_ARGS: Model configuration arguments
            PROFILE_ARGS: Profiling configuration arguments
            LAUNCH_SCRIPTS: Launch script path
            world_size: Total number of GPUs
            layernum_lists: Lists of layer numbers for different configurations

        Note:
            Only supports sequence or static profile modes
        """
        args = self.args
        assert (
            args.profile_mode == "static" or args.profile_mode == "sequence"
        ), "Memory profiling only supports sequence or static profile mode."
        max_tp_deg = min(world_size, self.args.max_tp_deg)
        if args.profile_mode != "static":
            max_tp_deg = 1
        sequence_length_list = list(product(*self.sequence_length_list))
        for seq in sequence_length_list:
            PROFILE_ARGS = self.prepare_profile_args()
            pp_deg = 1
            for checkpoint in [0, 1]:
                tp_deg = 1
                while tp_deg <= max_tp_deg:
                    if pp_deg * tp_deg <= world_size:
                        for enable_vocab_tp in [0, 1]:
                            if tp_deg == 1 and enable_vocab_tp == 1:
                                continue
                            for layernum_list in layernum_lists:
                                args_ = {}
                                self.get_layernum_args(args_, layernum_list)
                                self.get_seqlen_args(args_, seq)
                                args_["pp_deg"] = pp_deg
                                args_["global_tp_deg"] = tp_deg
                                args_["global_checkpoint"] = checkpoint
                                args_["vocab_tp"] = tp_deg if enable_vocab_tp == 1 else 1
                                ARGS_ = self.args2str(args_)
                                CMD = LAUNCH_SCRIPTS + MODEL_ARGS + PROFILE_ARGS + ARGS_
                                print(CMD)
                                os.system(CMD)
                    if checkpoint:
                        break
                    tp_deg *= 2

            for pp_deg in [2, 4]:
                layernum = pp_deg
                tp_deg = 1
                while tp_deg <= max_tp_deg:
                    if pp_deg * tp_deg <= world_size:
                        for enable_vocab_tp in [0, 1]:
                            if tp_deg == 1 and enable_vocab_tp == 1:
                                continue
                            args_ = {}
                            self.get_layernum_args(args_, [layernum] * self.num_layertype)
                            self.get_seqlen_args(args_, seq)
                            args_["pp_deg"] = pp_deg
                            args_["global_tp_deg"] = tp_deg
                            args_["global_checkpoint"] = 0
                            args_["vocab_tp"] = tp_deg if enable_vocab_tp == 1 else 1
                            ARGS_ = self.args2str(args_)
                            CMD = LAUNCH_SCRIPTS + MODEL_ARGS + PROFILE_ARGS + ARGS_
                            print(CMD)
                            os.system(CMD)
                    tp_deg *= 2

    def _launch_computation_profiling(
        self, MODEL_ARGS: str, PROFILE_ARGS: str, LAUNCH_SCRIPTS: str, layernum_lists: List[List[int]]
    ) -> None:
        """Launch computation profiling scripts

        Args:
            MODEL_ARGS: Model configuration arguments
            PROFILE_ARGS: Profiling configuration arguments
            LAUNCH_SCRIPTS: Launch script path
            layernum_lists: Lists of layer numbers for different configurations

        Note:
            Supports all profile modes (static, batch, sequence)
        """
        for layernum_list in layernum_lists:
            args_ = {}
            self.get_layernum_args(args_, layernum_list)
            args_["pp_deg"] = 1
            args_["global_tp_deg"] = 1
            args_["global_checkpoint"] = 0
            batch_size_list = self.get_bsz_list()
            sequence_length_list = list(product(*self.sequence_length_list))
            for bsz in batch_size_list:
                for seq in sequence_length_list:
                    PROFILE_ARGS = self.prepare_profile_args(batch_size=bsz)
                    self.get_seqlen_args(args_, seq)
                    ARGS_ = self.args2str(args_)
                    CMD = LAUNCH_SCRIPTS + MODEL_ARGS + PROFILE_ARGS + ARGS_
                    print(CMD)
                    os.system(CMD)

    # =============== For Processing Profiled Memory and Time ===============
    def process_profiled_data(self) -> None:
        """Process profiled data for both computation and memory profiling

        This method handles two types of profiling data:
        1. Computation profiling:
            - Calculates average computation time per layer type
            - Processes batch size and sequence length variations
            - Accounts for other computation overhead

        2. Memory profiling:
            - Processes parameter and activation memory usage
            - Handles different parallelism strategies (TP, PP)
            - Calculates memory overhead for different configurations

        The results are written to corresponding config files:
        - Computation results: time_config_path
        - Memory results: memory_config_path
        """
        _, _, _, world_size, layernum_lists = self.prepare_launch_args()
        args = self.args

        if args.profile_type == "computation":
            self._process_computation_data(layernum_lists)
        elif args.profile_type == "memory":
            self._process_memory_data(world_size, layernum_lists)

    def _process_computation_data(self, layernum_lists: List[List[int]]) -> None:
        """Process computation profiling data

        Args:
            layernum_lists: Lists of layer numbers for different configurations

        This method:
        1. Reads profiled computation time data
        2. Calculates per-layer computation time for each layer type
        3. Processes results for different batch sizes and sequence lengths
        4. Writes processed results to config file
        """
        time_config_path = self.time_profiling_path()
        config = read_json_config(time_config_path)
        batch_size_list = self.get_bsz_list()
        sequence_length_list = list(product(*self.sequence_length_list))

        for bsz in batch_size_list:
            for seq in sequence_length_list:
                # Process base configuration
                seq_info = num2str(list(seq), "seq")
                key_base = self.key_format(layernum_lists[0], bsz, seq_info)
                val_base = config[key_base]
                total_avg_time = []

                # Calculate per-layer computation time for each layer type
                for idx, layernum in enumerate(layernum_lists[1:]):
                    key = self.key_format(layernum, bsz, seq_info)
                    val = config[key]
                    avg_time = (val - val_base) / bsz / (self.args.layernum_max - self.args.layernum_min)
                    write_key = f"layertype_{idx}_bsz{bsz}_seq{seq[idx]}"
                    config[write_key] = avg_time
                    total_avg_time.append(avg_time)

                # Calculate other computation overhead
                other_time = val_base
                for idx in range(len(total_avg_time)):
                    other_time -= layernum_lists[0][idx] * total_avg_time[idx] * bsz
                other_time /= bsz
                write_key = f"layertype_other_bsz{bsz}_{seq_info}"
                config[write_key] = max(other_time, 0)

                # Write results to config file
                write_json_config(config, time_config_path)
                print(f"Already written processed computation time into env config file {time_config_path}!\n")

    def _process_memory_data(self, world_size: int, layernum_lists: List[List[int]]) -> None:
        """Process memory profiling data

        Args:
            world_size: Total number of GPUs
            layernum_lists: Lists of layer numbers for different configurations

        This method:
        1. Processes parameter and activation memory usage
        2. Handles different parallelism strategies:
            - Tensor Parallelism (TP)
            - Pipeline Parallelism (PP)
            - Sequence Parallelism (SP)
        3. Calculates memory overhead for different configurations
        4. Writes processed results to config file

        Note:
            Only supports sequence or static profile modes
        """
        assert (
            self.args.profile_mode == "static" or self.args.profile_mode == "sequence"
        ), "Memory profiling only support sequence or static profile mode."

        memory_config_path = self.memory_profiling_path()
        config = read_json_config(memory_config_path)

        # Initialize parameters
        bsz = self.args.profile_batch_size
        layernum_list_base = layernum_lists[0]
        layertype = len(layernum_list_base)
        layernum_lists = layernum_lists[1:]
        layernum_diff = self.args.layernum_max - self.args.layernum_min

        # Process each sequence length configuration
        sequence_length_list = list(product(*self.sequence_length_list))
        for seq in sequence_length_list:
            self._process_single_sequence_config(
                seq, world_size, layernum_list_base, layertype, layernum_lists, layernum_diff, bsz, config
            )

        # Write final results
        write_json_config(config, memory_config_path)

    def _process_single_sequence_config(
        self,
        seq: Tuple[int, ...],
        world_size: int,
        layernum_list_base: List[int],
        layertype: int,
        layernum_lists: List[List[int]],
        layernum_diff: int,
        bsz: int,
        config: Dict,
    ) -> None:
        """Process memory profiling data for a single sequence length configuration

        Args:
            seq: Tuple of sequence lengths for each layer type
            world_size: Total number of GPUs
            layernum_list_base: Base layer numbers for each layer type
            layertype: Number of layer types
            layernum_lists: Lists of layer numbers for different configurations
            layernum_diff: Difference between max and min layer numbers
            bsz: Batch size
            config: Configuration dictionary to store results

        This method processes:
        1. Parameter memory usage for different TP degrees
        2. Activation memory usage with and without checkpointing
        3. Memory overhead for different parallelism strategies
        4. Pipeline parallelism memory costs
        """
        seq_info = num2str(list(seq), "seq")
        print(f"Processing sequence length: {seq_info}")

        # Initialize result containers
        param_result_list = [dict() for _ in range(layertype)]
        act_result_list = [dict() for _ in range(layertype)]
        param_list = [-1] * layertype

        # Process tensor parallelism memory costs
        pp_deg, tp_deg = 1, 1
        while pp_deg * tp_deg <= world_size:
            strategy = f"{pp_deg}_{tp_deg}_{world_size//pp_deg//tp_deg}"
            if self.args.sequence_parallel:
                strategy += "_sp"

            if strategy in config:
                re = config[strategy]
                # Calculate memory costs for each layer type
                for l in range(layertype):
                    layernum_key_0 = layernum_list_base
                    layernum_key_1 = layernum_lists[l]

                    # Calculate parameter memory per layer
                    param_per_layer = (
                        (
                            re[self.key_format(layernum_key_1, bsz, seq_info, 0, "ms")]
                            - re[self.key_format(layernum_key_0, bsz, seq_info, 0, "ms")]
                        )
                        / layernum_diff
                        * pp_deg
                        / 4
                    )

                    # Calculate activation memory per sample
                    act_per_layer_per_sample = (
                        (
                            re[self.key_format(layernum_key_1, bsz, seq_info, 0, "act")]
                            - re[self.key_format(layernum_key_0, bsz, seq_info, 0, "act")]
                        )
                        / layernum_diff
                        * pp_deg
                        / (pp_deg * tp_deg)
                    )
                    act_per_layer_per_sample *= world_size / bsz

                    # Adjust for ZeRO-3
                    if self.args.profile_dp_type == "zero3":
                        param_per_layer *= world_size // pp_deg // tp_deg

                    # Update results
                    param_result_list[l][tp_deg] = param_per_layer
                    act_result_list[l][tp_deg] = act_per_layer_per_sample
                    param_list[l] = max(param_list[l], param_per_layer * tp_deg)

            tp_deg *= 2

        for l in range(layertype):
            print(f"layertype {l}:")
            print(f"param: {param_list[l]}")
            print(f"act_dict: {act_result_list[l]}")
        # Process checkpoint memory costs
        act_dict_c_list = [dict() for _ in range(layertype)]
        act_cpt_list = [-1] * layertype

        pp_deg, tp_deg = 1, 1
        while pp_deg * tp_deg <= world_size:
            strategy = f"{pp_deg}_{tp_deg}_{world_size//pp_deg//tp_deg}_c"
            if self.args.sequence_parallel:
                strategy += "_sp"

            if strategy in config:
                re = config[strategy]
                for l in range(layertype):
                    layernum_key_0 = layernum_list_base
                    layernum_key_1 = layernum_lists[l]

                    # Calculate activation memory with checkpointing
                    act_per_layer_per_sample = (
                        (
                            re[self.key_format(layernum_key_1, bsz, seq_info, 0, "act")]
                            - re[self.key_format(layernum_key_0, bsz, seq_info, 0, "act")]
                        )
                        / layernum_diff
                        * pp_deg
                        / (pp_deg * tp_deg)
                    )
                    act_per_layer_per_sample *= world_size / bsz

                    act_dict_c_list[l][tp_deg] = act_per_layer_per_sample
                    act_cpt_list[l] = max(act_cpt_list[l], act_per_layer_per_sample)

            tp_deg *= 2

        # Update activation results with checkpoint information
        for l in range(layertype):
            print(f"layertype {l}:")
            print(f"act_dict_c: {act_dict_c_list[l]}")
            print(f"act_cpt: {act_cpt_list[l]}")
            act_result_list[l]["checkpoint"] = act_cpt_list[l]

        # Process pipeline parallelism memory costs
        inf = 1e6
        other_memory_pp_off = {"model_states": defaultdict(lambda: inf), "activation": defaultdict(lambda: inf)}
        other_memory_pp_on_first = {"model_states": defaultdict(lambda: inf), "activation": defaultdict(lambda: inf)}
        other_memory_pp_on_last = {"model_states": defaultdict(lambda: inf), "activation": defaultdict(lambda: inf)}

        pp_deg = 1
        while pp_deg <= world_size:
            tp_deg = 1
            while pp_deg * tp_deg <= world_size:
                # Process different vocabulary parallelism configurations
                for enable_vocab_tp in [0, 1]:
                    if tp_deg == 1 and enable_vocab_tp == 1:
                        continue

                    strategy = f"{pp_deg}_{tp_deg}_{world_size//pp_deg//tp_deg}"
                    if enable_vocab_tp and tp_deg != 1:
                        strategy += "_vtp"
                    if self.args.sequence_parallel:
                        strategy += "_sp"

                    if strategy not in config:
                        continue

                    re = config[strategy]
                    # Calculate memory costs for current configuration
                    layernum = pp_deg if pp_deg > 1 else layernum_list_base[0]
                    layernum_list = [layernum] * layertype if pp_deg > 1 else layernum_list_base

                    # Calculate per-layer memory costs
                    ms_cost = [param_result_list[l][tp_deg] * 4 for l in range(layertype)]
                    act_cost = [act_result_list[l][tp_deg] for l in range(layertype)]

                    # Calculate total memory costs for first and last pipeline stages
                    layer_ms_costs_first = self.total_memcost(pp_deg, layernum, layertype, ms_cost, 0)
                    layer_ms_costs_last = self.total_memcost(pp_deg, layernum, layertype, ms_cost, pp_deg - 1)
                    layer_act_costs_first = self.total_memcost(pp_deg, layernum, layertype, act_cost, 0)
                    layer_act_costs_last = self.total_memcost(pp_deg, layernum, layertype, act_cost, pp_deg - 1)

                    # Calculate other memory costs
                    other_ms_first = re[self.key_format(layernum_list, bsz, seq_info, 0, "ms")] - layer_ms_costs_first
                    other_ms_last = (
                        re[self.key_format(layernum_list, bsz, seq_info, world_size - 1, "ms")] - layer_ms_costs_last
                    )

                    # Adjust for ZeRO-3
                    if self.args.profile_dp_type == "zero3":
                        other_ms_first = (
                            (
                                re[self.key_format(layernum_list, bsz, seq_info, 0, "ms")]
                                - layer_ms_costs_first / (world_size // pp_deg // tp_deg)
                            )
                            * (world_size // pp_deg)
                            / (tp_deg if enable_vocab_tp == 1 else 1)
                        )
                        other_ms_last = (
                            (
                                re[self.key_format(layernum_list, bsz, seq_info, world_size - 1, "ms")]
                                - layer_ms_costs_last / (world_size // pp_deg // tp_deg)
                            )
                            * (world_size // pp_deg)
                            / (tp_deg if enable_vocab_tp == 1 else 1)
                        )
                    # Calculate activation memory peaks
                    act_peak_first = max(
                        re[self.key_format(layernum_list, bsz, seq_info, 0, "act_peak")],
                        re[self.key_format(layernum_list, bsz, seq_info, 0, "act")],
                    )
                    act_peak_last = max(
                        re[self.key_format(layernum_list, bsz, seq_info, world_size - 1, "act_peak")],
                        re[self.key_format(layernum_list, bsz, seq_info, world_size - 1, "act")],
                    )

                    # Calculate other activation memory
                    other_act_first = (
                        act_peak_first - layer_act_costs_first * (bsz / (world_size // (pp_deg * tp_deg)))
                    ) / (bsz / world_size * pp_deg * (tp_deg if enable_vocab_tp else 1))
                    other_act_last = (
                        act_peak_last - layer_act_costs_last * (bsz / (world_size // (pp_deg * tp_deg)))
                    ) / (bsz / world_size * pp_deg * (tp_deg if enable_vocab_tp else 1))

                    # Ensure non-negative values
                    other_ms_first = max(other_ms_first, 0)
                    other_ms_last = max(other_ms_last, 0)
                    other_act_first = max(other_act_first, 0)
                    other_act_last = max(other_act_last, 0)

                    # Update memory dictionaries
                    tp_key = tp_deg if enable_vocab_tp else 1
                    if pp_deg == 1:
                        other_memory_pp_off["model_states"][tp_key] = min(
                            other_memory_pp_off["model_states"][tp_key], other_ms_first
                        )
                        other_memory_pp_off["activation"][tp_key] = min(
                            other_memory_pp_off["activation"][tp_key], other_act_first
                        )
                    else:
                        other_memory_pp_on_first["model_states"][tp_key] = min(
                            other_memory_pp_on_first["model_states"][tp_key], other_ms_first
                        )
                        other_memory_pp_on_first["activation"][tp_key] = min(
                            other_memory_pp_on_first["activation"][tp_key], other_act_first
                        )
                        other_memory_pp_on_last["model_states"][tp_key] = min(
                            other_memory_pp_on_last["model_states"][tp_key], other_ms_last
                        )
                        other_memory_pp_on_last["activation"][tp_key] = min(
                            other_memory_pp_on_last["activation"][tp_key], other_act_last
                        )

                tp_deg *= 2
            pp_deg *= 2

        # Handle sequence parallelism memory scaling
        if self.args.sequence_parallel:
            for tp in [2, 4, 8]:
                if tp not in act_result_list[0]:
                    act_result_list[0][tp] = act_result_list[0][tp // 2] / 2
                for memory_dict in [other_memory_pp_off, other_memory_pp_on_first, other_memory_pp_on_last]:
                    for key in ["model_states", "activation"]:
                        if tp not in memory_dict[key]:
                            memory_dict[key][tp] = memory_dict[key][tp // 2] / 2

        print("other_memory_pp_off:", other_memory_pp_off)
        print("other_memory_pp_on_first:", other_memory_pp_on_first)
        print("other_memory_pp_on_last:", other_memory_pp_on_last)
        # Store results in config
        config_key = "layertype_%d_sp" if self.args.sequence_parallel else "layertype_%d"
        for l in range(layertype):
            if config_key % l not in config:
                config[config_key % l] = dict()
            config[config_key % l][str(seq[l])] = {
                "parameter_size": param_list[l],
                "tp_activation_per_bsz_dict": act_result_list[l],
            }

        # Store other memory costs
        memory_keys = {
            "other_memory_pp_off": other_memory_pp_off,
            "other_memory_pp_on_first": other_memory_pp_on_first,
            "other_memory_pp_on_last": other_memory_pp_on_last,
        }

        suffix = "_sp" if self.args.sequence_parallel else ""
        for key, value in memory_keys.items():
            config_key = f"{key}{suffix}"
            if config_key not in config:
                config[config_key] = {}
            config[config_key][seq_info[3:]] = copy.deepcopy(value)

    # =============== Util functions ===============
    def key_format(
        self,
        layernum: Union[List[int], int],
        bsz: Optional[int] = None,
        seq: Optional[Union[str, int]] = None,
        rank: Optional[int] = None,
        type: Optional[str] = None,
    ) -> str:
        """Format key for config dictionary

        Args:
            layernum: Layer number or list of layer numbers
            bsz: Batch size (optional)
            seq: Sequence length or sequence info string (optional)
            rank: GPU rank (optional)
            type: Memory type ('ms' for model states or 'act' for activations) (optional)

        Returns:
            str: Formatted key string

        Example:
            >>> key_format([1,2,3], 32, "seq128", 0, "ms")
            "layernum[1,2,3]_bsz32_seq128_rank0_ms"
        """
        if isinstance(layernum, list):
            s = f"layernum[{array2str(layernum)}]"
        else:
            s = f"layernum{layernum}"

        if bsz is not None:
            s += f"_bsz{bsz}"
        if seq is not None:
            if isinstance(seq, str):
                s += f"_{seq}"
            else:
                s += f"_seq{seq}"
        if rank is not None and type is not None:
            s += f"_rank{rank}_{type}"
        return s

    def total_memcost(
        self, pp_deg: int, layernum: int, layertype: int, per_layer_cost: List[float], stage_idx: int
    ) -> float:
        """Calculate total memory cost for a pipeline stage

        Args:
            pp_deg: Pipeline parallelism degree
            layernum: Number of layers per type
            layertype: Number of layer types
            per_layer_cost: Memory cost per layer for each layer type
            stage_idx: Pipeline stage index

        Returns:
            float: Total memory cost for the specified pipeline stage

        Note:
            Assumes equal distribution of layers across pipeline stages
        """
        # Calculate memory cost for each layer
        layer_costs = []
        for l in range(layertype):
            layer_costs.extend([per_layer_cost[l]] * layernum)

        # Calculate layer distribution across pipeline stages
        total_layer_num = layertype * layernum
        avg_layer_num = int(total_layer_num // pp_deg)
        last_layer_num = total_layer_num - avg_layer_num * (pp_deg - 1)
        pp_divide = [avg_layer_num] * (pp_deg - 1) + [last_layer_num]

        # Verify equal distribution
        assert avg_layer_num == last_layer_num

        # Sum memory costs for the specified stage
        start_idx = int(np.sum(pp_divide[:stage_idx]))
        end_idx = int(np.sum(pp_divide[: stage_idx + 1]))
        return np.sum(layer_costs[start_idx:end_idx])

    def prepare_profile_args(self, batch_size: Optional[int] = None) -> str:
        """Prepare profiling arguments string

        Args:
            batch_size: Optional batch size override

        Returns:
            str: Formatted profiling arguments string including extra arguments

        Note:
            Handles extra arguments from args.extra_args_str if present
        """
        profile_args = self.profiling_general_args(batch_size)
        PROFILE_ARGS = self.args2str(profile_args)

        # zsh: Revise to accept extra_args_str
        if hasattr(self.args, "extra_args_str"):
            extra_args_list = self.args.extra_args_str.split("/")
            for arg in extra_args_list:
                if arg != "":
                    PROFILE_ARGS += f" --{arg}"
        return PROFILE_ARGS

    def prepare_launch_args(self) -> Tuple[str, str, str, int, List[List[int]]]:
        """Prepare all arguments needed for launching profiling

        Returns:
            Tuple containing:
            - MODEL_ARGS (str): Model configuration string
            - PROFILE_ARGS (str): Profiling configuration string
            - LAUNCH_SCRIPTS (str): Launch script path
            - world_size (int): Total number of GPUs
            - layernum_lists (List[List[int]]): Lists of layer numbers

        Note:
            Excludes profiling-specific arguments from model arguments
        """
        assert self.layernum_arg_names is not None
        # Define profiling-specific argument names to exclude
        profile_arg_names = [
            "profile_type",
            "set_model_config_manually",
            "set_layernum_manually",
            "set_seqlen_manually",
            "profile_batch_size",
            "profile_min_batch_size",
            "profile_max_batch_size",
            "profile_batch_size_step",
            "profile_seq_length_list",
            "profile_min_seq_length",
            "profile_max_seq_length",
            "profile_seq_length_step",
            "layernum_min",
            "layernum_max",
            "max_tp_deg",
            "profile_dp_type",
            "mixed_precision",
            "use_flash_attn",
            "sequence_parallel",
            "attention_dropout",
            "hidden_dropout",
            "kv_channels",
            "make_vocab_size_divisible_by",
            "padded_vocab_size",
            "ffn_hidden_size",
            "group_query_attention",
            "num_query_groups",
            "add_bias_linear",
            "swiglu",
            "extra_args_str",
            "seq_length",
            "encoder_seq_length",
            "decoder_seq_length",
        ]
        exclude_arg_names = profile_arg_names + self.layernum_arg_names
        MODEL_ARGS = self.args2str(self.args._get_kwargs(), exclude_arg_names)
        # print(MODEL_ARGS)

        PROFILE_ARGS = self.prepare_profile_args()
        # print(PROFILE_ARGS)

        env_args = self.env_args()
        LAUNCH_SCRIPTS = self.launch_scripts(env_args)
        print("Get environment args:", env_args)

        world_size = int(env_args["NUM_NODES"]) * int(env_args["NUM_GPUS_PER_NODE"])

        layernum_lists = self.get_layernum_list_for_profiling()

        return MODEL_ARGS, PROFILE_ARGS, LAUNCH_SCRIPTS, world_size, layernum_lists

    def get_layernum_list_for_profiling(self) -> List[List[int]]:
        """Generate layer number combinations for profiling

        Returns:
            List[List[int]]: List of layer number combinations:
                - First list contains minimum layer numbers for all layer types
                - Subsequent lists vary one layer type to maximum while keeping others at minimum
        """
        layernum_lists = []
        base_list = [self.args.layernum_min] * self.num_layertype
        layernum_lists.append(base_list)

        for idx in range(self.num_layertype):
            l = base_list.copy()
            l[idx] = self.args.layernum_max
            layernum_lists.append(l)
        return layernum_lists

    def argval2str(self, val: Union[List, Any]) -> str:
        """Convert argument value to string format

        Args:
            val: Value to convert, can be a list or single value

        Returns:
            str: Space-separated string for lists, or string representation for single values
        """
        if isinstance(val, list):
            return " ".join(str(i) for i in val).strip()
        return str(val)

    def arg2str(self, key: str, val: Union[List, Any]) -> str:
        """Format single argument as command line parameter

        Args:
            key: Argument name
            val: Argument value

        Returns:
            str: Formatted argument string (e.g., '--key value')
        """
        return f" --{key} {self.argval2str(val)}"

    def args2str(self, args: Union[Dict, List[Tuple]], exclude_args: List[str] = []) -> str:
        """Convert multiple arguments to command line format

        Args:
            args: Dictionary of arguments or list of (key, value) tuples
            exclude_args: List of argument names to exclude

        Returns:
            str: Space-separated string of formatted arguments
        """
        s = ""
        if isinstance(args, dict):
            for key, val in args.items():
                if key not in exclude_args:
                    s += self.arg2str(key, val)
        elif isinstance(args, (list, tuple)) and len(args) > 0 and len(args[0]) == 2:
            for key, val in args:
                if key not in exclude_args:
                    s += self.arg2str(key, val)
        return s

    def profiling_general_args(self, batch_size: Optional[int] = None) -> Dict[str, Union[int, float, str]]:
        """Get general profiling arguments

        Args:
            batch_size: Optional batch size override

        Returns:
            Dict: Dictionary of general profiling arguments including:
                - Model configuration settings
                - Training parameters
                - Profiling flags
                - Parallelism settings
        """
        args = {
            "set_model_config_manually": 0,
            "set_layernum_manually": 1,
            "set_seqlen_manually": 1,
            "global_train_batch_size": self.args.profile_batch_size if batch_size is None else batch_size,
            "epochs": 10,
            "lr": 1e-4,
            "adam_weight_decay": 0.01,
            "dropout_prob": 0.1,
            "check_loss": 0,
            "profile": 1,
            "save_profiled_memory": 1 if self.args.profile_type == "memory" else 0,
            "profile_forward": 1 if self.args.profile_type == "computation" else 0,
            "initialize_on_meta": 1,
            "global_tp_consec": 1,
            "sdp": 1 if self.args.profile_dp_type == "zero3" and self.args.profile_type == "memory" else 0,
            "chunks": 1,
            "pipeline_type": "gpipe",
            "default_dp_type": self.args.profile_dp_type if self.args.profile_type == "memory" else "ddp",
            "mixed_precision": self.args.mixed_precision,
            "shape_order": self.args.shape_order,
        }

        # Add optional flags
        if self.args.use_flash_attn:
            args["use-flash-attn"] = ""
        if self.args.sequence_parallel:
            args["sequence-parallel"] = ""
        return args

    def get_layernum_args(self, args: Dict[str, Any], layernum_list: List[int]) -> None:
        """Set layer number arguments in the configuration dictionary

        Args:
            args: Configuration dictionary to update
            layernum_list: List of layer numbers for each layer type

        Note:
            Handles two formats of layer number arguments:
            1. Individual arguments for each layer type (layernum_listed=False)
            2. Single list argument for all layers (layernum_listed=True, e.g., Swin `depths`)
        """
        assert (
            len(layernum_list) == self.num_layertype
        ), f"Expected {self.num_layertype} layer numbers, got {len(layernum_list)}"

        if not self.layernum_listed:
            # Set individual arguments for each layer type
            for layernum, arg_name in zip(layernum_list, self.layernum_arg_names):
                args[arg_name] = layernum
        else:
            # Set single list argument for all layers (e.g., Swin Transformer depths)
            assert len(self.layernum_arg_names) == 1, "List format requires exactly one argument name"
            arg_name = self.layernum_arg_names[0]
            args[arg_name] = layernum_list

    def get_seqlen_args(self, args: Dict[str, Any], seqlen_list: List[int]) -> None:
        """Set sequence length arguments in the configuration dictionary

        Args:
            args: Configuration dictionary to update
            seqlen_list: List of sequence lengths for each layer type
        """
        assert (
            len(seqlen_list) == self.num_layertype
        ), f"Expected {self.num_layertype} sequence lengths, got {len(seqlen_list)}"

        # Swin model does not support set seqlen manually
        if self.seqlen_arg_names is None:
            return
        
        for seqlen, arg_name in zip(seqlen_list, self.seqlen_arg_names):
            args[arg_name] = seqlen

    def env_args(self) -> Dict[str, Union[str, int]]:
        """Get environment configuration arguments

        Returns:
            Dict: Dictionary of environment variables with defaults:
                - PROFILE_LAUNCHER: Launcher command
                - PROFILE_TRAINER: Trainer script path
                - NUM_NODES: Number of nodes
                - NUM_GPUS_PER_NODE: GPUs per node
                - MASTER_ADDR/PORT: Communication settings
                - NCCL settings
        """
        return {
            "PROFILE_LAUNCHER": os.getenv("PROFILE_LAUNCHER", "python3 -m torch.distributed.launch"),
            "PROFILE_TRAINER": os.getenv("PROFILE_TRAINER", "train_dist.py"),
            "NUM_NODES": os.getenv("NUM_NODES", "1") if self.args.profile_type == "memory" else "1",
            "NUM_GPUS_PER_NODE": os.getenv("NUM_GPUS_PER_NODE", "8") if self.args.profile_type == "memory" else "1",
            "MASTER_ADDR": os.getenv("MASTER_ADDR", ""),
            "MASTER_PORT": os.getenv("MASTER_PORT", ""),
            "NCCL_SOCKET_IFNAME": os.getenv("NCCL_SOCKET_IFNAME", ""),
            "NODE_RANK": os.getenv("NODE_RANK", "0"),
        }

    def launch_scripts(self, env_args: Dict[str, str]) -> str:
        """Generate launch script command

        Args:
            env_args: Dictionary of environment arguments

        Returns:
            str: Formatted launch command string

        Note:
            Currently uses simplified launch command without node configuration
        """
        return f"{env_args['PROFILE_LAUNCHER']} {env_args['PROFILE_TRAINER']}"
