import os
from typing import List, Tuple, Union

import numpy as np

from galvatron.utils.config_utils import read_json_config, write_json_config

from .base_profiler import BaseProfiler


class HardwareProfiler(BaseProfiler):
    """Hardware profiler for analyzing communication bandwidth and other hardware characteristics"""

    def __init__(self, args):
        """Initialize hardware profiler

        Args:
            args: Arguments containing profiling configuration including:
            if backend == "nccl":
                - num_nodes: Number of nodes
                - num_gpus_per_node: Number of GPUs per node
                - nccl_test_dir: Directory of nccl-test
                - mpi_path: Path to MPI
                - start_mb: Starting communication size in MB
                - end_mb: Ending communication size in MB
                - scale: Memory scale of nccl-test
                - hostfile: Hostfile for nccl-test
            else:
                - num_nodes: Number of nodes
                - num_gpus_per_node: Number of GPUs per node
                - master_addr: Master node address
                - master_port: Master node port
                - node_rank: Current node rank
        """
        super().__init__(args)
        self.path = None

    def get_env(self) -> str:
        """Get environment configuration as string

        Returns:
            str: Environment configuration string with export commands
        """
        env = {
            "NUM_NODES": self.args.num_nodes,
            "NUM_GPUS_PER_NODE": self.args.num_gpus_per_node,
            "MASTER_ADDR": self.args.master_addr,
            "MASTER_PORT": self.args.master_port,
            "NODE_RANK": self.args.node_rank,
        }
        env_str = "\n".join([k for k in self.args.envs]) + "\n"
        env_str += "\n".join([f"export {k}={v}" for k, v in env.items()]) + "\n"

        return env_str

    def generate_script(self, num_nodes: int, num_gpus_per_node: int) -> None:
        """Generate test scripts for allreduce and p2p communication

        Args:
            num_nodes: Number of nodes to use
            num_gpus_per_node: Number of GPUs per node
        """
        world_size = num_nodes * num_gpus_per_node
        env = self.get_env()

        print("Generating allreduce test script...")

        # Generate allreduce test script
        def allreduce_script(allreduce_size: int, allreduce_consec: int) -> str:
            return (
                "python -m torch.distributed.launch "
                f"--nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE "
                "--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT "
                f"--node_rank=$NODE_RANK profile_allreduce.py "
                f"--global_tp_deg {allreduce_size} --global_tp_consec {allreduce_consec} "
                "--pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE \n"
            )

        # Write allreduce test script
        config_dir = os.path.join(self.path, "./scripts")
        with open(os.path.join(config_dir, "profile_allreduce.sh"), "w") as f:
            f.write(env)
            allreduce_size = num_nodes * num_gpus_per_node
            while allreduce_size > 1:
                for allreduce_consec in [1, 0]:
                    if world_size == allreduce_size and allreduce_consec == 0:
                        continue
                    script = allreduce_script(allreduce_size, allreduce_consec)
                    f.write(f'echo "Running: {script}"\n')
                    f.write(script)
                allreduce_size //= 2
                f.write("sleep 1\n")

        print("Generating p2p test script...")

        # Generate p2p test script
        def p2p_script(pp_deg: int) -> str:
            return (
                "python -m torch.distributed.launch "
                f"--nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE "
                "--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT "
                f"--node_rank=$NODE_RANK profile_p2p.py "
                f"--global_tp_deg 1 --global_tp_consec 1 --pp_deg {pp_deg} "
                "--nproc_per_node=$NUM_GPUS_PER_NODE \n"
            )

        # Write p2p test script
        with open(os.path.join(config_dir, "profile_p2p.sh"), "w") as f:
            f.write(env)
            pp_deg = 2
            while pp_deg <= world_size and pp_deg <= self.args.max_pp_deg:
                script = p2p_script(pp_deg)
                f.write(f'echo "Running: {script}"\n')
                f.write(script)
                pp_deg *= 2
                f.write("sleep 1\n")

    def generate_sp_script(self, num_nodes: int, num_gpus_per_node: int) -> None:
        """Generate test scripts for allreduce and all2all communication

        Args:
            num_nodes: Number of nodes to use
            num_gpus_per_node: Number of GPUs per node
        """
        env = self.get_env()

        print("Generating allreduce test script...")

        def allreduce_script(allreduce_size: int, allreduce_consec: int, buffer_size: int) -> str:
            return (
                "python -m torch.distributed.launch "
                "--nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE "
                "--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT "
                "--node_rank=$NODE_RANK profile_allreduce.py "
                f"--global_tp_deg {allreduce_size} --global_tp_consec {allreduce_consec} "
                f"--pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE "
                f"--local_batch_size {buffer_size} --profile_time 1\n"
            )

        args = self.args
        config_dir = os.path.join(self.path, "./scripts")

        # Write allreduce test script with sequence parallelism
        with open(os.path.join(config_dir, "profile_allreduce_sp.sh"), "w") as f:
            f.write(env)
            allreduce_size = min(num_nodes * num_gpus_per_node, args.max_tp_size)
            while allreduce_size > 1:
                buffer_size = 1024
                while buffer_size >= 1:
                    script = allreduce_script(allreduce_size, 1, buffer_size)
                    f.write(f'echo "Running: {script}"\n')
                    f.write(script)
                    f.write("sleep 1\n")
                    buffer_size //= 2
                allreduce_size //= 2

        print("Generating all2all test script...")

        def all2all_script(allreduce_size: int, allreduce_consec: int, buffer_size: int) -> str:
            return (
                "python -m torch.distributed.launch "
                "--nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE "
                "--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT "
                "--node_rank=$NODE_RANK profile_all2all.py "
                f"--global_tp_deg {allreduce_size} --global_tp_consec {allreduce_consec} "
                f"--pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE "
                f"--local_batch_size {buffer_size} --profile_time 1\n"
            )

        # Write all-to-all test script
        with open(os.path.join(config_dir, "profile_all2all_sp.sh"), "w") as f:
            f.write(env)
            all2all_size = min(num_nodes * num_gpus_per_node, args.max_tp_size)
            while all2all_size > 1:
                buffer_size = 1024
                while buffer_size >= 1:
                    script = all2all_script(all2all_size, 1, buffer_size)
                    f.write(f'echo "Running: {script}"\n')
                    f.write(script)
                    f.write("sleep 1\n")
                    buffer_size //= 2
                all2all_size //= 2

    def profile_bandwidth(self, backend: str = "nccl") -> None:
        """Profile communication bandwidth between devices

        This method profiles both allreduce and point-to-point communication bandwidth.
        Results are saved to hardware config files.

        Args:
            backend: Communication backend to use ("nccl" or "torch")

        Note:
            For NCCL backend, uses nccl-tests to measure bandwidth
            For torch backend, generates test scripts for later execution
        """
        args = self.args
        world_size = args.num_nodes * args.num_gpus_per_node
        if backend != "nccl":
            self.generate_script(args.num_nodes, args.num_gpus_per_node)
            # import os
            # os.system('sh scripts/allreduce_scrpit.sh')
            # os.system('sh scripts/p2p_scrpit.sh')
            return

        # Create hardware config directory
        hardware_config_dir = os.path.join(self.path, "./hardware_configs")
        os.makedirs(hardware_config_dir, exist_ok=True)

        # Profile allreduce bandwidth
        nccl_file = "build/all_reduce_perf"
        ARGS = self.prepare_nccltest_args(nccl_file)
        hardware_config_file = "allreduce_bandwidth_%dnodes_%dgpus_per_node.json" % (
            args.num_nodes,
            args.num_gpus_per_node,
        )
        hardware_config_path = os.path.join(hardware_config_dir, hardware_config_file)
        allreduce_size = world_size
        while allreduce_size > 1:
            for is_consecutive in [True, False]:
                if world_size == allreduce_size and not is_consecutive:
                    continue
                print(
                    "============= allreduce_size: %d, allreduce_consec: %d ============="
                    % (allreduce_size, is_consecutive)
                )
                allreduce_groups = self.generate_allreduce_groups(world_size, allreduce_size, is_consecutive)
                bandwidth = self.launch_nccl_test(allreduce_groups, args.num_gpus_per_node, ARGS)
                key = "allreduce_size_%d_consec_%d" % (allreduce_size, is_consecutive)
                self.write_config(hardware_config_path, key, bandwidth)
                print("=" * 70, "\n")
            allreduce_size //= 2

        # Profile p2p bandwidth
        nccl_file = "build/sendrecv_perf"
        ARGS = self.prepare_nccltest_args(nccl_file)
        hardware_config_file = "p2p_bandwidth_%dnodes_%dgpus_per_node.json" % (args.num_nodes, args.num_gpus_per_node)
        hardware_config_path = os.path.join(hardware_config_dir, hardware_config_file)
        pp_deg = 2
        while pp_deg <= world_size and pp_deg <= args.max_pp_deg:
            print("============= pp_size: %d =============" % (pp_deg))
            p2p_groups = self.generate_p2p_groups(world_size, pp_deg)
            bandwidth = self.launch_nccl_test(p2p_groups, args.num_gpus_per_node, ARGS)
            key = "pp_size_%d" % pp_deg
            self.write_config(hardware_config_path, key, bandwidth)
            print("=" * 70, "\n")
            pp_deg *= 2

        os.system("rm -rf %s" % (os.path.join(self.path, "nccl_test.log")))

    def profile_sp_bandwidth(self, backend="nccl"):
        """Profile bandwidth for sequence parallelism

        This method profiles both allreduce and all-to-all communication bandwidth
        with different buffer sizes for sequence parallelism.

        Args:
            backend: Communication backend to use ("nccl" or "torch")

        Note:
            For NCCL backend, uses nccl-tests to measure bandwidth
            For torch backend, generates test scripts for later execution
        """
        args = self.args
        world_size = args.num_nodes * args.num_gpus_per_node
        if backend != "nccl":
            self.generate_sp_script(args.num_nodes, args.num_gpus_per_node)
            # import os
            # os.system('sh scripts/allreduce_scrpit.sh')
            # os.system('sh scripts/p2p_scrpit.sh')
            return

        # Create hardware config directory
        hardware_config_dir = os.path.join(self.path, "./hardware_configs")
        if not os.path.exists(hardware_config_dir):
            os.mkdir(hardware_config_dir)

        # Profile allreduce bandwidth
        nccl_file = "build/all_reduce_perf"
        ARGS = self.prepare_nccltest_args(nccl_file)
        hardware_config_file = "sp_time_%dnodes_%dgpus_per_node.json" % (args.num_nodes, args.num_gpus_per_node)
        hardware_config_path = os.path.join(hardware_config_dir, hardware_config_file)
        allreduce_size = world_size
        while allreduce_size > 1:
            allreduce_consec = 1
            print(
                "============= allreduce_size: %d, allreduce_consec: %d ============="
                % (allreduce_size, allreduce_consec)
            )
            allreduce_groups = self.generate_allreduce_groups(world_size, allreduce_size, allreduce_consec)
            sizes, times = self.launch_nccl_test(allreduce_groups, args.num_gpus_per_node, ARGS, mode="detail")
            for size, time in zip(sizes, times):
                key = "allreduce_size_%d_%dMB_time" % (allreduce_size, size)
                self.write_config(hardware_config_path, key, time)
            print("=" * 70, "\n")
            allreduce_size //= 2

        # Profile all-to-all bandwidth
        nccl_file = "build/alltoall_perf"
        ARGS = self.prepare_nccltest_args(nccl_file)
        hardware_config_file = "sp_time_%dnodes_%dgpus_per_node.json" % (args.num_nodes, args.num_gpus_per_node)
        hardware_config_path = os.path.join(hardware_config_dir, hardware_config_file)
        all2all_size = world_size
        while all2all_size > 1:
            all2all_consec = 1
            print("============= all2all_size: %d, all2all_consec: %d =============" % (all2all_size, all2all_consec))
            allreduce_groups = self.generate_allreduce_groups(world_size, all2all_size, all2all_consec)
            sizes, times = self.launch_nccl_test(allreduce_groups, args.num_gpus_per_node, ARGS, mode="detail")
            for size, time in zip(sizes, times):
                key = "all2all_size_%d_%dMB_time" % (all2all_size, size)
                self.write_config(hardware_config_path, key, time)
            print("=" * 70, "\n")
            all2all_size //= 2

        os.system("rm -rf %s" % (os.path.join(self.path, "nccl_log")))

    def write_config(self, hardware_config_path: str, key: str, bandwidth: float) -> None:
        """Write bandwidth/time results to hardware config file

        Args:
            hardware_config_path: Path to the hardware config file
            key: Key for the bandwidth/time result
            bandwidth: Measured bandwidth or time value
        """
        config = read_json_config(hardware_config_path) if os.path.exists(hardware_config_path) else dict()
        config[key] = bandwidth
        write_json_config(config, hardware_config_path)
        print("Already written bandwidth/time %s into hardware config file %s!" % (key, hardware_config_path))

    def read_hostfile(self) -> List[str]:
        """Read hostnames from hostfile

        Returns:
            List[str]: List of hostnames from the hostfile
        """
        args = self.args
        hostfile = os.path.join(self.path, args.hostfile)
        with open(hostfile, "r") as f:
            hostnames = f.readlines()
        hostnames = [hostname.strip() for hostname in hostnames if hostname.strip() != ""]

        return hostnames

    def prepare_nccltest_args(self, nccl_file="build/all_reduce_perf") -> str:
        """Prepare arguments for NCCL tests

        Args:
            nccl_file: Path to NCCL test executable relative to nccl_test_dir

        Returns:
            str: Command line arguments for NCCL test

        Note:
            Will build NCCL test if not already built
        """
        args = self.args
        nccl_file = os.path.join(self.path, args.nccl_test_dir, nccl_file)
        if not os.path.exists(nccl_file):
            print("Nccl test file %s does not exist!" % nccl_file)
            print("Building nccl-test...")
            if args.num_nodes == 1:
                os.system(
                    "USE_EXPORT_VARIABLE=1 MAKE_MPI=0 sh %s" % (os.path.join(self.path, "scripts/build_nccl_test.sh"))
                )
            else:
                os.system(
                    "USE_EXPORT_VARIABLE=1 MAKE_MPI=1 MPI_PATH=%s sh %s"
                    % (args.mpi_path, os.path.join(self.path, "scripts/build_nccl_test.sh"))
                )
            print("Nccl-test built succesfully!")
        ARGS = ""
        ARGS += "USE_EXPORT_VARIABLE=1 "
        ARGS += "START_MB=%d " % args.start_mb
        ARGS += "END_MB=%d " % args.end_mb
        ARGS += "SCALE=%d " % args.scale
        ARGS += "NCCLTEST_FILE=%s " % nccl_file
        ARGS += "OUTPUT_TO_LOG=1 "
        return ARGS

    def generate_allreduce_groups(
        self, world_size: int, allreduce_size: int, allreduce_consec: bool
    ) -> List[List[int]]:
        """Generate groups for allreduce communication

        Args:
            world_size: Total number of processes
            allreduce_size: Size of each allreduce group
            allreduce_consec: Whether to use consecutive GPU mapping

        Returns:
            List[List[int]]: List of process groups for allreduce
        """
        allreduce_size = int(allreduce_size)
        num_allreduce_groups = int(world_size // allreduce_size)
        allreduce_groups = []
        for i in range(num_allreduce_groups):
            if allreduce_consec:
                ranks = list(range(i * allreduce_size, (i + 1) * allreduce_size))
            else:
                ranks = list(range(i, world_size, num_allreduce_groups))
            allreduce_groups.append(ranks)
        return allreduce_groups

    def generate_p2p_groups(self, world_size: int, pp_size: int) -> List[List[int]]:
        """Generate groups for point-to-point communication

        Args:
            world_size: Total number of processes
            pp_size: Size of each pipeline parallel group

        Returns:
            List[List[int]]: List of process groups for p2p communication
        """
        pp_size = int(pp_size)
        num_pp_groups = int(world_size // pp_size)
        pp_groups = []
        for i in range(num_pp_groups):
            ranks = list(range(i, world_size, num_pp_groups))
            pp_groups.append(ranks)
        return pp_groups

    def launch_nccl_test(
        self, groups: List[List[int]], num_gpus_per_node: int, ARGS: str, mode: str = "avg"
    ) -> Union[float, Tuple[List[int], List[float]]]:
        """Launch NCCL test for given process groups

        Args:
            groups: List of process groups to test
            num_gpus_per_node: Number of GPUs per node
            ARGS: Command line arguments for NCCL test
            mode: Test mode, either 'avg' for average bandwidth or 'detail' for detailed results

        Returns:
            Union[float, Tuple[List[int], List[float]]]:
                If mode=='avg': Average bandwidth in MB/s
                If mode=='detail': Tuple of (message sizes in MB, communication times in milliseconds)
        """
        hostnames = self.read_hostfile()
        bandwidths = []
        for group in groups:
            print("device group:", group)
            host_ids = sorted(list(set([rank // num_gpus_per_node for rank in group])))
            group_num_nodes = len(host_ids)
            group_num_gpus_per_node = len(group) // group_num_nodes
            cuda_visible_devices = sorted(list(set([rank % num_gpus_per_node for rank in group])))
            print(
                "num_nodes: %d, host_ids:" % group_num_nodes,
                host_ids,
                " num_gpus_per_node: %d, cuda_visible_devices:" % group_num_gpus_per_node,
                cuda_visible_devices,
            )
            hostname = ",".join([hostnames[i] for i in host_ids])
            DEVICE_ARGS = ""
            DEVICE_ARGS += "HOSTNAMES=%s " % hostname
            DEVICE_ARGS += "NUM_NODES=%d " % group_num_nodes
            DEVICE_ARGS += "NUM_GPUS_PER_NODE=%d " % group_num_gpus_per_node
            DEVICE_ARGS += 'DEVICES="CUDA_VISIBLE_DEVICES=%s" ' % (",".join([str(i) for i in cuda_visible_devices]))
            if mode == "detail":
                ARGS += "START_MB=1 "
                ARGS += "END_MB=1024 "
            print(DEVICE_ARGS + ARGS)
            os.system(DEVICE_ARGS + ARGS + "sh %s" % (os.path.join(self.path, "scripts/run_nccl_test.sh")))
            with open("nccl_log/1/rank.0/stdout", "r") as f:
                lines = f.readlines()
            if mode == "avg":
                for line in lines[::-1]:
                    if "Avg bus bandwidth" in line:
                        result = line
                        bandwidth = float(line.split()[-1])
                        break
                print(result)
                bandwidths.append(bandwidth)
                if self.args.avg_or_min_or_first == "first":
                    break
            else:
                sizes = []
                times = []
                for line in lines:
                    datas = line.split()
                    if len(datas) > 10 and datas[0].isdigit():
                        sizes.append(int(datas[0]) // 1024 // 1024)
                        times.append(float(datas[5]) / 1000)
                return sizes, times
        bandwidth = np.min(bandwidths) if self.args.avg_or_min_or_first == "min" else np.mean(bandwidths)
        print("Bandwidths:", bandwidths, "Average bandwidth:", bandwidth)
        print()
        return bandwidth

    # =============== For Launching Scripts for Profiling Overlap Slowdown Coefficient ===============
    def profile_overlap(self):
        """Profile overlap slowdown coefficient

        This method launches scripts to profile the overlap between computation and communication
        """
        args = self.args
        ARGS = ""
        ARGS += "USE_EXPORT_VARIABLE=1 "
        ARGS += "NUM_GPUS_PER_NODE=%d " % args.num_gpus_per_node
        ARGS += "OVERLAP_TIME_MULTIPLY=%d " % args.overlap_time_multiply
        os.system(ARGS + "sh %s" % (os.path.join(self.path, "scripts/profile_overlap.sh")))
