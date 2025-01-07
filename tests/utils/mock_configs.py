import json
from typing import Dict
from pathlib import Path
from tests.utils.search_args import SearchArgs
from tests.utils.model_utils import ModelFactory
from tests.models.configs.get_config_json import ConfigFactory
from galvatron.core.search_engine import GalvatronSearchEngine

def create_static_time_config() -> Dict[str, float]:
    """Create mock time config for static profiling mode"""
    return {
        "layertype_0_bsz8_seq4096": 11.219752883911134,
        "layertype_other_0_bsz8_seq4096": 27.296485137939456,
    }

def create_batch_time_config() -> Dict[str, float]:
    """Create mock time config for batch profiling mode"""
    return {
        "layertype_0_bsz1_seq4096": 12.4057201385498,
        "layertype_0_bsz2_seq4096": 11.603767204284669,
        "layertype_0_bsz3_seq4096": 11.878070322672523,
        "layertype_0_bsz4_seq4096": 11.152996063232425,
        "layertype_0_bsz5_seq4096": 10.984469451904294,
        "layertype_0_bsz6_seq4096": 10.83633092244466,
        "layertype_0_bsz7_seq4096": 11.184148515973764,
        "layertype_0_bsz8_seq4096": 11.219752883911134,
        "layertype_0_bsz9_seq4096": 11.234162224663628,
        "layertype_0_bsz10_seq4096": 11.236963653564455,
        "layertype_other_0_bsz1_seq4096": 31.97360305786134,
        "layertype_other_0_bsz2_seq4096": 29.767119598388675,
        "layertype_other_0_bsz3_seq4096": 27.621103922526043,
        "layertype_other_0_bsz4_seq4096": 29.155476379394514,
        "layertype_other_0_bsz5_seq4096": 28.962725830078124,
        "layertype_other_0_bsz6_seq4096": 28.964708455403656,
        "layertype_other_0_bsz7_seq4096": 27.860640171596003,
        "layertype_other_0_bsz8_seq4096": 27.296485137939456,
        "layertype_other_0_bsz9_seq4096": 27.257109239366326,
        "layertype_other_0_bsz10_seq4096": 27.296959228515618,
    }

def create_sequence_time_config() -> Dict[str, float]:
    """Create mock time config for sequence profiling mode"""
    return {
        "layertype_0_bsz1_seq4096": 12.4057201385498,
        "layertype_0_bsz1_seq8192": 28.454231262207003,
        "layertype_0_bsz1_seq12288": 39.43479309082031,
        "layertype_0_bsz1_seq16384": 52.60663909912111,
        "layertype_0_bsz1_seq20480": 70.75289154052746,
        "layertype_0_bsz1_seq24576": 82.6971145629883,
        "layertype_0_bsz1_seq28672": 106.13850097656245,
        "layertype_0_bsz1_seq32768": 123.1998901367187,
        "layertype_other_0_bsz1_seq4096": 31.97360305786134,
        "layertype_other_0_bsz1_seq8192": 56.27244796752933,
        "layertype_other_0_bsz1_seq12288": 86.6235107421875,
        "layertype_other_0_bsz1_seq16384": 121.2523483276367,
        "layertype_other_0_bsz1_seq20480": 141.90354614257797,
        "layertype_other_0_bsz1_seq24576": 177.68662719726558,
        "layertype_other_0_bsz1_seq28672": 197.4156311035157,
        "layertype_other_0_bsz1_seq32768": 225.79444885253918
    }

def create_static_memory_config():
    """Create mock memory profiling config for static profiling mode"""
    return {
        "layertype_0": {
            "4096": {
                "parameter_size": 768.859375,
                "tp_activation_per_bsz_dict": {
                    "1": 272.28173828125,
                    "2": 176.156494140625,
                    "4": 128.0938720703125,
                    "8": 104.14166259765625,
                    "checkpoint": 16.0
                }
            },
        },
        "other_memory_pp_off": {
            "4096": {
                "model_states": {
                    "1": 6540.0,
                    "2": 3270.0,
                    "4": 1635.0,
                    "8": 817.5
                },
                "activation": {
                    "1": 920.88623046875,
                    "2": 920.88623046875,
                    "4": 920.88623046875,
                    "8": 920.88623046876
                }
            },
        },
        "other_memory_pp_on_first": {
            "4096": {
                "model_states": {
                    "1": 3303.0,
                    "2": 1651.5,
                    "4": 825.75,
                    "8": 412.875
                },
                "activation": {
                    "1": 26.8785400390625,
                    "2": 26.8785400390625,
                    "4": 26.8785400390625,
                    "8": 26.8785400390625,
                }
            },
        },
        "other_memory_pp_on_last": {
            "4096": {
                "model_states": {
                    "1": 3431.0,
                    "2": 1715.5,
                    "4": 857.75,
                    "8": 428.875
                },
                "activation": {
                    "1": 352.786865234375,
                    "2": 352.786865234375,
                    "4": 352.786865234375,
                    "8": 352.786865234375,
                }
            },
        },
    }

def create_static_memory_config_sp():
    """Create mock memory profiling config for static profiling mode with sequence parallelism"""
    return {
        "layertype_0_sp": {
            "4096": {
                "parameter_size": 973.0283203125,
                "tp_activation_per_bsz_dict": {
                    "1": 1044.4697265625,
                    "checkpoint": 28.0,
                    "2": 522.23486328125,
                    "4": 261.117431640625,
                    "8": 130.5587158203125
                }
            },
        },
        "other_memory_pp_off_sp": {
            "4096": {
                "model_states": {
                    "1": 16768.29296875,
                    "2": 8384.146484375,
                    "4": 4192.0732421875,
                    "8": 2096.03662109375
                },
                "activation": {
                    "1": 2942.11962890625,
                    "2": 1471.059814453125,
                    "4": 735.5299072265625,
                    "8": 367.76495361328125
                }
            },
        },
        "other_memory_pp_on_first_sp": {
            "4096": {
                "model_states": {
                    "1": 8353.0009765625,
                    "2": 4176.50048828125,
                    "4": 2088.250244140625,
                    "8": 1044.1251220703125
                },
                "activation": {
                    "1": 409.4993896484375,
                    "2": 204.74969482421875,
                    "4": 102.37484741210938,
                    "8": 51.18742370605469
                }
            },
        },
        "other_memory_pp_on_last_sp": {
            "4096": {
                "model_states": {
                    "1": 8353.0556640625,
                    "2": 4176.52783203125,
                    "4": 2088.263916015625,
                    "8": 1044.1319580078125
                },
                "activation": {
                    "1": 2475.5216064453125,
                    "2": 1237.7608032226562,
                    "4": 618.8804016113281,
                    "8": 309.44020080566406
                }
            },
        }
    }

def create_sequence_memory_config_sp():
    """Create mock memory profiling config for sequence profiling mode with sequence parallelism"""
    return {
        "layertype_0_sp": {
            "512": {
                "parameter_size": 973.771484375,
                "tp_activation_per_bsz_dict": {
                    "1": 131.205078125,
                    "checkpoint": 3.5,
                    "2": 65.6025390625,
                    "4": 32.80126953125,
                    "8": 16.400634765625
                }
            },
            "1024": {
                "parameter_size": 973.771484375,
                "tp_activation_per_bsz_dict": {
                    "1": 261.1181640625,
                    "checkpoint": 7.0,
                    "2": 130.55908203125,
                    "4": 65.279541015625,
                    "8": 32.6397705078125
                }
            },
            "2048": {
                "parameter_size": 973.771484375,
                "tp_activation_per_bsz_dict": {
                    "1": 521.9853515625,
                    "checkpoint": 14.0,
                    "2": 260.99267578125,
                    "4": 130.496337890625,
                    "8": 65.2481689453125
                }
            },
            "4096": {
                "parameter_size": 973.0283203125,
                "tp_activation_per_bsz_dict": {
                    "1": 1044.4697265625,
                    "checkpoint": 28.0,
                    "2": 522.23486328125,
                    "4": 261.117431640625,
                    "8": 130.5587158203125
                }
            },
            "8192": {
                "parameter_size": 973.0283203125,
                "tp_activation_per_bsz_dict": {
                    "1": 2088.28955078125,
                    "checkpoint": 56.0,
                    "2": 1044.144775390625,
                    "4": 522.0723876953125,
                    "8": 261.03619384765625
                }
            }
        },
        "other_memory_pp_off_sp": {
            "512": {
                "model_states": {
                    "1": 16762.12890625,
                    "2": 8381.064453125,
                    "4": 4190.5322265625,
                    "8": 2095.26611328125
                },
                "activation": {
                    "1": 2728.296875,
                    "2": 1364.1484375,
                    "4": 682.07421875,
                    "8": 341.037109375
                }
            },
            "1024": {
                "model_states": {
                    "1": 16762.16015625,
                    "2": 8381.080078125,
                    "4": 4190.5400390625,
                    "8": 2095.27001953125
                },
                "activation": {
                    "1": 2598.3837890625,
                    "2": 1299.19189453125,
                    "4": 649.595947265625,
                    "8": 324.7979736328125
                }
            },
            "2048": {
                "model_states": {
                    "1": 16762.22265625,
                    "2": 8381.111328125,
                    "4": 4190.5556640625,
                    "8": 2095.27783203125
                },
                "activation": {
                    "1": 2562.38623046875,
                    "2": 1281.193115234375,
                    "4": 640.5965576171875,
                    "8": 320.29827880859375
                }
            },
            "4096": {
                "model_states": {
                    "1": 16768.29296875,
                    "2": 8384.146484375,
                    "4": 4192.0732421875,
                    "8": 2096.03662109375
                },
                "activation": {
                    "1": 2942.11962890625,
                    "2": 1471.059814453125,
                    "4": 735.5299072265625,
                    "8": 367.76495361328125
                }
            },
            "8192": {
                "model_states": {
                    "1": 16768.54296875,
                    "2": 8384.271484375,
                    "4": 4192.1357421875,
                    "8": 2096.06787109375
                },
                "activation": {
                    "1": 5487.8828125,
                    "2": 2743.94140625,
                    "4": 1371.970703125,
                    "8": 685.9853515625
                }
            }
        },
        "other_memory_pp_on_first_sp": {
            "512": {
                "model_states": {
                    "1": 8349.5908203125,
                    "2": 4174.79541015625,
                    "4": 2087.397705078125,
                    "8": 1043.6988525390625
                },
                "activation": {
                    "1": 395.7950439453125,
                    "2": 197.89752197265625,
                    "4": 98.94876098632812,
                    "8": 49.47438049316406
                }
            },
            "1024": {
                "model_states": {
                    "1": 8350.6533203125,
                    "2": 4175.32666015625,
                    "4": 2087.663330078125,
                    "8": 1043.8316650390625
                },
                "activation": {
                    "1": 272.7569580078125,
                    "2": 136.37847900390625,
                    "4": 68.18923950195312,
                    "8": 34.09461975097656
                }
            },
            "2048": {
                "model_states": {
                    "1": 8349.7783203125,
                    "2": 4174.88916015625,
                    "4": 2087.444580078125,
                    "8": 1043.7222900390625
                },
                "activation": {
                    "1": 221.1243896484375,
                    "2": 110.56219482421875,
                    "4": 55.281097412109375,
                    "8": 27.640548706054688
                }
            },
            "4096": {
                "model_states": {
                    "1": 8353.0009765625,
                    "2": 4176.50048828125,
                    "4": 2088.250244140625,
                    "8": 1044.1251220703125
                },
                "activation": {
                    "1": 409.4993896484375,
                    "2": 204.74969482421875,
                    "4": 102.37484741210938,
                    "8": 51.18742370605469
                }
            },
            "8192": {
                "model_states": {
                    "1": 8351.5009765625,
                    "2": 4175.75048828125,
                    "4": 2087.875244140625,
                    "8": 1043.9376220703125
                },
                "activation": {
                    "1": 787.1483154296875,
                    "2": 393.57415771484375,
                    "4": 196.78707885742188,
                    "8": 98.39353942871094
                }
            }
        },
        "other_memory_pp_on_last_sp": {
            "512": {
                "model_states": {
                    "1": 8351.5908203125,
                    "2": 4175.79541015625,
                    "4": 2087.897705078125,
                    "8": 1043.9488525390625
                },
                "activation": {
                    "1": 425.352783203125,
                    "2": 212.6763916015625,
                    "4": 106.33819580078125,
                    "8": 53.169097900390625
                }
            },
            "1024": {
                "model_states": {
                    "1": 8349.7080078125,
                    "2": 4174.85400390625,
                    "4": 2087.427001953125,
                    "8": 1043.7135009765625
                },
                "activation": {
                    "1": 527.6573486328125,
                    "2": 263.82867431640625,
                    "4": 131.91433715820312,
                    "8": 65.95716857910156
                }
            },
            "2048": {
                "model_states": {
                    "1": 8349.8330078125,
                    "2": 4174.91650390625,
                    "4": 2087.458251953125,
                    "8": 1043.7291259765625
                },
                "activation": {
                    "1": 1177.1954345703125,
                    "2": 588.5977172851562,
                    "4": 294.2988586425781,
                    "8": 147.14942932128906
                }
            },
            "4096": {
                "model_states": {
                    "1": 8353.0556640625,
                    "2": 4176.52783203125,
                    "4": 2088.263916015625,
                    "8": 1044.1319580078125
                },
                "activation": {
                    "1": 2475.5216064453125,
                    "2": 1237.7608032226562,
                    "4": 618.8804016113281,
                    "8": 309.44020080566406
                }
            },
            "8192": {
                "model_states": {
                    "1": 8351.5556640625,
                    "2": 4175.77783203125,
                    "4": 2087.888916015625,
                    "8": 1043.9444580078125
                },
                "activation": {
                    "1": 5073.4478759765625,
                    "2": 2536.7239379882812,
                    "4": 1268.3619689941406,
                    "8": 634.1809844970703
                }
            }
        }
    }

def create_hardware_configs():
    """Create mock hardware configs"""
    return {
        "allreduce": {
            "allreduce_size_8_consec_1": 160.445,
            "allreduce_size_4_consec_1": 164.272,
            "allreduce_size_4_consec_0": 165.493,
            "allreduce_size_2_consec_1": 155.647,
            "allreduce_size_2_consec_0": 153.933
        },
        "p2p": {
            "pp_size_2": 147.32,
            "pp_size_4": 133.469,
            "pp_size_8": 108.616
        },
        "overlap": {
            "overlap_coe": 1.1534195950157762
        },
        "sp": {
            "allreduce_size_8_1MB_time": 0.07895,
            "allreduce_size_8_2MB_time": 0.10940000000000001,
            "allreduce_size_8_4MB_time": 0.1333,
            "allreduce_size_8_8MB_time": 0.1827,
            "allreduce_size_8_16MB_time": 0.29410000000000003,
            "allreduce_size_8_32MB_time": 0.4157,
            "allreduce_size_8_64MB_time": 0.6518999999999999,
            "allreduce_size_8_128MB_time": 1.2826,
            "allreduce_size_8_256MB_time": 2.3584,
            "allreduce_size_8_512MB_time": 4.6768,
            "allreduce_size_8_1024MB_time": 8.1409,
            "allreduce_size_4_1MB_time": 0.07981,
            "allreduce_size_4_2MB_time": 0.09109,
            "allreduce_size_4_4MB_time": 0.10909999999999999,
            "allreduce_size_4_8MB_time": 0.1581,
            "allreduce_size_4_16MB_time": 0.21830000000000002,
            "allreduce_size_4_32MB_time": 0.3205,
            "allreduce_size_4_64MB_time": 0.5848,
            "allreduce_size_4_128MB_time": 1.0725,
            "allreduce_size_4_256MB_time": 2.0709,
            "allreduce_size_4_512MB_time": 3.7352,
            "allreduce_size_4_1024MB_time": 7.187399999999999,
            "allreduce_size_2_1MB_time": 0.0703,
            "allreduce_size_2_2MB_time": 0.07931999999999999,
            "allreduce_size_2_4MB_time": 0.09008,
            "allreduce_size_2_8MB_time": 0.10840000000000001,
            "allreduce_size_2_16MB_time": 0.1434,
            "allreduce_size_2_32MB_time": 0.2281,
            "allreduce_size_2_64MB_time": 0.39239999999999997,
            "allreduce_size_2_128MB_time": 0.7417,
            "allreduce_size_2_256MB_time": 1.3887,
            "allreduce_size_2_512MB_time": 2.6886,
            "allreduce_size_2_1024MB_time": 5.1594,
            "all2all_size_8_1MB_time": 0.1124,
            "all2all_size_8_2MB_time": 0.1135,
            "all2all_size_8_4MB_time": 0.11090000000000001,
            "all2all_size_8_8MB_time": 0.1502,
            "all2all_size_8_16MB_time": 0.2003,
            "all2all_size_8_32MB_time": 0.243,
            "all2all_size_8_64MB_time": 0.3997,
            "all2all_size_8_128MB_time": 0.7135,
            "all2all_size_8_256MB_time": 1.2980999999999998,
            "all2all_size_8_512MB_time": 2.4821999999999997,
            "all2all_size_8_1024MB_time": 4.8151,
            "all2all_size_4_1MB_time": 0.05244,
            "all2all_size_4_2MB_time": 0.07992,
            "all2all_size_4_4MB_time": 0.1065,
            "all2all_size_4_8MB_time": 0.1255,
            "all2all_size_4_16MB_time": 0.1514,
            "all2all_size_4_32MB_time": 0.22369999999999998,
            "all2all_size_4_64MB_time": 0.3654,
            "all2all_size_4_128MB_time": 0.6439,
            "all2all_size_4_256MB_time": 1.1567,
            "all2all_size_4_512MB_time": 2.1003000000000003,
            "all2all_size_4_1024MB_time": 4.0389,
            "all2all_size_2_1MB_time": 0.0709,
            "all2all_size_2_2MB_time": 0.09942000000000001,
            "all2all_size_2_4MB_time": 0.11009999999999999,
            "all2all_size_2_8MB_time": 0.1047,
            "all2all_size_2_16MB_time": 0.12029999999999999,
            "all2all_size_2_32MB_time": 0.17880000000000001,
            "all2all_size_2_64MB_time": 0.2928,
            "all2all_size_2_128MB_time": 0.4756,
            "all2all_size_2_256MB_time": 0.8806,
            "all2all_size_2_512MB_time": 1.7752000000000001,
            "all2all_size_2_1024MB_time": 3.4954
        }
    }

def write_time_config(
    configs_dir: Path,
    model_name: str = "test",
    precision: str = "bf16",
    profile_mode: str = "static"
) -> None:
    """Write time profiling config to file"""
    configs_dir.mkdir(exist_ok=True)
    
    # Select time config based on profile mode
    time_config = {
        "static": create_static_time_config,
        "batch": create_batch_time_config,
        "sequence": create_sequence_time_config
    }[profile_mode]()
    
    with open(configs_dir / f"computation_profiling_{precision}_{model_name}.json", "w") as f:
        json.dump(time_config, f)

def write_memory_config(
    configs_dir: Path,
    model_name: str = "test",
    precision: str = "bf16",
    profile_mode: str = "static",
    sp_mode: bool = False,
) -> None:
    """Write memory profiling config to file"""
    configs_dir.mkdir(exist_ok=True)
    
    memory_config = {
        "static": create_static_memory_config if not sp_mode else create_static_memory_config_sp,
        "sequence": create_sequence_memory_config_sp,
    }[profile_mode]()
    
    with open(configs_dir / f"memory_profiling_{precision}_{model_name}.json", "w") as f:
        json.dump(memory_config, f)

def write_hardware_config(
    hardware_dir: Path,
    num_nodes: int = 1,
    gpus_per_node: int = 8
) -> None:
    """Write hardware profiling configs to files"""
    hardware_dir.mkdir(exist_ok=True)
    hw_configs = create_hardware_configs()
    
    # Write allreduce config
    with open(hardware_dir / f"allreduce_bandwidth_{num_nodes}nodes_{gpus_per_node}gpus_per_node.json", "w") as f:
        json.dump(hw_configs["allreduce"], f)
    
    # Write p2p config
    with open(hardware_dir / f"p2p_bandwidth_{num_nodes}nodes_{gpus_per_node}gpus_per_node.json", "w") as f:
        json.dump(hw_configs["p2p"], f)
    
    # Write overlap config
    with open(hardware_dir / "overlap_coefficient.json", "w") as f:
        json.dump(hw_configs["overlap"], f)
    
    # Write sp config
    with open(hardware_dir / f"sp_time_{num_nodes}nodes_{gpus_per_node}gpus_per_node.json", "w") as f:
        json.dump(hw_configs["sp"], f)

def initialize_search_engine(base_config_dirs, model_type, backend, time_mode = "static", memory_mode = "static", sp_enabled = False, seqlen_list = None, **kwargs):
    """Initialize search engine"""
    configs_dir, hardware_dir, output_dir = base_config_dirs

    # Setup search engine
    args = SearchArgs()
    model_layer_configs, model_name = ModelFactory.get_meta_configs(model_type, backend)
    config_json = ConfigFactory.get_config_json(model_type)
    args.model_size = config_json
    args.local_rank = 0
    config = ModelFactory.create_config(model_type, backend, args)

    # Set profiling paths and modes
    args.time_profiling_path = str(configs_dir)
    args.memory_profiling_path = str(configs_dir)
    args.allreduce_bandwidth_config_path = str(hardware_dir)
    args.p2p_bandwidth_config_path = str(hardware_dir)
    args.overlap_coe_path = str(hardware_dir)
    args.sp_time_path = str(hardware_dir)
    output_dir.mkdir(exist_ok=True)
    args.output_config_path = str(output_dir)
    args.time_profile_mode = time_mode
    args.memory_profile_mode = memory_mode
    args.sequence_parallel = sp_enabled

    for k, v in kwargs.items():
        setattr(args, k, v)

    # Initialize search engine
    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(str(configs_dir), model_layer_configs(config), model_type)
    if seqlen_list is not None:
        search_engine.seqlen_list = seqlen_list

    # Write config files
    write_time_config(configs_dir, profile_mode=time_mode, model_name=model_type)
    write_memory_config(configs_dir, profile_mode=memory_mode, sp_mode=sp_enabled, model_name=model_type)
    write_hardware_config(hardware_dir)
    # Initialize search engine
    search_engine.initialize_search_engine()

    return search_engine

