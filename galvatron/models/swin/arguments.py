
def model_args(parser):
    group = parser.add_argument_group(title='Model Arguments')

    group.add_argument(
        "--model_size", type=str, default='swin-large', help="Model size.", choices=['swin-large', 'swin-huge']
    )
    group.add_argument(
        "--mlp_ratio", type=int, default=4, help="MLP ratio."
    )
    group.add_argument(
        "--drop_path_rate", type=float, default=0.2, help="Drop path rate."
    )
    group.add_argument(
        "--embed_dim", type=int, default=320, help="Embed dim.",
    )
    group.add_argument(
        "--depths", nargs='+', type=int, default=[1], help="Depths."
    )
    group.add_argument(
        "--num_heads", nargs='+', type=int, default=[2], help="Num heads."
    )
    group.add_argument(
        "--window_size", type=int, default=7, help="Window size."
    )
    group.add_argument(
        "--image_size", type=int, default=224, help="Input image size."
    )
    group.add_argument(
        "--patch_size", type=int, default=16, help="Patch size of Swin Transformer."
    )
    group.add_argument(
        "--num_channels", type=int, default=3, help="Number of channels."
    )
    group.add_argument(
        "--num_classes", type=int, default=1000, help="Number of labels for image classification."
    )
    group.add_argument(
        "--seq_length", type=int, default=None, help="Sequence length. (Add to compatible with profiler)"
    )
    return parser


def layernum_arg_names():
    return ['depths']


def seqlen_arg_names():
    return ['seq_length']
