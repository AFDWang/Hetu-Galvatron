def model_args(parser):
    group = parser.add_argument_group(title="Model Arguments")

    group.add_argument(
        "--model_size",
        type=str,
        default="t5-small",
        help="Model size.",
        choices=["t5-small", "t5-base", "t5-large", "t5-3B"],
    )
    group.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="Hidden size of transformer model",
    )
    group.add_argument("--num_encoder_layers", type=int, default=24, help="Number of encoder layers")
    group.add_argument("--num_decoder_layers", type=int, default=24, help="Number of decoder layers")
    group.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=16,
        help="Number of attention heads",
    )
    group.add_argument("--encoder_seq_length", type=int, default=512, help="Maximum encoder sequence len")
    group.add_argument("--decoder_seq_length", type=int, default=512, help="Maximum decoder sequence len")
    group.add_argument("--vocab_size", type=int, default=32128, help="Total number of vocab")
    return parser


def layernum_arg_names():
    return ["num_encoder_layers", "num_decoder_layers"]


def seqlen_arg_names():
    return ["encoder_seq_length", "decoder_seq_length"]
