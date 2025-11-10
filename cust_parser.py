import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="setup the model warmup, the model architecture, and the possible further components")
    
    #warmup settings (DEFAULT: false)
    parser.add_argument(
        "--warmup_set",
        action="store_true",
        default=False,
        help="Use warmup set for training"
    )
    parser.add_argument(
        "--warmup_slice",
        action="store_true",
        default=False,
        help="Use warmup slice for training"
    )

    #model components
    parser.add_argument(
        "--use_gru",
        action="store_true",
        default=False,
        help="Use GRU for the model"
    )
    parser.add_argument(
        "--use_pd",
        action="store_true",
        default=False,
        help="Use purity-diversity for the model"
    )

    #base model architecture(default: true)
    parser.add_argument(
        "--use_mhsa",
        action="store_false",
        default=True,
        help="Use multi-head self-attention for the model"
    )
    parser.add_argument(
        "--use_G_prompt",
        action="store_false",
        default=True,
        help="Use G prompt for the model"
    )
    parser.add_argument(
        "--use_E_prompt",
        action="store_false",
        default=True,
        help="Use E prompt for the model"
    )
    parser.add_argument(
        "--use_T_prompt",
        action="store_false",
        default=True,
        help="Use T prompt for the model"
    )

    #hyperparameters
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help="Dimension of embeddings (default: 32)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimension of hidden states (default: 64)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)"
    )
    parser.add_argument(
        "--pd_balance",
        type=float,
        default=0.5,
        help="Balance between purity and diversity in purity-diversity (default: 0.5)"
    )
    parser.add_argument(
        "--slice_size",
        type=int,
        default=6000,
        help="Size of the slice for warmup (default: 6000)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )

    return parser.parse_args()