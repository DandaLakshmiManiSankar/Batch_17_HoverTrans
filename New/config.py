import argparse


def config():
    parser = argparse.ArgumentParser()

    # ======================
    # DATA PATHS
    # ======================
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='/kaggle/working/models')
    parser.add_argument('--ssl_model_path', type=str, default='mae_encoder.pth')

    # ======================
    # GENERAL SETTINGS
    # ======================
    parser.add_argument('--model_name', type=str, default='hovertrans')
    parser.add_argument('--writer_comment', type=str, default='GDPH_SYSUCC')
    parser.add_argument('--save_model', action='store_true')

    # ======================
    # IMAGE SETTINGS
    # ======================
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--class_num', type=int, default=2)

    # ======================
    # TRAINING SETTINGS
    # ======================
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--log_step', type=int, default=5)

    # ======================
    # OPTIMIZER & SCHEDULER
    # ======================
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_decay', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--step', type=int, default=5)

    # ======================
    # HOVERTRANS ARCHITECTURE
    # ======================
    parser.add_argument('--patch_size', nargs='+', type=int, default=[2, 2, 2, 2])
    parser.add_argument('--hover_size', nargs='+', type=int, default=[2, 2, 2, 2])
    parser.add_argument('--dim', nargs='+', type=int, default=[4, 8, 16, 32])
    parser.add_argument('--depth', nargs='+', type=int, default=[2, 4, 4, 2])
    parser.add_argument('--num_heads', nargs='+', type=int, default=[2, 4, 8, 16])
    parser.add_argument('--num_inner_head', nargs='+', type=int, default=[2, 4, 8, 16])

    # ======================
    # LOSS
    # ======================
    parser.add_argument('--loss_function', type=str, default='CE')

    # ======================
    # SSL PRETRAIN SETTINGS
    # ======================
    parser.add_argument('--ssl_pretrain', action='store_true')
    parser.add_argument('--ssl_epochs', type=int, default=150)
    parser.add_argument('--ssl_batch_size', type=int, default=32)
    parser.add_argument('--ssl_lr', type=float, default=1e-4)

    # ======================
    # FINETUNE SETTINGS
    # ======================
    parser.add_argument('--finetune_epochs', type=int, default=50)
    parser.add_argument('--finetune_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args
