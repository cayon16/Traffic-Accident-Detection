import argparse


DATA_ROOT = r"data\finetune_data"

parser = argparse.ArgumentParser(
    description="Separate five-crop VADCLIP phase-1 and phase-2 training"
)

parser.add_argument("--seed", default=234, type=int)

parser.add_argument("--embed-dim", default=512, type=int)
parser.add_argument("--visual-length", default=256, type=int)
parser.add_argument("--visual-width", default=512, type=int)
parser.add_argument("--visual-head", default=1, type=int)
parser.add_argument("--visual-layers", default=2, type=int)
parser.add_argument("--attn-window", default=8, type=int)
parser.add_argument("--prompt-prefix", default=10, type=int)
parser.add_argument("--prompt-postfix", default=10, type=int)
parser.add_argument("--normal-prompt", default="normal")
parser.add_argument("--accident-prompt", default="roadAccidents")
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--classes-num", default=2, type=int)
parser.add_argument(
    "--use-padding-mask",
    default=False,
    action=argparse.BooleanOptionalAction,
)

parser.add_argument("--crop-count", default=5, type=int)
parser.add_argument("--spatial-top-k", default=3, type=int)
parser.add_argument("--top-k-divisor", default=16, type=int)
parser.add_argument("--top-k-min", default=2, type=int)
parser.add_argument("--top-k-max", default=5, type=int)
parser.add_argument("--text-loss-weight", default=5e-2, type=float)

parser.add_argument(
    "--train-list",
    default=DATA_ROOT + r"\list_5crop\train_features.csv",
)
parser.add_argument(
    "--test-list",
    default=DATA_ROOT + r"\list_5crop\val_features.csv",
)
parser.add_argument(
    "--gt-path",
    default=DATA_ROOT + r"\list_5crop\gt_ucf.npy",
)
parser.add_argument(
    "--timestamp-excel",
    default=DATA_ROOT + r"\list_5crop\metadata_train_filename_accident_time.xlsx",
)

parser.add_argument(
    "--phase1-pretrained-path",
    default=r"models\model_ucf.pth",
)
parser.add_argument(
    "--phase1-output-dir",
    default=DATA_ROOT + r"\model_2_phase\phase1_mil",
)
parser.add_argument("--phase1-epochs", default=10, type=int)
parser.add_argument("--phase1-lr", default=1e-5, type=float)
parser.add_argument(
    "--phase1-scheduler-milestones",
    default=[4,8],
    nargs="+",
    type=int,
)
parser.add_argument("--phase1-scheduler-rate", default=0.5, type=float)
parser.add_argument("--phase1-early-stop-patience", default=0, type=int)
parser.add_argument("--phase1-resume-checkpoint", default="")

parser.add_argument(
    "--phase2-pretrained-path",
    default=r"models\phase1_mil\model_best.pth",
)
parser.add_argument(
    "--phase2-output-dir",
    default=DATA_ROOT + r"\model_2_phase\phase2_timestamp",
)
parser.add_argument("--phase2-epochs", default=160, type=int)
parser.add_argument("--phase2-lr", default=5e-5, type=float)
parser.add_argument(
    "--phase2-scheduler-milestones",
    default=[30,90,130],
    nargs="+",
    type=int,
)
parser.add_argument("--phase2-scheduler-rate", default=0.5, type=float)
parser.add_argument("--phase2-early-stop-patience", default=0, type=int)
parser.add_argument("--phase2-resume-checkpoint", default="")

parser.add_argument("--reference-fps", default=30.0, type=float)
parser.add_argument("--frames-per-snippet", default=16, type=int)
parser.add_argument("--pre-normal-buffer-seconds", default=1.0, type=float)
parser.add_argument("--accident-window-start-seconds", default=-0.5, type=float)
parser.add_argument("--accident-window-end-seconds", default=2.5, type=float)
parser.add_argument("--post-normal-start-seconds", default=4.0, type=float)
parser.add_argument("--phase2-mil-weight", default=1.0, type=float)
parser.add_argument("--gap-weight", default=0.2, type=float)
parser.add_argument("--gap-margin", default=0.8, type=float)
parser.add_argument("--gap-warmup-epochs", default=3, type=int)
parser.add_argument("--max-grad-norm", default=1.0, type=float)

parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--num-workers", default=0, type=int)
parser.add_argument("--weight-decay", default=0.1, type=float)
parser.add_argument("--early-stop-min-delta", default=1e-5, type=float)
parser.add_argument("--save-every-epoch", action="store_true")

parser.add_argument(
    "--use-wandb",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument("--wandb-project", default="VADCLIP_accident_detection")
parser.add_argument("--wandb-entity", default="")
parser.add_argument("--phase1-wandb-run-name", default="stage1_basic_mil")
parser.add_argument(
    "--phase2-wandb-run-name",
    default="stage2_timestamp_gap",
)
