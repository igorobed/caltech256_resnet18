import argparse
from pipelines import train_base, train_contrastive, eval


parser = argparse.ArgumentParser(description="Hello World")
parser.add_argument(
    "--pipeline",
    default="train_base",
    type=str,
    metavar="P",
    help="Define strategy: train_base, train_simclr, eval",
    dest="pipeline"
    )
parser.add_argument(
    "--resnet18",
    default="torch",
    type=str,
    metavar="R18",
    help="Define resnet18 version: torch or my",
    dest="resnet18_version"
)
parser.add_argument(
    "--eval_model_dir",
    default="./",
    type=str,
    metavar="EMD",
    help="Path for dirrectory with training results of the model. Example: './runs/run_train_1/'",
    dest="eval_model_dir"
)

# вся остальная необходимая для выполнения пайплайнов инфа задается в соответствующих
# конфигурационных файлах


def main():
    args = parser.parse_args()
    if args.pipeline == "train_base":
        train_base.train(args.resnet18_version)
    elif args.pipeline == "train_simclr":
        train_contrastive.train(args.pipeline, args.resnet18_version)
    elif args.pipeline == "eval":
        eval.eval(args.eval_model_dir)


if __name__ == "__main__":
    main()
