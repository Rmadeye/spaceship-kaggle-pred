import argparse

from network.models.classicml import train_logistic_regression


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/rmadeye/kaggle/spaceship/data/inputs/base')
    parser.add_argument('--penalty', type=int, default=1)
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--solver', type=str, default='lbfgs')
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--multi_class', type=str, default='auto')
    parser.add_argument('--class_weight', type=int, default=0)
    parser.add_argument('--use_scaler', type=bool, default=True)
    parser.add_argument('--wandb_log', type=bool, default=True)
    args = parser.parse_args()

    train_logistic_regression(data_dir=args.data_dir,
                                penalty=args.penalty,
                                C=args.C,
                                solver=args.solver,
                                max_iter=args.max_iter,
                                multi_class=args.multi_class,
                                class_weight=args.class_weight,
                                wandb_log=args.wandb_log)