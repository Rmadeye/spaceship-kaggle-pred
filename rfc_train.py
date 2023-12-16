import argparse

from network.models.rfc import train_rfc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/rmadeye/kaggle/spaceship/data/inputs/new_base/')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=100)
    parser.add_argument('--criterion', type=str, default='gini')
    parser.add_argument('--max_features', type=int, default=0)
    parser.add_argument('--class_weight', type=int, default=0)
    parser.add_argument('--ccp_alpha', type=float, default=0.0)
    parser.add_argument('--bootstrap', type=bool, default=True)
    parser.add_argument('--use_scaler', type=bool, default=True)
    parser.add_argument('--wandb_log', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=False)
    args = parser.parse_args()
    train_rfc(data_dir=args.data_dir,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                criterion=args.criterion,
                max_features=args.max_features,
                class_weight=args.class_weight,
                ccp_alpha=args.ccp_alpha,
                bootstrap=args.bootstrap,
                use_scaler=args.use_scaler,
                wandb_log=args.wandb_log,
                save_model=args.save_model)
