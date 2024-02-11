import argparse

from network.models.lgbm import train_lgbm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/rmadeye/kaggle/spaceship/data/inputs/knnfill/')
    parser.add_argument('--boosting_type', type=str, default='gbdt')
    parser.add_argument('--num_leaves', type=int, default=31)
    parser.add_argument('--max_depth', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--subsample_for_bin', type=int, default=200000)
    parser.add_argument('--objective', type=str, default=None)
    parser.add_argument('--class_weight', type=str, default=None)
    parser.add_argument('--min_split_gain', type=float, default=0.0)
    parser.add_argument('--min_child_weight', type=float, default=0.001)
    parser.add_argument('--min_child_samples', type=int, default=20)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--subsample_freq', type=int, default=0)
    parser.add_argument('--colsample_bytree', type=float, default=1.0)
    parser.add_argument('--reg_alpha', type=float, default=0.0)
    parser.add_argument('--reg_lambda', type=float, default=0.0)
    parser.add_argument('--random_state', type=int, default=None)
    parser.add_argument('--n_jobs', type=int, default=None)
    parser.add_argument('--importance_type', type=str, default='split')
    parser.add_argument('--use_scaler', type=bool, default=True)
    parser.add_argument('--wandb_log', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--final_train', type=bool, default=False)
    args = parser.parse_args()
    train_lgbm(data_dir=args.data_dir,
               boosting_type=args.boosting_type,
               num_leaves=args.num_leaves,
               max_depth=args.max_depth,
               learning_rate=args.learning_rate,
               n_estimators=args.n_estimators,
               subsample_for_bin=args.subsample_for_bin,
               objective=args.objective,
            #    class_weight=args.class_weight,
               min_split_gain=args.min_split_gain,
               min_child_weight=args.min_child_weight,
               min_child_samples=args.min_child_samples,
               subsample=args.subsample,
               subsample_freq=args.subsample_freq,
               colsample_bytree=args.colsample_bytree,
               reg_alpha=args.reg_alpha,
               reg_lambda=args.reg_lambda,
               random_state=args.random_state,
               n_jobs=args.n_jobs,
               importance_type=args.importance_type,
               use_scaler=args.use_scaler,
               wandb_log=args.wandb_log,
               save_model=args.save_model,
               final_train=args.final_train)

