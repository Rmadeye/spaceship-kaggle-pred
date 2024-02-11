import argparse

from network.models.knn import train_knn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train KNN model')
    parser.add_argument('--data_dir', type=str, help='Path to data directory',default='/home/rmadeye/kaggle/spaceship/data/inputs/knnfill/')
    parser.add_argument('--n_neighbors', type=int, help='Number of neighbors')
    parser.add_argument('--weights', type=int, help='Weight function used in prediction')
    parser.add_argument('--algorithm', type=int, help='Algorithm used to compute the nearest neighbors')
    parser.add_argument('--leaf_size', type=int, help='Leaf size passed to BallTree or KDTree')
    parser.add_argument('--p', type=int, help='Power parameter for the Minkowski metric')
    parser.add_argument('--use_scaler', type=bool, help='Use scaler')
    parser.add_argument('--wandb_log', type=bool, help='Use wandb',  default=True)
    parser.add_argument('--save_model', type=bool, help='Save model', default=False)
    args = parser.parse_args()
    train_knn(data_dir=args.data_dir,
                n_neighbors=args.n_neighbors,
                weights=args.weights,
                algorithm=args.algorithm,
                leaf_size=args.leaf_size,
                p=args.p,
                use_scaler=args.use_scaler,
                wandb_log=args.wandb_log,
                save_model=args.save_model)
    