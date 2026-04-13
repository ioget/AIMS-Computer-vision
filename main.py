import argparse
import torch

from utils.prep import get_data_pytorch, get_data_tensorflow
from models.cnn_with_Pytorch import CNN1
from models.train import Trainer, TFTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intel Image Classification – PyTorch or TensorFlow"
    )
    parser.add_argument(
        '--framework', type=str, choices=['pytorch', 'tensorflow'],
        required=True,
        help="Framework to use: 'pytorch' or 'tensorflow'",
    )
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument('--wd', type=float, default=1e-4,
                        help="Weight decay – PyTorch only (default: 1e-4)")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'],
                        default='train',
                        help="'train' or 'eval' (default: train)")
    parser.add_argument('--cuda', action='store_true',
                        help="Use GPU if available (PyTorch only)")
    return parser.parse_args()


def run_pytorch(args):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"[PyTorch] device: {device}")

    train_loader, test_loader = get_data_pytorch()
    model = CNN1(num_classes=6).to(device)

    if args.mode == 'eval':
        model.load_state_dict(torch.load("rosly_model.pth", map_location=device))
        print("Loaded weights from rosly_model.pth")

    trainer = Trainer(model, train_loader, test_loader,
                      args.lr, args.wd, args.epochs, device)

    if args.mode == 'train':
        trainer.train(save=True, plot=True)

    trainer.evaluate()


def run_tensorflow(args):
    # Lazy import so PyTorch-only users don't need TF installed
    from models.cnn_with_Tensorfow import build_model

    print("[TensorFlow] building model …")
    train_gen, test_gen = get_data_tensorflow()
    model = build_model(num_classes=6)
    model.summary()

    if args.mode == 'eval':
        import tensorflow as tf
        model = tf.keras.models.load_model("rosly_model.keras")
        print("Loaded model from rosly_model.keras")

    trainer = TFTrainer(model, train_gen, test_gen, args.lr, args.epochs)

    if args.mode == 'train':
        trainer.train(save=True, plot=True)

    trainer.evaluate()


def main():
    args = parse_args()

    if args.framework == 'pytorch':
        run_pytorch(args)
    else:
        run_tensorflow(args)


if __name__ == '__main__':
    main()
