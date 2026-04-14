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
        '--framework', type=str,
        choices=['pytorch', 'tensorflow', 'both'],
        required=True,
        help="Framework to use: 'pytorch', 'tensorflow', or 'both'",
    )
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument('--lr', type=float, default=0.0005,
                        help="Learning rate (default: 0.0005)")
    parser.add_argument('--wd', type=float, default=1e-4,
                        help="Weight decay – PyTorch only (default: 1e-4)")
    parser.add_argument('--patience', type=int, default=5,
                        help="Early stopping patience (default: 5)")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'],
                        default='train',
                        help="'train' or 'eval' (default: train)")
    parser.add_argument('--cuda', action='store_true',
                        help="Use GPU if available (PyTorch only)")
    return parser.parse_args()


def run_pytorch(args):
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"\n[PyTorch] device: {device}")

    train_loader, test_loader = get_data_pytorch()
    model = CNN1(num_classes=6).to(device)

    if args.mode == 'eval':
        model.load_state_dict(torch.load("rosly_mamekem_model.pth", map_location=device))
        print("Loaded weights from rosly_mamekem_model.pth")

    trainer = Trainer(model, train_loader, test_loader,
                      args.lr, args.wd, args.epochs, args.patience, device)

    if args.mode == 'train':
        trainer.train(save=False, plot=True)
        trainer.evaluate()
        try:
            torch.save(model.state_dict(), "rosly_mamekem_model.pth")
            print("Model saved → rosly_mamekem_model.pth")
        except Exception as e:
            print(f"Warning: could not save model — {e}")
    else:
        trainer.evaluate()


def run_tensorflow(args):
    from models.cnn_with_Tensorfow import build_model

    print("\n[TensorFlow] building model …")
    train_gen, test_gen = get_data_tensorflow()
    model = build_model(num_classes=6)
    model.summary()

    if args.mode == 'eval':
        import tensorflow as tf
        model = tf.keras.models.load_model("rosly_mamekem_model.keras")
        print("Loaded model from rosly_mamekem_model.keras")

    trainer = TFTrainer(model, train_gen, test_gen,
                        args.lr, args.epochs, args.patience)

    if args.mode == 'train':
        trainer.train(save=False, plot=True)
        trainer.evaluate()
        try:
            model.save('rosly_mamekem_model.keras')
            print("Model saved → rosly_mamekem_model.keras")
        except Exception as e:
            print(f"Warning: could not save model — {e}")
    else:
        trainer.evaluate()


def main():
    args = parse_args()

    if args.framework in ('pytorch', 'both'):
        run_pytorch(args)

    if args.framework in ('tensorflow', 'both'):
        run_tensorflow(args)


if __name__ == '__main__':
    main()
