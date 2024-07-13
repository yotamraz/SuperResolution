import os
import argparse


class Trainer:
    pass


def download_data(path_to_folder):
    pass

def get_args():
    parser = argparse.ArgumentParser(description="Image Super-Resolution")
    parser.add_argument("--path_to_folder", type=str, required=True, help="Path to the folder containing input images")
    parser.add_argument("--img_size", type=int, default=224, help="input image size")
    parser.add_argument("--scale_factor", type=int, default=4, help="scale factor for upscaling")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda", help="device for training (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs for training")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth", help="path to save the checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")
    parser.add_argument("--mode", type=str, default='train', help="running in Train or Test mode")

    return parser.parse_args()


if __name__ == "__main__":
    opt = get_args()
    if len(os.listdir(opt.path_to_folder)) == 0:
        download_data(opt.path_to_folder)
    trainer = Trainer(opt)
    if opt.mode == 'train':
        trainer.train()
    else:
        trainer.test()
