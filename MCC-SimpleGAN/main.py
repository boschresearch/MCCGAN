import argparse
import os
import prepareinput
import train

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='.', help="output folder to save the model after training")
parser.add_argument('--input_images', default='.', help="input images array path")
parser.add_argument('--countvec_path', default='.', help="path to the input count vector csv file")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--c_lambda", type=float, default=1.0, help="countloss co-efficient")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--discriminator_lr", type=float, default=0.0001, help="discriminator learning rate")
parser.add_argument("--generator_lr", type=float, default=0.0001, help="generator learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=200, help="latent space dimension")
parser.add_argument("--countvec_len", type=int, default=2, help="length of the multi-class count vector")
parser.add_argument("--img_size", type=int, default=128, help="image height/width dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
args = parser.parse_args()
print(args)
# Create output folder
if not os.path.exists(args.outf):
    os.makedirs(args.outf)


def main():
    # Configure dataloader
    dataloader, count_gt = prepareinput.prepareDataLoader(args.input_images, args.countvec_path, args.img_size,
                                                          args.batch_size)
    # Train the network
    train.train(args.img_size, args.countvec_len, args.channels, args.generator_lr, args.discriminator_lr, args.beta1,
                args.beta2, args.epochs, dataloader,
                args.batch_size, args.latent_dim, count_gt, args.c_lambda, args.outf)


if __name__ == "__main__":
    main()
