import torch
from torch.autograd import Variable
import numpy as np
import random
from generator import Generator
from discriminator import Discriminator

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def train(img_size, countvec_len, channels, gen_lr, disc_lr, beta1, beta2, epochs, dataloader, batch_size, latent_dim,
          count_gt, c_lambda, outf):
    """
        To train the generator and discriminator.

        Parameters:
            img_size (int)      : Input image dimensions
            countvec_len (int)  : Length of the multi-class count vector
            channels (int)      : Input image channels
            gen_lr (float)      : Learning rate of generator
            disc_lr (float)     : Learning rate of discriminator
            beta1 (float)       : Adam: decay of first order momentum of gradient
            beta2 (float)       : Adam: decay of second order momentum of gradient
            epochs (int)        : Number of epochs to train
            dataloader          : Pytorch dataloader
            batch_size (int)    : Batch size for training
            latent_dim (int)    : Dimension of latent space
            count_gt (float)    : list of ground truth count vectors
            c_lambda (float)    : Count loss co-efficient
            outf (float)        : Output folder to save the model after training
    """
    # Initialize generator and discriminator
    generator = Generator(img_size, channels, countvec_len, latent_dim)
    discriminator = Discriminator(channels, img_size, countvec_len)
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    count_loss = torch.nn.MSELoss()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        count_loss.cuda()
    # Optimizer for generator and discriminator
    opt_G = torch.optim.Adam(generator.parameters(), lr=gen_lr, betas=(beta1, beta2))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(beta1, beta2))
    for epoch in range(epochs):
        for i, (imgs, count) in enumerate(dataloader):
            # Adversarial ground truths
            valid = Variable(LongTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(LongTensor(batch_size, 1).fill_(0.0), requires_grad=False)
            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            count = Variable(count.type(LongTensor))
            # Sample noise and count vector as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
            gen_labels_value = np.asarray(random.sample(count_gt, batch_size))
            gen_labels = Variable(FloatTensor(gen_labels_value))
            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            # Prediction for real images
            real_pred, real_count = discriminator(real_imgs)
            # Prediction for fake images
            fake_images = generator(z, gen_labels)
            # Count loss for real images
            d_real_countloss = c_lambda * (count_loss(real_count.type(FloatTensor), count.type(FloatTensor)))
            d_real_loss = (adversarial_loss(real_pred.type(FloatTensor), valid.type(FloatTensor))) + d_real_countloss
            fake_pred, fake_count = discriminator(fake_images.detach())
            # Count loss for generated images
            d_fake_countloss = c_lambda * (count_loss(fake_count.type(FloatTensor), gen_labels.type(FloatTensor)))
            d_fake_loss = (adversarial_loss(fake_pred.type(FloatTensor), fake.type(FloatTensor))) + d_fake_countloss
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            opt_D.step()
            # -----------------
            #  Train Generator
            # -----------------
            opt_G.zero_grad()
            validity, pred_count = discriminator(fake_images)
            # Count loss for the generated images
            g_countloss = c_lambda * (count_loss(pred_count, gen_labels.type(FloatTensor)))
            g_loss = adversarial_loss(validity, valid.type(FloatTensor)) + g_countloss
            g_loss.backward()
            opt_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
    # Save the generator and discriminator model after training
    torch.save(generator.state_dict(), '%s/Gen_epoch_%d.pth' % (outf, epoch))
    torch.save(discriminator.state_dict(), '%s/Disc_epoch_%d.pth' % (outf, epoch))
