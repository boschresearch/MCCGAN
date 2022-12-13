import torch.nn as nn


class Discriminator(nn.Module):
    """
        The Discriminator architecture along with the count regression sub-network

        Args:
            channels (int)      : The number of channels in the input image
            img_size (int)      : The height/width of the input image
            countvec_len (int) : The length of the multi-class count vector
    """

    def __init__(self, channels, img_size, countvec_len):
        super(Discriminator, self).__init__()

        def disc_block(input_filters, output_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(input_filters, output_filters, 5, 2, 2), nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.45)]
            if bn:
                block.append(nn.BatchNorm2d(output_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *disc_block(channels, 16, bn=False),
            *disc_block(16, 32),
            *disc_block(32, 64),
            *disc_block(64, 128)
        )
        # height and width of downsampled image
        self.ds_size = img_size // 2 ** 4
        # adversarial output layer
        self.adv_layer = nn.Sequential(nn.Linear(128 * self.ds_size ** 2, 1), nn.Sigmoid())
        # count output layer
        self.count_layer = nn.Sequential(nn.Linear(128 * self.ds_size ** 2, countvec_len))

    def forward(self, img):
        """
            The input image is forwarded to predict the real/fake label and count vector

            Parameters:
                img (float)      : Input image tensor
            Returns:
                validity (float)    : The predicted label for images (real/fake)
                count (float)       : The predicted multi-class count vector
        """
        out_conv = self.conv_blocks(img)
        validity = self.adv_layer(out_conv.view(out_conv.shape[0], -1))
        count = self.count_layer(out_conv.view(out_conv.shape[0], -1))
        return validity, count
