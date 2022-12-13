import torch
import torch.nn as nn


class Dense_Block(nn.Module):
    """
        The Denseblock architecture with 3 dense layers

        Args:
            input_channels (int) : The number of input channels
    """

    def __init__(self, input_channels):
        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=input_channels)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv_1 = self.relu(self.conv1(self.bn(x)))
        conv_2 = self.relu(self.conv2(conv_1))
        c2_dense = self.relu(torch.cat([conv_1, conv_2], 1))
        conv_3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv_1, conv_2, conv_3], 1))
        conv_4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv_1, conv_2, conv_3, conv_4], 1))
        return c4_dense


class Transition_Layer(nn.Module):
    """
        The transition layer architecture with a 1 X 1 convolution layer and upsampling

        Args:
            input_channels (int)  : The number of input channels
            output_channels (int) : The number of output channels
    """

    def __init__(self, input_channels, output_channels):
        super(Transition_Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
        # 1x1 convolution
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, bias=False)
        # upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        bn = self.bn(self.relu(self.conv(x)))
        out = self.upsample(bn)
        return out


class Generator(nn.Module):
    """
        The Generator architecture with dense blocks

        Args:
            img_size (int)      : The height/width of the input image
            channels (int)      : The input image channnels
            latent_dim (int)    : The latent space dimension
            countvec_len (int)  : The length of the multi-class count vector
    """

    def __init__(self, img_size, channels, latent_dim, countvec_len):
        super(Generator, self).__init__()
        # Initial size before upsampling
        self.init_size = img_size // 4
        self.fc = nn.Sequential(nn.Linear(latent_dim + countvec_len, 128 * self.init_size ** 2), nn.ReLU())

        # Create Dense Blocks
        self.denseblock1 = self._make_dense_block(Dense_Block, 128)
        self.denseblock2 = self._make_dense_block(Dense_Block, 128)

        # Create Transition Layers
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels=256, out_channels=128)
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels=256, out_channels=128)

        # Convolution layers
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128, 0.8),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    @staticmethod
    def _make_dense_block(block, in_channels):
        layers = [block(in_channels)]
        return nn.Sequential(*layers)

    @staticmethod
    def _make_transition_layer(layer, in_channels, out_channels):
        modules = [layer(in_channels, out_channels)]
        return nn.Sequential(*modules)

    def forward(self, noise, count):
        """
            The randomly sampled noise and the multi-class count vector is forwarded to generate image
            Parameters:
                noise (float)    : Randomly sampled noise
                count (float)    : Input multi-class count vector
            Returns:
                image (float)    : The generated image
        """
        gen_input = torch.cat((count, noise), -1)
        # output from fully connected layer
        fc_out = self.fc(gen_input.type(torch.cuda.FloatTensor))
        # output from dense block 1
        dense_block1_out = self.denseblock1(fc_out.view(fc_out.shape[0], 128, self.init_size, self.init_size))
        # output from transition layer 1
        transition1_out = self.transitionLayer1(dense_block1_out)
        # output from dense block 1
        dense_block2_out = self.denseblock2(transition1_out)
        # output from transition layer 2
        transition2_out = self.transitionLayer2(dense_block2_out)
        # output from last two convolution layers
        img = self.conv_blocks(transition2_out)
        return img
