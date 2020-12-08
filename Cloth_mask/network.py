
import torch
import torch.nn as nn
from torch.nn import init
import functools


def init_weights(net, init_type="kaiming", init_gain=0.02):
    """ Initialize network weights.
    Parameters:
        net (network): network to be initialized
        init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print("[%s] Initialize network with %s." % (net.__name__(), init_type))
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, device_ids=None):
    """ Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network): the network to be initialized
        init_type (str): the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float): scaling factor for normal, xavier and orthogonal.
        device_ids (int list): which GPUs the network runs on: e.g., 0, 1, 2
    Return an initialized network.
    """
    if torch.cuda.is_available():
        if device_ids is None:
            device_ids = []
        net.to(device_ids[0])
        if len(device_ids) > 1:
            net = nn.DataParallel(net, device_ids=device_ids)  # multi-GPUs
    else:
        net.to("cpu")
    init_weights(net, init_type=init_type, init_gain=init_gain)  # Weight initialization

    return net


class ConvBlock(nn.Module):
    """ Define a convolution block (conv + norm + actv). """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding_type="zero", padding=1, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(ConvBlock, self).__init__()

        # use_bias setup
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv_block = []
        # Padding option
        p = 0
        if padding_type == "reflect":
            self.conv_block += [nn.ReflectionPad2d(padding)]
        elif padding_type == "replicate":
            self.conv_block += [nn.ReplicationPad2d(padding)]
        elif padding_type == "zero":
            p = padding
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        self.conv_block += [nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=p,
                                      dilation=dilation,
                                      bias=use_bias)]
        self.conv_block += [norm_layer(num_features=out_channels)] if norm_layer is not None else []
        self.conv_block += [activation(inplace=True)] if activation is not None else []
        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        return self.conv_block(x)


class ConvBlock2(nn.Module):
    """ Define a double convolution block (conv + norm + actv). """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding_type="zero", padding=1, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, activation=nn.ReLU):
        super(ConvBlock2, self).__init__()

        self.conv_block = []
        self.conv_block += [ConvBlock(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, padding_type=padding_type, padding=padding,
                                      dilation=dilation, stride=1,
                                      norm_layer=norm_layer, activation=activation),
                            ConvBlock(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, padding_type=padding_type, padding=padding,
                                      dilation=dilation, stride=stride,
                                      norm_layer=norm_layer, activation=activation)]

        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    """ Define a convolution block with upsampling. """
    def __init__(self, in_channels, out_channels, scale_factor=2,
                 kernel_size=(3, 3), padding=(1, 1),
                 mode="nearest"):
        super(UpConv, self).__init__()

        self.up_conv = [nn.Upsample(scale_factor=scale_factor, mode=mode),
                        ConvBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size)]
        self.up_conv = nn.Sequential(*self.up_conv)

    def forward(self, x):
        return self.up_conv(x)


class UNet(nn.Module):
    """ U-Net architecture """
    def __init__(self, in_channels, out_channels, num_features=32,
                 feature_mode="fixed", pool="avg", upsample_mode="bilinear"):
        """ Initialize the U-Net architecture
        Parameters:
            in_channels (int): the number of input channels
            out_channels (int): the number of output channels
            num_features (int): the number of features in the first layer
            feature_mode (str): feature-increasing mode along the depth ("fixed" or "pyramid")
            pool (str): pooling method ("max" or "avg")
            upsample_mode (str): upsampling method ("bilinear" or "nearest")
        """
        super(UNet, self).__init__()

        # Assert
        assert feature_mode in ["fixed", "pyramid"]
        assert pool in ["max", "avg"]
        assert upsample_mode in ["nearest", "bilinear"]

        # Layer definition
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) if pool == "max" \
            else nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.enc_conv0 = ConvBlock2(in_channels=in_channels, out_channels=num_features)

        if feature_mode == "fixed":
            self.enc_conv1 = ConvBlock2(in_channels=num_features, out_channels=num_features)
            self.enc_conv2 = ConvBlock2(in_channels=num_features, out_channels=num_features)
            self.enc_conv3 = ConvBlock2(in_channels=num_features, out_channels=num_features)
            self.enc_conv4 = ConvBlock2(in_channels=num_features, out_channels=num_features)

            self.up_conv4 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode)
            self.up_conv3 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode)
            self.up_conv2 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode)
            self.up_conv1 = UpConv(in_channels=num_features, out_channels=num_features, mode=upsample_mode)

            self.dec_conv4 = ConvBlock2(in_channels=2 * num_features, out_channels=num_features)
            self.dec_conv3 = ConvBlock2(in_channels=2 * num_features, out_channels=num_features)
            self.dec_conv2 = ConvBlock2(in_channels=2 * num_features, out_channels=num_features)
            self.dec_conv1 = ConvBlock2(in_channels=2 * num_features, out_channels=num_features)
        else:
            self.enc_conv1 = ConvBlock2(in_channels=1 * num_features, out_channels=2 * num_features)
            self.enc_conv2 = ConvBlock2(in_channels=2 * num_features, out_channels=4 * num_features)
            self.enc_conv3 = ConvBlock2(in_channels=4 * num_features, out_channels=8 * num_features)
            self.enc_conv4 = ConvBlock2(in_channels=8 * num_features, out_channels=16 * num_features)

            self.up_conv4 = UpConv(in_channels=16 * num_features, out_channels=8 * num_features, mode=upsample_mode)
            self.up_conv3 = UpConv(in_channels=8 * num_features, out_channels=4 * num_features, mode=upsample_mode)
            self.up_conv2 = UpConv(in_channels=4 * num_features, out_channels=2 * num_features, mode=upsample_mode)
            self.up_conv1 = UpConv(in_channels=2 * num_features, out_channels=1 * num_features, mode=upsample_mode)

            self.dec_conv4 = ConvBlock2(in_channels=16 * num_features, out_channels=8 * num_features)
            self.dec_conv3 = ConvBlock2(in_channels=8 * num_features, out_channels=4 * num_features)
            self.dec_conv2 = ConvBlock2(in_channels=4 * num_features, out_channels=2 * num_features)
            self.dec_conv1 = ConvBlock2(in_channels=2 * num_features, out_channels=1 * num_features)

        self.conv_1x1 = nn.Conv2d(in_channels=num_features, out_channels=out_channels, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoding path
        enc0 = self.enc_conv0(x)
        enc1 = self.enc_conv1(self.pool(enc0))
        enc2 = self.enc_conv2(self.pool(enc1))
        enc3 = self.enc_conv3(self.pool(enc2))
        enc4 = self.enc_conv4(self.pool(enc3))

        # Decoding path with skip connection
        dec = self.dec_conv4(torch.cat((self.up_conv4(enc4), enc3), dim=1))
        dec = self.dec_conv3(torch.cat((self.up_conv3(dec), enc2), dim=1))
        dec = self.dec_conv2(torch.cat((self.up_conv2(dec), enc1), dim=1))
        dec = self.dec_conv1(torch.cat((self.up_conv1(dec), enc0), dim=1))

        # 1x1 Conv
        y = self.conv_1x1(dec)
        y = self.softmax(y)

        return y