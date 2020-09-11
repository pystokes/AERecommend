#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models

logger = getLogger('ML_ENGINE')

from functools import wraps
import time

def stop_watch(func) :
    @wraps(func)
    def wrapper(*args, **kargs) :
        start = time.time()
        result = func(*args,**kargs)
        process_time =  time.time() - start
        print(f"{func.__name__} takes {process_time} sec")
        return result
    return wrapper

class AutoEncoder(nn.Module):

    def __init__(self, exec_type, config):

        super(AutoEncoder, self).__init__()

        self.exec_type = exec_type
        self.config = config

        # Set specific config
        self.l_relu_factor = config.model.leakyReLU_factor

        # Build network
        self.encoder = Encoder()
        self.decoder = Decoder(self.l_relu_factor)

    @stop_watch
    def forward(self, x):

        x = self.encoder(x)

        if self.exec_type == 'detect':
            return x

        x = self.decoder(x)        

        return x

    def init_weights(self):

        self.decoder.apply(self.init_conv_layer)
        
        logger.info(f'Initialized deconvolution layers')

    def init_conv_layer(self, layer):

        if isinstance(layer, nn.Conv2d):
            init.xavier_uniform_(layer.weight.data)
            layer.bias.data.zero_()

    def load_weights(self, trained_weights):

        state_dict = torch.load(trained_weights, map_location=lambda storage, loc: storage)
        try:
            # Load weights trained by single GPU into single GPU
            self.load_state_dict(state_dict) 
        except:
            # Load weights trained by multi GPU into single GPU
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v

            self.load_state_dict(new_state_dict)        


class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()

        #self.pretrained = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        #self.pretrained = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        #self.pretrained = nn.Sequential(*list(models.mobilenet_v2(pretrained=True).features.children())[:-2])
        self.pretrained = nn.Sequential(*list(models.squeezenet1_1(pretrained=True).features))

    def forward(self, x):
        
        return self.pretrained(x)


class Decoder(nn.Module):

    def __init__(self, l_relu_factor):

        super(Decoder, self).__init__()

        # ResNet50
        """
        self.deconv1 = UpSampleForResNet50(2048 // 2**0, 2048 // 2**2, l_relu_factor)
        self.deconv2 = UpSampleForResNet50(2048 // 2**2, 2048 // 2**4, l_relu_factor)
        self.deconv3 = UpSampleForResNet50(2048 // 2**4, 2048 // 2**6, l_relu_factor)
        self.deconv4 = UpSampleForResNet50(2048 // 2**6, 2048 // 2**8, l_relu_factor)
        self.deconv5 = UpSampleForResNet50(2048 // 2**8, 3, l_relu_factor)
        """

        # SqueezeNet
        self.deconv1 = UpSampleForSqueezeNet(512, 384, l_relu_factor, 27)
        self.deconv2 = UpSampleForSqueezeNet(384, 256, l_relu_factor, 55)
        self.deconv3 = UpSampleForSqueezeNet(256, 128, l_relu_factor)
        self.deconv4 = UpSampleForSqueezeNet(128, 96, l_relu_factor, 111)
        self.deconv5 = UpSampleForSqueezeNet(96, 3, l_relu_factor, 224)

    def forward(self, x):

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        
        return x


class UpSampleForResNet50(nn.Sequential):
    
    def __init__(self, in_feature, out_feature, leakyReLU_factor):

        super(UpSampleForResNet50, self).__init__()        

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(leakyReLU_factor)

    def forward(self, x):

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.conv(x)
        x = self.leakyrelu(x)

        return x

class UpSampleForSqueezeNet(nn.Sequential):
    
    def __init__(self, in_feature, out_feature, leakyReLU_factor, size=None):

        super(UpSampleForSqueezeNet, self).__init__()        

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(leakyReLU_factor)
        self.size = size

    def forward(self, x):

        if self.size:
            x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=True)

        x = self.conv(x)
        x = self.leakyrelu(x)

        return x


if __name__ == '__main__':

    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.resolve()
    sys.path.append(str(current_dir.joinpath('..')))

    import numpy as np
    from config import Config

    config = Config().build_config()

    model = AutoEncoder('detect', config)

    img_size = 224 # MobileNet and SqueezeNet
    img = torch.Tensor(np.random.randint(0, 255, img_size*img_size*3).reshape(1, 3, img_size, img_size))
    y = model(img)
    print(y.shape)
