import torch
import torch.nn as nn
import torch.nn.functional as F
import models.resnet as resnet
from operator import add
from functools import reduce

class FeatureExtractionHyperPixel(nn.Module):
    """ HyperPixel 
    """
    def __init__(self, hyperpixel_ids, feature_size, freeze=True):
        super().__init__()
        self.backbone = resnet.resnet101(pretrained=True)
        self.feature_size = feature_size
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        nbottlenecks = [3, 4, 23, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.hyperpixel_ids = hyperpixel_ids
    
    def forward(self, img):
        r"""Extract desired a list of intermediate features"""
        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                #if hid + 1 == max(self.hyperpixel_ids):
                #    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            feats[idx] = F.interpolate(feat, self.feature_size, None, 'bilinear', True)

        return feats

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels) :
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def foward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class SimpleResnet(nn.Module):
    """ Resnet backbone / Using Uncov
    """
    def __init__(self, feature_size=256, freeze=True, fuse_layer=False) :
        super(SimpleResnet, self).__init__()
        # backbone
        self.backbone = resnet.resnet101(pretrained=True)
        resnet_feat_dim = [256, 512, 1024, 2048]
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Fusing
        if fuse_layer :
            self.conv1 = nn.Sequential(
                nn.Conv2d(resnet_feat_dim[0], feature_size, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(resnet_feat_dim[1], feature_size, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(resnet_feat_dim[2], resnet_feat_dim[1], kernel_size=3, stride=1, padding=1, bias=False, groups=1),
                nn.BatchNorm2d(resnet_feat_dim[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(resnet_feat_dim[1], feature_size, kernel_size=3, stride=1, padding=1, bias=False, groups=1),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True)
            )
        else :
            self.conv1 = nn.Conv2d(resnet_feat_dim[0], feature_size, kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(resnet_feat_dim[1], feature_size, kernel_size=1, stride=1, padding=0)
            self.conv3 = nn.Conv2d(resnet_feat_dim[2], feature_size, kernel_size=1, stride=1, padding=0)
            self.conv4 = nn.Conv2d(resnet_feat_dim[3], feature_size, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        """ 
        Args
            x       : [B, 3, H, W]
        Return
            layerx  : [B, dim, h, w]
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        layer1 = self.backbone.layer1(x)        # [B, 256, h/4, w/4]
        layer2 = self.backbone.layer2(layer1)   # [B, 512, h/8, w/8]
        layer3 = self.backbone.layer3(layer2)   # [B, 1024, h/16, w/16]
        layer4 = self.backbone.layer4(layer3)   # [B, 2048, h/32, w/32]

        layer1 = self.conv1(layer1)   # hidden_dim
        layer2 = self.conv2(layer2)   # hidden_dim
        layer3 = self.conv3(layer3)   # hidden_dim
        layer4 = self.conv4(layer4)   # hidden_dim

        return [layer4, layer3, layer2, layer1]
        #return [layer4, layer3, layer2]