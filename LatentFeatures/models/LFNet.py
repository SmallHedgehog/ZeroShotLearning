import torch.nn as nn
from .vgg import vgg19_bn
from .vgg import vgg19

__all__ = ['FNet', 'Enet', 'ZNet', 'LFNet']


class ZNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(ZNet, self).__init__()
        self.trans = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_features)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.trans(x)
        return self.sigmoid(x)


class Enet(nn.Module):
    def __init__(self, in_features, num_attributes, augmented=False):
        super(Enet, self).__init__()
        if not augmented:
            self.matrix = nn.Linear(in_features, num_attributes)
        else:
            self.matrix = nn.Linear(in_features, 2 * num_attributes)

    def forward(self, x):
        return self.matrix(x)


class FNet(nn.Module):
    def __init__(self, pretrained=False, backbone='vgg19'):
        super(FNet, self).__init__()
        assert backbone in ('vgg19', 'vgg19bn')
        if backbone == 'vgg19':
            model = vgg19(pretrained=pretrained)
        else:
            model = vgg19_bn(pretrained=pretrained)
        del model.classifier[-1]
        self.model = model

    def forward(self, x):
        features, conv = self.model(x)
        return features, conv


class ZoomRegion(nn.Module):
    def __init__(self):
        super(ZoomRegion, self).__init__()

    def forward(self, x, region):
        raise NotImplementedError


class LFNet(nn.Module):
    def __init__(self, num_attributes, ZNets=None, pretrained=False,
                 backbone='vgg19', augmented=False, num_scales=1):
        super(LFNet, self).__init__()
        assert num_scales >= 1
        if ZNets is not None:
            assert len(ZNets) == (num_scales - 1)

        self.num_attributes = num_attributes
        self.augmented = augmented
        self.num_scales = num_scales

        self.FNets = nn.ModuleList([FNet(pretrained, backbone) for _ in range(self.num_scales)])
        self.ENets = nn.ModuleList([Enet(self.FNets[0].model.classifier[-3].out_features,
                           num_attributes, augmented) for _ in range(self.num_scales)])
        self.ZNets = ZNets
        if ZNets and not isinstance(ZNets, nn.ModuleList):
            self.ZNets = nn.ModuleList(ZNets)

        # self.zoom_region = ZoomRegion()

    def forward(self, x):
        semantic_attributes = []
        for idx in range(self.num_scales):
            features, conv_x = self.FNets[idx](x)
            semantic = self.ENets[idx](features)
            semantic_attributes.append(semantic)

            # # Zoomed
            # if (idx + 1) == self.num_scales:
            #     continue
            # conv_x = conv_x.view(conv_x.size(0), -1)
            # region = self.ZNets[idx](conv_x)
            # x = self.zoom_region(x, region)
        return semantic_attributes
