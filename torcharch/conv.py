from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramid(nn.Module):
    """Feature Pyramid Network

    Introduced by Facebook Research in https://arxiv.org/pdf/1612.03144.pdf,
    this is a generic approach to combine bottom-up and top-down feature maps
    from any CNN backbone so they are more suitable for multi-scale tasks like
    object detection
    """

    def __init__(self, backbone, depth2channels, pyramid_feats=256):
        """Define relevant feature maps from the backbone

        Typical CNN architectures increase the number of channels while
        decreasing the spatial dimensions as you go deeper in the network.
        The FPN paper (section 3) recommends defining one pyramid level per
        "stage" of the network, where a stage consists of multiple feature
        maps with the same number of channels. In this case, the last layer of
        each stage should be used.

        For instance ``mobilenet_v2`` has 8 stages (designated by the dashed
        separators). We ignore the unusual layer 0 since it has more channels
        than layer 1

        Network Layer    Number of Channels    Final Stage Layer
        ========================================================
        0                32
        --------------------------------------------------------
        1                16                    *
        --------------------------------------------------------
        2                24
        3                24                    *
        --------------------------------------------------------
        4                32
        5                32
        6                32                    *
        --------------------------------------------------------
        7                64
        8                64
        9                64
        10               64                    *
        --------------------------------------------------------
        11               96
        12               96
        13               96                    *
        --------------------------------------------------------
        14               160
        15               160
        16               160                   *
        --------------------------------------------------------
        17               320                   *
        --------------------------------------------------------
        18               1280                  *

        :param nn.Module backbone: Callable CNN model. Must define an iterable
            ``features`` attribute
        :param dict{int: int} depth2channels: Specifies what layers from the
            list of features should be used to build the pyramid. For example
            using the mobilenetv2 table above, the dict would be ``{1: 16,
            3: 24, 6: 32, 10: 64, 13: 96, 16: 160, 17: 320, 18: 1280}``. Note
            the values that give the number of channels are not strictly
            necessary to construct an FPN, however they are used in
            ``forward()`` as a useful runtime check
        :param int pyramid_feats: Number of channels in the final feature maps
        """
        super(FeaturePyramid, self).__init__()

        # Validate backbone (tough to test for iterability, so just use try-except)
        # See https://stackoverflow.com/q/1952464/7446465 for discussion of __iter__ vs. __getitem__
        features_attr = getattr(backbone, 'features', None)
        try:
            _ = iter(features_attr)
        except TypeError as e:
            raise AttributeError(f'Backbone {backbone} must define an iterable set of features') from e
        if not isinstance(backbone, nn.Module):
            raise TypeError(f'Backbone must be nn.Module, got {type(backbone)}')

        # Transfer parameters to attributes
        self.pyramid_feats = pyramid_feats
        self.depth2channels = depth2channels
        self.backbone = backbone

        # Generate corresponding channel reductions for lateral connections
        self.laterals = nn.ModuleList([nn.Conv2d(c, pyramid_feats, kernel_size=1)
                                       for c in self.depth2channels.values()])

        # Anti-aliasing convolution to smooth effects of upsampling
        # Padding required to accommodate spatial dimensions smaller than the kernel size
        self.smooth = nn.Conv2d(pyramid_feats, pyramid_feats, kernel_size=3, padding=1)

    def forward(self, imgs):
        """Generate fused feature pyramid

        Supplemental figures showing operations to fuse bottom-up and top-down
        https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c

        Some notes on upsampling procedure
            * ``F.upsample`` has been deprecated in favor of ``F.interpolate``
            * Bilinear mode is preferred since using nearest (the paper's
              recommendation) may not match the lateral dimensions when they
              are odd. This is noted in other Github implementations
            * ``align_corners`` must be explicitly given to avoid warnings

        :param torch.Tensor imgs: Raw pixel data with shape ``(N, C, H, W)``
        :return list(torch.Tensor) pyramid: Smoothed feature pyramid in order
            of high to low spatial resolutions
        """

        # Bottom-up generation of sequentially smaller + deeper feature maps
        bottom_up = []
        for i, layer in enumerate(self.backbone.features):
            imgs = layer(imgs)
            if i in self.depth2channels:
                if imgs.shape[1] != self.depth2channels[i]:
                    raise RuntimeError(f'Expected {self.depth2channels[i]} channels for layer {i}, got {imgs.shape[1]}')
                bottom_up.append(imgs)

        # Start top-down progression with 1x1 conv on coarsest bottom-up layer
        top_down = [self.laterals[-1](bottom_up[-1])]

        # Iteratively apply upsampling and lateral fusion to construct remainder of top-down
        reverse_layers = range(2, len(bottom_up) + 1)
        for i in reverse_layers:
            lateral = bottom_up[-i]
            _, _, h, w = lateral.shape
            upsampled = F.interpolate(top_down[-1], size=(h, w), mode='bilinear', align_corners=False)
            top_down.append(upsampled + self.laterals[-i](lateral))

        # Smoothing of final pyramid
        top_down.reverse()
        pyramid = [self.smooth(t) for t in top_down]
        return pyramid


class SpatialPyramidPooling(nn.Module):
    """Generate fixed length representation regardless of image dimensions

    Based on the paper "Spatial Pyramid Pooling in Deep Convolutional Networks
    for Visual Recognition" (https://arxiv.org/pdf/1406.4729.pdf)

    :param [int] num_pools: Number of pools to split each input feature map into.
        Each element must be a perfect square in order to equally divide the
        pools across the feature map. Default corresponds to the original
        paper's implementation
    :param str mode: Specifies the type of pooling, either max or avg
    """

    def __init__(self, num_pools=[1, 4, 16], mode='max'):
        super(SpatialPyramidPooling, self).__init__()
        self.name = 'SpatialPyramidPooling'
        if mode == 'max':
            pool_func = nn.AdaptiveMaxPool2d
        elif mode == 'avg':
            pool_func = nn.AdaptiveAvgPool2d
        else:
            raise NotImplementedError(f"Unknown pooling mode '{mode}', expected 'max' or 'avg'")
        self.pools = nn.ModuleList([])
        for p in num_pools:
            side_length = sqrt(p)
            if not side_length.is_integer():
                raise ValueError(f'Bin size {p} is not a perfect square')
            self.pools.append(pool_func(int(side_length)))

    def forward(self, feature_maps):
        """Pool feature maps at different bin levels and concatenate

        :param torch.tensor feature_maps: Arbitrarily shaped spatial and
            channel dimensions extracted from any generic convolutional
            architecture. Shape ``(N, C, H, W)``
        :return torch.tensor pooled: Concatenation of all pools with shape
            ``(N, C, sum(num_pools))``
        """
        assert feature_maps.dim() == 4, 'Expected 4D input of (N, C, H, W)'
        batch_size = feature_maps.size(0)
        channels = feature_maps.size(1)
        pooled = []
        for p in self.pools:
            pooled.append(p(feature_maps).view(batch_size, channels, -1))
        return torch.cat(pooled, dim=2)
