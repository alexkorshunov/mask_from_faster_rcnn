from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

import pdb

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input
        in_height = input
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__
    
class _Mask(nn.Module):
    """Mask layers"""
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(_fasterRCNN, self).__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.image_shape = image_shape
        
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_sizel=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, rois):
        
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
 
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)

        return x
    
    
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    print("ok")
    
    N = 50
    W, H = 200, 200
    M = 50
    
    gt_masks = np.zeros((2, H, W), dtype=np.int32)
    gt_masks[0, 50:150, 50:150] = 1
    gt_masks[1, 100:150, 50:150] = 1
    gt_boxes = np.asarray(
      [
        [20, 20, 100, 100, 1],
        [100, 100, 180, 180, 2]
      ])
    rois = gt_boxes[:, :4]
    print (rois)
    print(rois[0])
    
    mask = _Mask(256, 14, np.array([200, 200, 3]), 2)
    mask.eval()
    Mask = mask(gt_masks, rois)
    plt.figure(1)
    plt.imshow(Mask)
    plt.show()
    time.sleep(2)