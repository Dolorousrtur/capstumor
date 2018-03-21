import sys

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


from torch.autograd import Variable
# from torch.optim import Adam
# from torchnet.engine import Engine
# from torchnet.logger import VisdomPlotLogger, VisdomLogger
# from torchvision.utils import make_grid
# from torchvision.datasets.mnist import MNIST
# from tqdm import tqdm
# import torchnet as tnt

def squash(x, dim=-1):
    squared_norm = (x**2).sum(dim=dim, keepdim=True)
    out = (squared_norm * x) / ((1 + squared_norm) * torch.sqrt(squared_norm))
    return out

class PrimaryCaps(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_capsules):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList()
        
        for i in range(num_capsules):
            self.capsules.append(nn.Conv2d(in_channels, out_channels, 9, stride=2))
            
    def forward(self, x):
        out = []
        for cap in self.capsules:
            out.append(cap(x))

        out = torch.stack(out, dim=1)
        out = out.view(x.size()[0], len(self.capsules), -1)
        out = out.permute(0,2,1)
        out = squash(out, dim=-1)
        
        return out
    
class ClassesCaps(torch.nn.Module):
    def __init__(self, in_channels, out_channels, in_vectors, n_capsules, n_iters):
        super(ClassesCaps, self).__init__()
        
        self.weights = nn.Parameter(torch.randn(n_capsules, in_vectors, in_channels, out_channels))
        self.n_iters = n_iters
        
        self.in_vectors = in_vectors
        self.n_capsules = n_capsules
        
    def forward(self, x):
        predictions = x[None, :, :, None, :] @ self.weights[:, None, :, :, :]
        predictions = predictions.squeeze(3)
        predictions = predictions.permute(2,0,1,3)
        
        
#         print('predictions', predictions.shape)
        B = Variable(torch.zeros(self.in_vectors, self.n_capsules, x.shape[0])).cuda()
#         print('B', B.shape)
        
        for i in range(self.n_iters):
            C = torch.nn.Softmax(dim=0)(B)
#             print('C', C.shape)
            S = (predictions*C[:,:,:,None]).sum(dim=0, keepdim=True)
#             print('S', S.shape)
            S = squash(S, dim=3)
            

            if i != self.n_iters - 1:
                B_del = (predictions*S).sum(dim=-1)
#                 print('B_del', B_del.shape)
                B = B + B_del
                
        S = S.squeeze(0)
        S = S.permute(1,0,2)
        return S
    
    
class Decoder(torch.nn.Module):    
    def __init__(self, in_features, n_classes, img_height, img_width):
        super(Decoder, self).__init__()
        self.shape = (img_height, img_width)
        out_features = img_height * img_width
        seq = []
        seq.append(nn.Linear(in_features*n_classes, 512))
        seq.append(nn.ReLU())
        seq.append(nn.Linear(512, 1024))
        seq.append(nn.ReLU())
        seq.append(nn.Linear(1024, img_height*img_width))
        seq.append(nn.Sigmoid())
        
        self.pipe = nn.Sequential(*seq)
        
    def forward(self, x):
        x = self.pipe(x)
#         x = x.view(x.shape[0], *self.shape)
        return(x)


class CapsuleNet(nn.Module):
    def __init__(self, img_shape, n_pcaps, n_ccaps, conv_channels=64, n_iterations=3):
        super(CapsuleNet, self).__init__()
        
        height, width = img_shape
        
        height_dn = (height - 8*2) // 2
        width_dn = (height - 8*2) // 2
        
        self.n_ccaps = n_ccaps
        

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=9, stride=1)
        
        self.primary_capsules = PrimaryCaps(in_channels=conv_channels, out_channels=32, num_capsules=n_pcaps)

        self.digit_capsules = ClassesCaps(in_channels=8, out_channels=16, in_vectors=32 * height_dn * width_dn, 
                                              n_capsules=n_ccaps, n_iters=n_iterations)

        self.decoder = Decoder(in_features=16, n_classes=n_ccaps, img_height=height, img_width=width)

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
#         print(x.shape)
        x = self.primary_capsules(x)
#         print(x.shape)
        
        x = self.digit_capsules(x)
        x = x.squeeze(2).squeeze(2)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        
        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.sparse.torch.eye(self.n_ccaps)).cuda().index_select(dim=0, index=max_length_indices)
        
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions
        
        
        
class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)
    
    

        