import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

class encoder(nn.Module):

    def __init__(self,ef_dim=32,p_dim=256,use_bn=False):
        super(encoder,self).__init__()
        """
        Args:
            ef_dim: #channels in the filters
            p_dim: number of half space constraints/planes
            use_bn: whether to BN layer

        Going to return the plane parameters 
        """

        self.ef_dim = ef_dim
        self.p_dim = p_dim
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(1,self.ef_dim,4,2,1) #32x32x32
        if self.use_bn:
            self.conv1_bn = nn.BatchNorm2d(self.ef_dim)
        self.conv2 = nn.Conv2d(self.ef_dim,self.ef_dim*2,4,2,1) #64x16x16
        if self.use_bn:
            self.conv2_bn = nn.BatchNorm2d(self.ef_dim*2)
        self.conv3 = nn.Conv2d(self.ef_dim*2,self.ef_dim*4,4,2,1) #128x8x8
        if self.use_bn:
            self.conv3_bn = nn.BatchNorm2d(self.ef_dim*4)
        self.conv4 = nn.Conv2d(self.ef_dim*4,self.ef_dim*8,4,2,1) #256x4x4
        if self.use_bn:
            self.conv4_bn = nn.BatchNorm2d(self.ef_dim*8)
        self.conv5 = nn.Conv2d(self.ef_dim*8,self.ef_dim*8,4,1) #256x1x1
        if self.use_bn:
            self.conv5_bn = nn.BatchNorm2d(self.ef_dim*8)
        

        self.l1 = nn.Linear(self.ef_dim*8,self.ef_dim*16)
        if self.use_bn:
            self.l1_bn = nn.BatchNorm1d(self.ef_dim*16) 
        self.l2 = nn.Linear(self.ef_dim*16,self.ef_dim*32)
        if self.use_bn:
            self.l2_bn = nn.BatchNorm1d(self.ef_dim*32)
        self.l3 = nn.Linear(self.ef_dim*32,self.ef_dim*64)
        if self.use_bn:
            self.l3_bn = nn.BatchNorm1d(self.ef_dim*64)

        self.l_m = nn.Linear(self.ef_dim*64,self.p_dim*2)
        
        self.l_b = nn.Linear(self.ef_dim*64,self.p_dim)
        

            
    def forward(self, inp):
        """
        Define the forward pass here.
        Args: 
            inp : NxWxHx1
        """
        conv1 = self.conv1(inp)
        if self.use_bn:
            conv1_bn = self.conv1_bn(conv1)
            conv1_lr = F.leaky_relu(conv1_bn)
        else:
            conv1_lr = F.leaky_relu(conv1)
        

        conv2 = self.conv2(conv1_lr)
        if self.use_bn:
            conv2_bn = self.conv2_bn(conv2)
            conv2_lr = F.leaky_relu(conv2_bn)
        else:
            conv2_lr = F.leaky_relu(conv2)

        conv3 = self.conv3(conv2_lr)
        if self.use_bn:
            conv3_bn = self.conv3_bn(conv3)
            conv3_lr = F.leaky_relu(conv3_bn)
        else:
            conv3_lr = F.leaky_relu(conv3)

        conv4 = self.conv4(conv3_lr)
        if self.use_bn:
            conv4_bn = self.conv4_bn(conv4)
            conv4_lr = F.leaky_relu(conv4_bn)
        else:
            conv4_lr = F.leaky_relu(conv4)

        conv5 = self.conv5(conv4_lr)
        if self.use_bn:
            conv5_bn = self.conv5_bn(conv5)
            conv5_lr = F.leaky_relu(conv5_bn)
        else:
            conv5_lr = F.leaky_relu(conv5)


        conv5_lr = conv5_lr.view(-1,self.ef_dim*8) #256x1

        l1 = self.l1(conv5_lr)
        if self.use_bn:
            l1_bn = self.l1_bn(l1)
            l1_lr=F.leaky_relu(l1_bn) #512
        else:
            l1_lr = F.leaky_relu(l1) 
        
        l2 = self.l2(l1_lr)
        if self.use_bn:
            l2_bn = self.l2_bn(l2)
            l2_lr = F.leaky_relu(l2_bn) #1024
        else:
            l2_lr = F.leaky_relu(l2)
        
        l3 = self.l3(l2_lr)
        if self.use_bn:
            l3_bn = self.l3_bn(l3)
            l3_lr = F.leaky_relu(l3) #2048
        else:
            l3_lr = F.leaky_relu(l3)

        l_m = self.l_m(l3_lr) #512, (256 planes)
        l_b = self.l_b(l3_lr) #256, (256 planes)

        l_m = l_m.view(-1,2,self.p_dim)
        l_b = l_b.view(-1,1,self.p_dim)

        return l_m,l_b

    def init_weights(self):
        """
        Weight initialization goes here.
        """
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()



class generator(nn.Module):

    def __init__(self, p_dim, gf_dim, phase):
        super (generator,self).__init__()
        """
        Args: 
            p_dim: number of half space constraints/planes
            gf_dim: number of convexes for a shape
        """

        self.p_dim = p_dim
        self.gf_dim = gf_dim
        self.phase = phase

        convex_layer_weights = torch.zeros((self.p_dim, self.gf_dim))
        concave_layer_weights = torch.zeros((self.gf_dim, 1))
        self.convex_layer_weights = nn.Parameter(convex_layer_weights)
        self.concave_layer_weights = nn.Parameter(concave_layer_weights)
    
    def forward(self, points, plane_m, plane_b):

        if self.phase == 0:
            h1 = torch.matmul(points, plane_m) + plane_b
            h1 = torch.clamp(h1, min=0)

            #level 2
            h2 = torch.matmul(h1, self.convex_layer_weights)
            h2 = torch.clamp(1-h2, min=0, max=1)

            #level 3
            h3 = torch.matmul(h2, self.concave_layer_weights)
            h3 = torch.clamp(h3, min=0, max=1)
            h3_max,_ = torch.max(h2, axis=2, keepdims=True) #torch.max returns a tuple : (value,indices)

            return h3, h3_max, h2, self.convex_layer_weights, self.concave_layer_weights
        
        else:
            h1 = torch.matmul(points, plane_m) + plane_b
            h1 = torch.clamp(h1, min=0)

            #level 2
            h2 = torch.matmul(h1, (self.convex_layer_weights>0.01).float())

            #level 3
            h3,_ = torch.min(h2, axis=2, keepdims=True) #torch.min returns a tuple : (value,indices)
            h3_ = h3.detach()
            h3_01 = torch.clamp(1-h3_, min=0, max=1)

            return h3, h3_01, h2, self.convex_layer_weights, None

    
    def init_weights(self):
        nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)
        nn.init.normal_(self.concave_layer_weights, mean=1e-5, std=0.02)

class BSP_Model(nn.Module):

    def __init__(self,p_dim=256,ef_dim=32,gf_dim=64,phase=1,use_bn=False):
        super(BSP_Model,self).__init__()
        """
        Args:
            p_dim: number of half space constraints/plane
            ef_dim: #channels in the filters
            gf_dim: number of convexes
            use_bn: add BN layer or not
        """

        self.p_dim = p_dim
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim
        self.phase = phase
        self.use_bn = use_bn#binary

        self.encoder = encoder(self.ef_dim,self.p_dim,self.use_bn)
        self.generator = generator(self.p_dim,self.gf_dim,self.phase)
 
    def forward(self, inp, coords,mode="train"):
        """
        Define the forward pass here.

        Args:
            inp: shape, to obtain plane parameters
            coords: (x,y) locations in the 2D grid.
            mode: train or test 
        """

        if mode=="train":
            #get the estimated plane parameters
            E_m, E_b = self.encoder.forward(inp)
            # get in/out of shape and in/out of convext -- verify this once by checking dimensions. %%%%
            G, _, G2, cw2, cw3 = self.generator.forward(coords,E_m,E_b)
            return G, G2, cw2, cw3

        else:
            sE_m, sE_b = self.encoder.forward(inp)
            sG, sG_max, sG2, cw2, _ = self.generator.forward(coords, sE_m, sE_b)

            return sG, sG_max, sG2, cw2, sE_m, sE_b

		

        

    def init_weights(self):
        """
        Weight initialization goes here.
        """
  
        self.encoder.init_weights()
        self.generator.init_weights()
