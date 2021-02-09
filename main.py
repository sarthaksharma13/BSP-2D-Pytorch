import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import torch
import argparse
import h5py
import random
from torch.utils.tensorboard import SummaryWriter
from visualize import *


from model import BSP_Model

parser = argparse.ArgumentParser()

parser.add_argument("--phase", type=int, default=0, help="phase of training, 0 for continuous, 1 for discrete")
parser.add_argument("--sample_vox_size", default=64, help="Voxel resolution for coarse-to-fine training")
parser.add_argument("--gf_dim",default=64,help="number of convexes for a shape")
parser.add_argument("--p_dim",default=256,help="number of half space constraints")
parser.add_argument("--ef_dim",default=32,help="filter channels")

parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("--shape_batch_size", type=int, default=20, help="batch size")
parser.add_argument("--LR",type=float, default=0.00002, help="learning rate")
parser.add_argument("--optimizer", type=str, default="adam", help="type of optimizer")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--use_bn",default=False,help=" use batch norm or not")

parser.add_argument("--exp_dir", type=None, default="tmp", help="directory to save checkpoints, outputs etc")
parser.add_argument("--model_path", type=str, default="", help="pre-trained model path")

args = parser.parse_args()

op_dir = os.path.join("exp", args.exp_dir,"phase_" + str(args.phase),"op")
checkpoint_dir = os.path.join("exp", args.exp_dir,"phase_" + str(args.phase),"checkpoints")
log_dir = os.path.join("exp", args.exp_dir,"phase_" + str(args.phase),"logs")
if not os.path.exists(op_dir):
    os.makedirs(op_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

data_hdf5_name = os.path.join('./data','complex_elements.hdf5')
data_hdf5_val_name = os.path.join('./data','complex_elements_val.hdf5')
'''
data_dict has only one (key,value) pair.
data_voxels is a numpy array, with shape NxWxHx1, where
N: number of 2D shapes
W: width of image
H: height of image
greyscale image, hence 1
'''

if os.path.exists(data_hdf5_name) and os.path.exists(data_hdf5_val_name):
    data_dict = h5py.File(data_hdf5_name, 'r')
    data_voxels = data_dict['pixels'][:]
    data_dict_val = h5py.File(data_hdf5_val_name,'r')
    data_voxels_val = data_dict_val['pixels'][:] 
else:
    print("create data first !")


n_epochs = args.epochs
shape_batch_size = args.shape_batch_size
shape_batch_size_val = data_voxels_val.shape[0]

num_shapes = len(data_voxels)
shape_index_list = np.arange(num_shapes)
dim = args.sample_vox_size #2D grid dimension, width=height
point_batch_size = dim*dim #number of points

# Get the model and initialize weights
bsp_2d = BSP_Model(p_dim=256,ef_dim=32,gf_dim=64,phase=args.phase, use_bn=args.use_bn)
if  os.path.exists(args.model_path):
    print("loading pretrained weights !")
    bsp_2d.load_state_dict(torch.load(args.model_path))
else:
    bsp_2d.init_weights()
    

bsp_2d.cuda()

# load a predefined model if it exist

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(bsp_2d.parameters(), lr=args.LR, betas=(args.beta1, 0.999))
else:
    optimizer = torch.optim.SGD(bsp_2d.parameters(), lr=args.LR)

loss_best = 1e7
epoch_best = 0
writer = SummaryWriter(log_dir=log_dir,comment="LR is" + str(args.LR) + "batch_size is" + str(args.shape_batch_size))
mode="train"
for epoch in range(n_epochs):
    
    #random shuffle the shape, and then iterate over all batch wise
    random.shuffle(shape_index_list)
    avg_loss_sp = 0
    avg_loss = 0
    avg_num = 0

    for itt in range(int(num_shapes/shape_batch_size)):
        batch_index_list = shape_index_list[itt*shape_batch_size:(itt+1)*shape_batch_size]
        shapes = data_voxels[batch_index_list,:,:,:].transpose(0,3,1,2)

        #get coords	: reschaling and shifting by 0.5 %%%% ?
        coords = np.zeros([dim,dim,2],np.float32)
        for i in range(dim):
            for j in range(dim):
                coords[i,j,0] = i
                coords[i,j,1] = j
        coords = (coords+0.5)/dim-0.5
        coords = np.tile(np.reshape(coords,[1,point_batch_size,2]),[shape_batch_size,1,1])
        coord_vals = np.reshape(shapes,(shape_batch_size,point_batch_size,1)) # just binary {0,1}

        #convert coord and shape to torch cuda tensor
        coords = torch.from_numpy(coords).float().to('cuda')
        shapes = torch.from_numpy(shapes).float().to('cuda')
        coords_val = torch.from_numpy(coord_vals).float().to('cuda')
    
        bsp_2d.train()
        #forward the shape and coordinates
        # shape would be used for getting the plane parameters out
        # coord for getting the in/out vals
        G, G2, cw2, cw3 = bsp_2d.forward(shapes, coords,mode="train")

        #L_recon + L_W + L_T
        #G2 - network output (convex layer), the last dim is the number of convexes
        #G - network output (final output)
        #point_value - ground truth inside-outside value for each point
        #cw2 - connections T
        #cw3 - auxiliary weights W

        # compute loss and backprop.
        loss_sp = None
        loss = None
        if args.phase == 0:
            #self.loss_sp = tf.reduce_mean(tf.square(self.point_value - self.G))
            #self.loss = self.loss_sp + tf.reduce_sum(tf.abs(self.cw3-1))*1.0 + (tf.reduce_sum(tf.maximum(self.cw2-1,0)) - tf.reduce_sum(tf.minimum(self.cw2,0)))*1.0
            loss_sp = torch.mean((coords_val - G)**2)
            torch.clamp(cw2-1, min=0)
            loss = loss_sp + torch.sum(torch.abs(cw3-1)) + (torch.sum(torch.clamp(cw2-1, min=0) - torch.clamp(cw2, max=0)))
        else:
            loss_sp = torch.mean((1-coords_val)*(1-torch.clamp(G, max=1)) + coords_val*(torch.clamp(G, min=0)))
            loss = loss_sp
        
        bsp_2d.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss_sp += loss_sp.item()
        avg_loss += loss.item()
        
        
  
    avg_loss_sp = avg_loss_sp/itt
    avg_loss = avg_loss/itt
    writer.add_scalar('Average loss shape', avg_loss_sp, epoch)
    writer.add_scalar('Average loss total', avg_loss, epoch)
    
    if avg_loss < loss_best:
        # save the checkpoint
        epoch_best = epoch
        loss_best = avg_loss
        torch.save(bsp_2d.state_dict(),os.path.join(checkpoint_dir,'best_model.pth'))


    if epoch%100==0:
        print("==> Epoch: ", epoch, " ,average loss shape: ", avg_loss_sp, " ,average loss total", avg_loss)
        bsp_2d.eval()
        
        shapes = data_voxels_val.transpose(0,3,1,2)
        
        coords = np.zeros([dim,dim,2],np.float32)
        for i in range(dim):
            for j in range(dim):
                coords[i,j,0] = i
                coords[i,j,1] = j
        coords = (coords+0.5)/dim-0.5
        coords = np.tile(np.reshape(coords,[1,point_batch_size,2]),[shape_batch_size_val,1,1])
        
        #convert coord and shape to torch cuda tensor
        coords = torch.from_numpy(coords).float().to('cuda')
        shapes = torch.from_numpy(shapes).float().cuda()
        
        G, G_max, G2, cw2, E_m, E_b  = bsp_2d.forward(shapes, coords,mode="eval")
        test_1(args, G, G_max, G2, cw2, E_m, E_b, shapes, op_dir,shape_batch_size_val)
        
    

print("Best loss,: ", loss_best, " found in epoch", epoch_best)
writer.close()


    
