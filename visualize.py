import torch
import numpy
import cv2
import os

from bspt_2d import *


def test_1(args, sG, sG_max, sG2, cw2, out_m, out_b, batch_voxels,op_dir, shape_batch_size):
    if args.phase==0:
        outG = sG
    elif args.phase==1 or args.phase==2:
        outG = sG_max
    t = 0
    
    outG = outG.cpu().detach().numpy()
    sG = sG.cpu().detach().numpy()
    sG_max = sG_max.cpu().detach().numpy()
    sG2 = sG2.cpu().detach().numpy()
    cw2 = cw2.cpu().detach().numpy()
    out_m = out_m.cpu().detach().numpy()
    out_b = out_b.cpu().detach().numpy()
    batch_voxels = batch_voxels.cpu().detach().numpy().transpose(0,2,3,1)

    imgs = np.clip(np.resize(outG,[shape_batch_size, args.sample_vox_size, args.sample_vox_size])*256, 0, 255).astype(np.uint8)

    for t in range(shape_batch_size):
        cv2.imwrite(os.path.join(op_dir,str(t)+"_out.png"), imgs[t])
        cv2.imwrite(os.path.join(op_dir,str(t)+"_gt.png"), batch_voxels[t]*255)

    if args.phase==1 or args.phase==2:
        image_out_size = 256
        
        model_out = np.resize(sG2,[args.shape_batch_size,args.sample_vox_size,args.sample_vox_size,args.gf_dim])

        for t in range(shape_batch_size):
            bsp_convex_list = []
            color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
            color_idx_list = []

            for i in range(args.gf_dim):
                min_v = np.min(model_out[t,:,:,i])
                if min_v<0.01:
                    box = []
                    for j in range(args.p_dim):
                        if cw2[j,i]>0.01:
                            a = -out_m[t,0,j]
                            b = -out_m[t,1,j]
                            d = -out_b[t,0,j]
                            box.append([a,b,d])
                    if len(box)>0:
                        bsp_convex_list.append(np.array(box,np.float32))
                        color_idx_list.append(i%len(color_list))



            #convert bspt to mesh
            vertices = []
            polygons = []
            polygons_color = []

            img_out = np.full([image_out_size,image_out_size,3],255,np.uint8)
            for i in range(len(bsp_convex_list)):
                vg, tg = digest_bsp(bsp_convex_list[i], bias=0)
                cg = color_list[color_idx_list[i]]
                for j in range(len(tg)):
                    x1 = ((vg[tg[j][0]][1]+0.5)*image_out_size).astype(np.int32)
                    y1 = ((vg[tg[j][0]][0]+0.5)*image_out_size).astype(np.int32)
                    x2 = ((vg[tg[j][1]][1]+0.5)*image_out_size).astype(np.int32)
                    y2 = ((vg[tg[j][1]][0]+0.5)*image_out_size).astype(np.int32)
                    cv2.line(img_out, (x1,y1), (x2,y2), cg, thickness=1)

            cv2.imwrite(os.path.join(op_dir,str(t)+"_bsp.png"), img_out)

    print("[sample]")

# #output bsp shape with color
#     def test_bsp(self, config):
#     could_load, checkpoint_counter = self.load(self.checkpoint_dir)
#     if could_load:
#     print(" [*] Load SUCCESS")
#     else:
#     print(" [!] Load failed...")
#     return

#     image_out_size = 256
#     w2 = self.sess.run(self.cw2, feed_dict={})

#     start_n = config.start
#     batch_voxels = self.data_voxels[start_n:start_n+self.shape_batch_size]
#     model_out, out_m, out_b = self.sess.run([self.sG2, self.sE_m, self.sE_b],
#     feed_dict={
#         self.vox3d: batch_voxels,
#     })
#     model_out = np.resize(model_out,[self.shape_batch_size,self.sample_vox_size,self.sample_vox_size,self.gf_dim])

#     for t in range(self.shape_batch_size):
#     bsp_convex_list = []
#     color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
#     color_idx_list = []

#     for i in range(self.gf_dim):
#         min_v = np.min(model_out[t,:,:,i])
#         if min_v<0.01:
#             box = []
#             for j in range(self.p_dim):
#                 if w2[j,i]>0.01:
#                     a = -out_m[t,0,j]
#                     b = -out_m[t,1,j]
#                     d = -out_b[t,0,j]
#                     box.append([a,b,d])
#             if len(box)>0:
#                 bsp_convex_list.append(np.array(box,np.float32))
#                 color_idx_list.append(i%len(color_list))

#     #print(bsp_convex_list)
#     print(len(bsp_convex_list))

#     #convert bspt to mesh
#     vertices = []
#     polygons = []
#     polygons_color = []

#     img_out = np.full([image_out_size,image_out_size,3],255,np.uint8)
#     for i in range(len(bsp_convex_list)):
#         vg, tg = digest_bsp(bsp_convex_list[i], bias=0)
#         cg = color_list[color_idx_list[i]]
#         for j in range(len(tg)):
#             x1 = ((vg[tg[j][0]][1]+0.5)*image_out_size).astype(np.int32)
#             y1 = ((vg[tg[j][0]][0]+0.5)*image_out_size).astype(np.int32)
#             x2 = ((vg[tg[j][1]][1]+0.5)*image_out_size).astype(np.int32)
#             y2 = ((vg[tg[j][1]][0]+0.5)*image_out_size).astype(np.int32)
#             cv2.line(img_out, (x1,y1), (x2,y2), cg, thickness=1)

#     cv2.imwrite(os.path.join(op_dir,+str(t)+"_bsp.png", img_out))


# #output h3
# def test_dae3(self, config):
#     could_load, checkpoint_counter = self.load(self.checkpoint_dir)
#     if could_load:
#         print(" [*] Load SUCCESS")
#     else:
#         print(" [!] Load failed...")
#         return
#     t = 0
#     batch_voxels = self.data_voxels[t:t+self.shape_batch_size]
#     model_out = self.sess.run(self.sG,
#         feed_dict={
#             self.vox3d: batch_voxels,
#         })
#     imgs = np.clip(np.resize(model_out,[self.shape_batch_size,self.sample_vox_size,self.sample_vox_size])*256, 0, 255).astype(np.uint8)
#     for t in range(self.shape_batch_size):
#         cv2.imwrite(config.sample_dir+"/"+str(t)+"_out.png", imgs[t])
#         cv2.imwrite(config.sample_dir+"/"+str(t)+"_gt.png", batch_voxels[t]*255)
#     print("[sample]")