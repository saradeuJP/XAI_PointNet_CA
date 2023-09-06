import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import csv
from scipy.special import softmax
import PVGeo

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


HOSTNAME = socket.gethostname()

# Read data

# ModelNet40 official train/test split
TRAIN_FILES_0 = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/train_data.txt'))


# data_0
train_file_idxs = np.arange(0, len(TRAIN_FILES_0))

x0 = []
y0 = []
z0 = []
u0 = []
v0 = []
w0 = []

for fn in range(len(TRAIN_FILES_0)): # len(TRAIN_FILES_0)
    print(train_file_idxs[fn])
    x,y,z,u,v,w = provider.loadDataFile(TRAIN_FILES_0[train_file_idxs[fn]],f'../data_voxel') # 1 x NUM_POINT x 1
    
    # coord_pc = np.concatenate([np.array(x).reshape(-1,1),
                               # np.array(y).reshape(-1,1),
                               # np.array(z).reshape(-1,1)],axis=1)
    # pts = PVGeo.points_to_poly_data(coord_pc)
    # # pts['u'] = U_star.flatten(order="F") 
    # pts.save("geo_orig1.vtk")
    # dummy
    
    idx_data = np.random.choice(x.shape[1], NUM_POINT)
    x0.append(x[:,idx_data,:])
    y0.append(y[:,idx_data,:])
    z0.append(z[:,idx_data,:])
    u0.append(u[:,idx_data,:])
    v0.append(v[:,idx_data,:])
    w0.append(w[:,idx_data,:])
    #print(x0)

        
x0 = np.array(x0).reshape(-1,NUM_POINT,1) # N x NUM_POINT x 1
y0 = np.array(y0).reshape(-1,NUM_POINT,1)
z0 = np.array(z0).reshape(-1,NUM_POINT,1)
u0 = np.array(u0).reshape(-1,NUM_POINT,1)
v0 = np.array(v0).reshape(-1,NUM_POINT,1)
w0 = np.array(w0).reshape(-1,NUM_POINT,1)

label=[]
with open('data/train_label.txt') as f:
    reader = csv.reader(f, delimiter = ',')
    for r in reader:
        #print(float(r))
        label.append(float(r[0])) 
        #current_train_label.append(float(r[1])) 

label = np.array(label)
print(label)

tmin = np.min([x0.min(),y0.min(),z0.min()])
tmax = np.max([x0.max(),y0.max(),z0.max()])
umin = np.min([u0.min(),v0.min(),w0.min()])
umax = np.max([u0.max(),v0.max(),w0.max()])

x_pc = (x0-tmin)/(tmax-tmin)
y_pc = (y0-tmin)/(tmax-tmin)
z_pc = (z0-tmin)/(tmax-tmin)
u_pc = (u0-umin)/(umax-umin)
v_pc = (v0-umin)/(umax-umin)
w_pc = (w0-umin)/(umax-umin) # N x NUM_POINT x 1

coord_pc = np.concatenate([x_pc,y_pc,z_pc],axis=2)
vel_pc = np.concatenate([u_pc,v_pc,w_pc],axis=2)

fn = 0
x,y,z,u,v,w = provider.loadDataFile(TRAIN_FILES_0[train_file_idxs[fn]],f'../data_voxel') # 1 x NUM_POINT x 1

x = (x-tmin)/(tmax-tmin)
y = (y-tmin)/(tmax-tmin)
z = (z-tmin)/(tmax-tmin)
u = (u-umin)/(umax-umin)
v = (v-umin)/(umax-umin)
w = (w-umin)/(umax-umin)
u = np.array(u)
v = np.array(v)
w = np.array(w)
vel = np.sqrt(np.square(u) + np.square(v) + np.square(w))

coord_pc_view = np.concatenate([np.array(x).reshape(-1,1),
                           np.array(y).reshape(-1,1),
                           np.array(z).reshape(-1,1)],axis=1)
pts = PVGeo.points_to_poly_data(coord_pc_view)
# pts['u'] = u.flatten(order="F") 
# pts['v'] = v.flatten(order="F") 
# pts['w'] = w.flatten(order="F") 
pts['vel'] = vel.flatten(order="F") 

with open('original_points.csv','ab') as f:               
            np.savetxt(f, vel.reshape(-1,1))

pts.save("geo_orig.vtk")



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, hx = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'hx': hx}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    
    num_batch = int(coord_pc.shape[0]/BATCH_SIZE)
    print(num_batch) # 6
    

    
    for idx in np.arange(num_batch):
        eval_data = np.stack([coord_pc[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,...],vel_pc[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,...]],-1)
        feed_dict = {ops['pointclouds_pl']: eval_data, 
                     ops['labels_pl']: label[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,...].flatten(),
                     ops['is_training_pl']: is_training}
        loss_val, pred_val,hx = sess.run([ops['loss'], ops['pred'], ops['hx']], feed_dict=feed_dict)
        
        hx = hx[0,...]  
        test = eval_data[0,...]
        points = test[...,0].reshape(NUM_POINT,3)
        vels = test[...,1].reshape(NUM_POINT,3)
        vel_view = np.sqrt(np.square(vels[:,0])+np.square(vels[:,1])+np.square(vels[:,2]))
        
        pts = PVGeo.points_to_poly_data(points)
        pts['vel'] = vel_view.flatten(order="F") 
        pts.save("geo_tot.vtk")
        with open('train_points.csv','ab') as f:               
            np.savetxt(f, vel_view.reshape(-1,1))
        
        cs_index = np.argmax(np.squeeze(hx), axis = 0) #find which point contributed to max-pooling features
        # print(cs_index)
        # print(cs_index.shape)

        cs = []
        vel_cs = []
        for index in cs_index:
            cs.append(points[index])
            vel_cs.append(vels[index])
        
        cs = np.array(cs).reshape(-1,3)
        vel_cs = np.array(vel_cs).reshape(-1,3)
        vel_cs = np.sqrt(np.square(vel_cs[:,0])+np.square(vel_cs[:,1])+np.square(vel_cs[:,2]))
        
        print(cs)
        print(cs.shape)
        
        pts = PVGeo.points_to_poly_data(cs)
        pts['vel'] = vel_cs.flatten(order="F") 
        pts.save("geo_cs.vtk")
        with open('cs_points.csv','ab') as f:               
            np.savetxt(f, vel_cs.reshape(-1,1))
        
        
        dummy
        
        with open('x.csv','ab') as f:               
                  np.savetxt(f, x0[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,:,:].reshape(-1,1))
        with open('y.csv','ab') as f:               
                  np.savetxt(f, y0[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,:,:].reshape(-1,1))     
        with open('z.csv','ab') as f:               
                  np.savetxt(f, z0[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,:,:].reshape(-1,1))           
        
        with open('u.csv','ab') as f:               
                  np.savetxt(f, u0[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,:,:].reshape(-1,1))
        with open('v.csv','ab') as f:               
                  np.savetxt(f, v0[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,:,:].reshape(-1,1))     
        with open('w.csv','ab') as f:               
                  np.savetxt(f, w0[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,:,:].reshape(-1,1))  
        
        with open('max_points.csv','ab') as f:               
                  np.savetxt(f, max_points.reshape(-1,1)) 
        
        with open('max_idx.csv','ab') as f:               
                  np.savetxt(f, max_idx.reshape(-1,1)) 
        
        dummy
        
        pred_val_1 = np.argmax(pred_val, 1)
        
        with open('test_label_internal.csv','ab') as f:               
                  np.savetxt(f, label[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,...].reshape(-1,1))
        
        with open('test_pred_logit_internal.csv','ab') as f:               
                  np.savetxt(f, (np.array(pred_val_1)).reshape(-1,1))
        
        with open('test_pred_internal.csv','ab') as f:               
                  np.savetxt(f, (np.array(pred_val[:,1:2])).reshape(-1,1))

        


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
