import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
#import setGPU
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
# import pc_util
import csv
import PVGeo
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
parser.add_argument('--num_drop', type=int, default=5, help='num of points to drop each step')
parser.add_argument('--num_steps', type=int, default=20, help='num of steps to drop each step')
parser.add_argument('--drop_neg', action='store_true',help='drop negative points')
parser.add_argument('--power', type=int, default=1, help='x: -dL/dr*r^x')
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

NUM_CLASSES = 2

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES_0 = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/train_data.txt'))


train_file_idxs = np.arange(0, len(TRAIN_FILES_0))

x0 = []
y0 = []
z0 = []
u0 = []
v0 = []
w0 = []

for fn in range(len(TRAIN_FILES_0)): # len(TRAIN_FILES_0)
    # print(train_file_idxs[fn])
    x,y,z,u,v,w = provider.loadDataFile(TRAIN_FILES_0[train_file_idxs[fn]],f'../../Recurrence_Unrecurrence/data_voxel_new') # 1 x NUM_POINT x 1
    
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

tmin = np.min([x0.min(),y0.min(),z0.min()])
tmax = np.max([x0.max(),y0.max(),z0.max()])
umin = np.min([u0.min(),v0.min(),w0.min()])
umax = np.max([u0.max(),v0.max(),w0.max()])

print(tmin)
print(tmax)

###################### READ TEST ###########################

TRAIN_FILES_0 = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/test_data.txt'))

current_label=[]
with open('data/test_label.txt') as f:
    reader = csv.reader(f, delimiter = ',')
    for r in reader:
        #print(float(r))
        current_label.append(float(r[0])) 
        #current_train_label.append(float(r[1]))

current_label = np.array(current_label)
		
train_file_idxs = np.arange(0, len(TRAIN_FILES_0))

x0 = []
y0 = []
z0 = []
u0 = []
v0 = []
w0 = []

for fn in range(len(TRAIN_FILES_0)): # len(TRAIN_FILES_0)
    # print(train_file_idxs[fn])
    x,y,z,u,v,w = provider.loadDataFile(TRAIN_FILES_0[train_file_idxs[fn]],f'../../Recurrence_Unrecurrence/data_voxel_new') # 1 x NUM_POINT x 1
    
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
		
x_pc = (x0-tmin)/(tmax-tmin)
y_pc = (y0-tmin)/(tmax-tmin)
z_pc = (z0-tmin)/(tmax-tmin)
u_pc = (u0-umin)/(umax-umin)
v_pc = (v0-umin)/(umax-umin)
w_pc = (w0-umin)/(umax-umin) # N x NUM_POINT x 1

coord_pc = np.concatenate([x_pc,y_pc,z_pc],axis=2)
vel_pc = np.concatenate([u_pc,v_pc,w_pc],axis=2)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
    
def save_vtk(pc, pcvel, pc_adv, pcvel_adv, img_filename):
    print(pc.shape)
    print(pcvel_adv.shape)
    pts = PVGeo.points_to_poly_data(pc_adv)
    pts['u'] = pcvel_adv[:,0].flatten(order="F") 
    pts['v'] = pcvel_adv[:,1].flatten(order="F") 
    pts['w'] = pcvel_adv[:,2].flatten(order="F") 
    # pts['vel'] = vel.flatten(order="F") 

    pts.save(img_filename+'adv.vtk')
    
    pts = PVGeo.points_to_poly_data(pc)
    pts['u'] = pcvel[:,0].flatten(order="F") 
    pts['v'] = pcvel[:,1].flatten(order="F") 
    pts['w'] = pcvel[:,2].flatten(order="F") 
    # pts['vel'] = vel.flatten(order="F") 

    pts.save(img_filename+'ori.vtk')

class SphereAttack():
    def __init__(self, num_drop, num_steps):
        self.a = num_drop # how many points to remove
        self.k = num_steps
        
        self.is_training = False
        # self.count = np.zeros((NUM_CLASSES, ), dtype=bool)
        # self.all_counters = np.zeros((NUM_CLASSES, 3), dtype=int)
        
        # The number of points is not specified
        self.pointclouds_pl, self.labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, None)
        self.is_training_pl = tf.placeholder(tf.bool, shape=())
        
        # simple model
        self.pred, self.hx = MODEL.get_model(self.pointclouds_pl, self.is_training_pl)
        self.classify_loss = MODEL.get_loss(self.pred, self.labels_pl)
        
        self.grad = tf.gradients(self.classify_loss, self.pointclouds_pl)[0]
        
        
        
    def drop_points(self, pointclouds_pl, vel_pl, labels_pl, idx, sess):## BxNx3
        
        pl = np.stack([pointclouds_pl, vel_pl],-1) # BxNx3x2
        grad = sess.run(self.grad, feed_dict={self.pointclouds_pl: pl,
                                              self.labels_pl: labels_pl,
                                              self.is_training_pl: self.is_training})
        grad = grad[...,0]
        # print(grad.shape)
        # grad = np.squeeze(grad,axis=-1) # BxNx3
        
        # change the grad into spherical axis and compute r*dL/dr
        ## mean value            
        #sphere_core = np.sum(pointclouds_pl_adv, axis=1, keepdims=True)/float(pointclouds_pl_adv.shape[1])
        ## median value
        
        sphere_core = np.median(pointclouds_pl, axis=1, keepdims=True)
        
        sphere_r = np.sqrt(np.sum(np.square(pointclouds_pl - sphere_core), axis=2)) ## BxN
        
        sphere_axis = pointclouds_pl - sphere_core ## BxNx3

        if FLAGS.drop_neg:
            sphere_map = np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2), np.power(sphere_r, FLAGS.power))
        else:
            sphere_map = -np.multiply(np.sum(np.multiply(grad, sphere_axis), axis=2), np.power(sphere_r, FLAGS.power))
        
        sphere_map = (sphere_map - sphere_map.min())/(sphere_map.max() - sphere_map.min())
        
        
        pointclouds_pl = pointclouds_pl * (tmax-tmin) + tmin
        
        savedata = pd.DataFrame({'x': pointclouds_pl[:,:,0].flatten(order="F"),
                                'y': pointclouds_pl[:,:,1].flatten(order="F"),
                                'z': pointclouds_pl[:,:,2].flatten(order="F"),
                                'score': sphere_map.flatten(order="F")},dtype = float)

        savedata = savedata.sort_values(by=['score'])  
        
        
        savedata.iloc[-10:,0:3].to_csv(str(idx+1)+'pc_score.csv', index=False)
   
    def plot_natural_and_advsarial_samples_all_situation(self, pointclouds_pl, pcvel_pl,
                                                         pointclouds_pl_adv, pcvel_pl_adv,
                                                         labels_pl, pred_val, pred_val_adv,batch_idx):
        
        
        for i in range(labels_pl.shape[0]):
            if labels_pl[i] == pred_val[i]:
                if labels_pl[i] != pred_val_adv[i]:
                    img_filename = 'no_%s_label_%s_pred_%s_advpred_%s' % (batch_idx, labels_pl[i],
                                                              pred_val[i],
                                                              pred_val_adv[i])
                    #self.all_counters[labels_pl[i]][0] += 1
                    img_filename = os.path.join(DUMP_DIR+'/pred_correct_adv_wrong', img_filename)
                    
                    save_vtk(pointclouds_pl[i], pcvel_pl[i], pointclouds_pl_adv[i], pcvel_pl_adv[i], img_filename)
                    
                    # pc_util.pyplot_draw_point_cloud_nat_and_adv(pointclouds_pl[i], pointclouds_pl_adv[i], img_filename) 
                else:
                    print('NO CHANGE')
                    print('no_%s_label_%s_pred_%s_advpred_%s' % (batch_idx, labels_pl[i],
                                                                 pred_val[i],
                                                                 pred_val_adv[i]))
            else:
                if labels_pl[i] == pred_val_adv[i]:
                    img_filename = 'no_%s_label_%s_pred_%s_advpred_%s.vtk' % (batch_idx, labels_pl[i],
                                                              pred_val[i],
                                                              pred_val_adv[i])
                    #self.all_counters[labels_pl[i]][1] += 1        
                    img_filename = os.path.join(DUMP_DIR+'/pred_wrong_adv_correct', img_filename)
                    save_vtk(pointclouds_pl[i], pcvel_pl[i], pointclouds_pl_adv[i], pcvel_pl_adv[i], img_filename)
                    # pc_util.pyplot_draw_point_cloud_nat_and_adv(pointclouds_pl[i], pointclouds_pl_adv[i], img_filename)
                else:
                    
                    print('NO CHANGE')
                    print('no_%s_label_%s_pred_%s_advpred_%s' % (batch_idx, labels_pl[i],
                                                                 pred_val[i],
                                                                 pred_val_adv[i]))
                    
                    # img_filename = 'no_%s_label_%s_pred_%s_advpred_%s' % (batch_idx, labels_pl[i],
                                                              # pred_val[i],
                                                              # pred_val_adv[i])
                    #self.all_counters[labels_pl[i]][2] += 1
                    # img_filename = os.path.join(DUMP_DIR+'/pred_wrong_adv_wrong', img_filename)
                
                
                

def evaluate(num_votes):
    is_training = False
    num_drop, num_steps = FLAGS.num_drop, FLAGS.num_steps
    attack = SphereAttack(num_drop, num_steps)
        
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

    ## ops built on attributes defined in attack
    ops = {'pointclouds_pl': attack.pointclouds_pl,
           'labels_pl': attack.labels_pl,
           'is_training_pl': attack.is_training_pl,
           'pred': attack.pred,
           'loss': attack.classify_loss}

    NUM_POINT = FLAGS.num_point
    NUM_POINT_ADV = NUM_POINT - num_drop*num_steps
    
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    # for fn in range(len(TRAIN_FILES_0)):
        # log_string('----'+str(fn)+'----')
        
    # print(coord_pc.shape[0])
    
    file_size = coord_pc.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(num_batches)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        
        # Aggregating BEG
        batch_loss_sum = 0 # sum of losses for the batch
        batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
        batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
        
        ## Produce adversarial samples
        
        attack.drop_points(coord_pc[start_idx:end_idx, :, :],
                                    vel_pc[start_idx:end_idx, :, :], 
                                    current_label[start_idx:end_idx], 
                                    batch_idx, sess)
        ## Natural data
        # for vote_idx in range(num_votes):
            # rotated_data = provider.rotate_point_cloud_by_angle(coord_pc[start_idx:end_idx, :, :],
                                              # vote_idx/float(num_votes) * np.pi * 2)
            # eval_data = np.stack([rotated_data,
                                  # vel_pc[start_idx:end_idx, :, :]],-1)
            # #print(eval_data.shape)
            # feed_dict = {ops['pointclouds_pl']: eval_data,
                         # ops['labels_pl']: current_label[start_idx:end_idx],
                         # ops['is_training_pl']: is_training}
            # loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                      # feed_dict=feed_dict)
            # batch_pred_sum += pred_val
            # batch_pred_val = np.argmax(pred_val, 1)
            # for el_idx in range(cur_batch_size):
                # batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            # batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
        # pred_val = np.argmax(batch_pred_sum, 1)
        
        # ## Adversarial data
        
        # batch_loss_sum_adv = 0 # sum of losses for the batch
        # batch_pred_sum_adv = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
        # batch_pred_classes_adv = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
        
        # for vote_idx in range(num_votes):
            # rotated_data = provider.rotate_point_cloud_by_angle(cur_batch_data_adv,
                                              # vote_idx/float(num_votes) * np.pi * 2)
            # eval_data_adv = np.stack([rotated_data,vel_batch_data_adv],axis=-1)
            # feed_dict = {ops['pointclouds_pl']: eval_data_adv,
                         # ops['labels_pl']: current_label[start_idx:end_idx],
                         # ops['is_training_pl']: is_training}
            # loss_val_adv, pred_val_adv = sess.run([ops['loss'], ops['pred']],
                                      # feed_dict=feed_dict)
            # batch_pred_sum_adv += pred_val_adv
            # batch_pred_val_adv = np.argmax(pred_val_adv, 1)
            # for el_idx in range(cur_batch_size):
                # batch_pred_classes_adv[el_idx, batch_pred_val_adv[el_idx]] += 1
            # batch_loss_sum_adv += (loss_val_adv * cur_batch_size / float(num_votes))
        # pred_val_adv = np.argmax(batch_pred_sum_adv, 1)

        # attack.plot_natural_and_advsarial_samples_all_situation(coord_pc[start_idx:end_idx, :, :], vel_pc[start_idx:end_idx, :, :], 
                                                                # cur_batch_data_adv, vel_batch_data_adv,
                                                                # current_label[start_idx:end_idx], pred_val, pred_val_adv,batch_idx)
        # correct = np.sum(pred_val_adv == current_label[start_idx:end_idx])
        # # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
        # total_correct += correct
        # total_seen += cur_batch_size
        # loss_sum += batch_loss_sum_adv

        # for i in range(start_idx, end_idx):
            # l = int(current_label[i])
            # total_seen_class[l] += 1
            # total_correct_class[l] += (pred_val_adv[i-start_idx] == l)
            # fout.write('%d, %d\n' % (pred_val_adv[i-start_idx], l))
            
            # # if pred_val[i-start_idx] != l and FLAGS.visu: # ERROR CASE, DUMP!
                # # img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
                                                       # # SHAPE_NAMES[pred_val[i-start_idx]])
                # # img_filename = os.path.join(DUMP_DIR, img_filename)
                # # output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, :, :]))
                # # scipy.misc.imsave(img_filename, output_img)
                    # # error_cnt += 1
                
    # log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    # log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    # log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    
    # class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    # for i, name in enumerate(SHAPE_NAMES):
        # log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
