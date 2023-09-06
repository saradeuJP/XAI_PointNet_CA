import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import csv
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import PVGeo
from PVGeo.filters import VoxelizePoints
import copy
#from sklearn import cross_validation

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=2000, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

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
    x,y,z,u,v,w = provider.loadDataFile(TRAIN_FILES_0[train_file_idxs[fn]],f'../../Recurrence_Unrecurrence/data_voxel_new') # 1 x NUM_POINT x 1
    #print(x.shape)
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

mintotc = np.min([x0.min(), y0.min(), z0.min()])
maxtotc = np.max([x0.max(), y0.max(), z0.max()])

mintotv = np.min([u0.min(), v0.min(), w0.min()])
maxtotv = np.max([u0.max(), v0.max(), w0.max()])

x_pc = (x0-mintotc)/(maxtotc-mintotc)
y_pc = (y0-mintotc)/(maxtotc-mintotc)
z_pc = (z0-mintotc)/(maxtotc-mintotc)
u_pc = (u0-mintotv)/(maxtotv-mintotv)
v_pc = (v0-mintotv)/(maxtotv-mintotv)
w_pc = (w0-mintotv)/(maxtotv-mintotv) # N x NUM_POINT x 1

coord_pc = np.concatenate([x_pc,y_pc,z_pc],axis=2)
vel_pc = np.concatenate([u_pc,v_pc,w_pc],axis=2)

# x_train, x_test, y_train, y_test = cross_validation.train_test_split( coord_pc, label, test_size=0.4, random_state=0) 
# print(x_train.shape)
# dummy

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    global coord_pc_0, vel_pc_0, coord_pc_1, vel_pc_1
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred,hx = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'hx': hx,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        
        train_loss_set = []
        eval_loss_set = []
        
        best_eval = 0
        best_train = 0
        best_epoch = 0
        best_train_loss = 1000000
        best_train_std = 1000000
        best_eval_loss = 1000000
        
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            current_coord_pc, current_vel_pc, current_label, _ = provider.shuffle_data(coord_pc, vel_pc, label)
            kfold =10
            #print(current_coord_pc.shape[0])
            knr = current_coord_pc.shape[0] / kfold # 8
            #print(int(knr*(kfold-1)))
            
            train_coord = current_coord_pc[0:int(knr*(kfold-1)),...]
            train_vel = current_vel_pc[0:int(knr*(kfold-1)),...]
            train_label = current_label[0:int(knr*(kfold-1)),...] # 4*9=36
            
            val_coord = current_coord_pc[int(knr*(kfold-1)):,...] # 
            val_vel = current_vel_pc[int(knr*(kfold-1)):,...]
            val_label = current_label[int(knr*(kfold-1)):,...]
            

            train_loss, train_accuracy, train_std = train_one_epoch(train_coord, train_vel, train_label, sess, ops, train_writer)
            eval_loss, eval_accuracy = eval_one_epoch(val_coord, val_vel, val_label, sess, ops, test_writer)
            
            #train_loss_set.append(train_loss)
            #eval_loss_set.append(eval_loss)
            
            # print(train_loss_set[-3:])
                
            if (eval_loss <= best_eval_loss):
               # &(train_loss <= best_train_loss):
               #(eval_accuracy >= best_eval)  (train_accuracy >= best_train) & \
               # &   # & \
               #(train_std <= best_train_std):
               #  & \(train_loss <= np.mean(train_loss_set[-3:]))
               
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Best model saved in file: %s" % save_path)
                
                best_eval = eval_accuracy
                best_train = train_accuracy
                best_epoch = epoch
                best_train_loss = train_loss
                best_eval_loss = eval_loss
                best_train_std = train_std
    
    log_string('Best epoch: %f' % best_epoch)

def train_one_epoch(coord_pc_train, vel_pc_train, label_train, sess, ops, train_writer): # 72 x NUM_POINT x 3
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    current_coord_pc, current_vel_pc, current_label, _ = provider.shuffle_data(coord_pc_train, vel_pc_train, label_train) # 72, 1024, 3
    num_batch = int(coord_pc_train.shape[0]/BATCH_SIZE)
    #print(num_batch) # 9
    
    loss_set = []
    accuracy_set = 0
    
    for idx in np.arange(num_batch):
        rotated_data = provider.rotate_point_cloud(current_coord_pc[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,...]) 
        jittered_data = provider.jitter_point_cloud(rotated_data)
        
        #random_vel = provider.random_vel(current_vel_pc[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,...])
        random_vel = provider.random_vel(current_vel_pc[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,...])
        
        jittered_data = np.stack([jittered_data,random_vel],-1) # B, 1024, 3, 2

        feed_dict = {ops['pointclouds_pl']: jittered_data,
                     ops['labels_pl']: current_label[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,...].flatten(),
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        
        
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].flatten())

        loss_set.append(loss_val)
        accuracy_set+=correct
    accuracy = accuracy_set/coord_pc_train.shape[0]
    loss_set = np.array(loss_set)
    log_string('mean loss: %f' % loss_set.mean())
    log_string('accuracy: %f' % accuracy)
    
    with open('train_loss_mean.csv','ab') as f:               
                  np.savetxt(f,loss_set.mean().reshape(-1,1))
    
    with open('train_loss_std.csv','ab') as f:               
                  np.savetxt(f,loss_set.std().reshape(-1,1))
    
    with open('train_accuracy.csv','ab') as f:               
                  np.savetxt(f,accuracy.reshape(-1,1))
    
    return loss_set.mean(), accuracy.mean(),loss_set.std()   
        
def eval_one_epoch(coord_pc_eval, vel_pc_eval, label_eval, sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    
    current_coord_pc, current_vel_pc, current_label, _ = provider.shuffle_data(coord_pc_eval, vel_pc_eval, label_eval) # 1 x 8 x N x 3 
    jittered_data = np.stack([current_coord_pc, current_vel_pc],-1) # B, 1024, 3, 2
        
    feed_dict = {ops['pointclouds_pl']: jittered_data,
                 ops['labels_pl']: current_label.flatten(),
                 ops['is_training_pl']: is_training}
    summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
        ops['loss'], ops['pred']], feed_dict=feed_dict)
        
    pred_val = np.argmax(pred_val, 1)
    correct = np.sum(pred_val == current_label.flatten())
    accuracy = correct / coord_pc_eval.shape[0]
            
    log_string('eval mean loss: %f' % loss_val)
    log_string('eval accuracy: %f'% accuracy)

    with open('eval_loss.csv','ab') as f:               
                  np.savetxt(f,loss_val.reshape(-1,1))
    
    with open('eval_accuracy.csv','ab') as f:               
                  np.savetxt(f,accuracy.reshape(-1,1))    
    return loss_val, accuracy

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
