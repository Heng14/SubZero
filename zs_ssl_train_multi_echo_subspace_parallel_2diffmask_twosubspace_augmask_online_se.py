import tensorflow as tf
# import tensorflow.compat.v1 as tf
import scipy.io as sio
import numpy as np
# import cupy as cp
import time
from datetime import datetime
import os
import h5py as h5
import utils
import tf_utils_subspace_p2_se as tf_utils
import tf_utils_subspace_se as tf_utils_subspace
import parser_ops
import UnrollNet_subspace_2diffmask_p2_se as UnrollNet
import UnrollNet_subspace_2diffmask_p1_se as UnrollNet_subspace
# from tensorflow.python.profiler import model_analyzer
# from tensorflow.python.profiler import option_builder



# tf.disable_v2_behavior()

parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "7" #set it to available GPU


# from numba import cuda
# cuda.select_device(0)

# @cuda.jit
# def mat_multiply(x1, x2, y):
#     i, j= cuda.grid(2)
#     y[i][j] = x1[i][j] * x2[i][j]


if args.transfer_learning:
    print('Getting weights from trained model:')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    loadChkPoint_tl = tf.train.latest_checkpoint(args.TL_path)
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(args.TL_path + '/modelTst.meta')
        new_saver.restore(sess, loadChkPoint_tl)
        trainable_variable_collections = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        pretrained_model_weights = [sess.run(v) for v in trainable_variable_collections]


save_dir ='saved_models'
directory = os.path.join(save_dir, args.data_name + '_' + args.data_opt + '_Rate'+ str(args.acc_rate)+'_'+ str(args.num_reps)+'reps')
if not os.path.exists(directory):
    os.makedirs(directory)

#................................................................................
start_time = time.time()
print('.................ZS-SSL Training.....................')
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data..........................................
print('Loading data  for training............... ')

# data = sio.loadmat(f'kdata_multiecho_SE_zsssl.mat')
data = sio.loadmat(args.data_dir)
kspace_train,sens_maps0, original_mask= data['kspace'], data['sens_maps'], data['mask']

TE_use = data['TE_use'][0]
possibleT2Values_use = np.arange(0, 1000.0001, 0.1)

dict0 = np.zeros((len(possibleT2Values_use), len(TE_use)))
print (dict0.shape)

for t in range(len(possibleT2Values_use)):
    dict0[t] = np.exp(-TE_use / possibleT2Values_use[t])

U, S, Vh = np.linalg.svd(dict0.T)
Uk = U[:,:args.n_basis]

print('create a test model for the testing')
test_graph_generator = tf_utils.test_graph(directory, Uk)
test_graph_generator_subspace = tf_utils_subspace.test_graph(directory, Uk)


# args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB  = kspace_train.shape

print('Normalize the kspace to 0-1 region')
kspace_train= kspace_train / np.max(np.abs(kspace_train[:]))

# kspace_list = []
# for ki in range(kspace_train.shape[-2]):
#     print(f'Normalize the kspace to 0-1 region of {ki}')
#     kspace_train_i = kspace_train[:,:,ki,:]
#     kspace_train_i =  kspace_train_i/ np.max(np.abs(kspace_train_i[:]))
#     kspace_list.append(kspace_train_i)
# kspace_train = np.stack(kspace_list, axis=-2)

args.nrow_GLOB, args.ncol_GLOB, args.n_echo ,args.ncoil_GLOB  = kspace_train.shape

#..................Generate validation mask.....................................
cv_trn_mask, cv_val_mask = utils.uniform_selection(kspace_train, original_mask, rho=args.rho_val)
remainder_mask, cv_val_mask=np.copy(cv_trn_mask),np.copy(np.complex64(cv_val_mask))

print('size of kspace: ', kspace_train[np.newaxis,...].shape, ', maps: ', sens_maps0.shape, ', mask: ', original_mask.shape)


trn_mask, loss_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo), dtype=np.complex64), \
                                np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo), dtype=np.complex64)
trn_mask_subspace, loss_mask_subspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo), dtype=np.complex64), \
                                np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo), dtype=np.complex64)

# train data
nw_input = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo), dtype=np.complex64)
nw_input_subspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo), dtype=np.complex64)

# ref_kspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
ref_kspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo, args.ncoil_GLOB), dtype=np.complex64)
ref_kspace_subspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo, args.ncoil_GLOB), dtype=np.complex64)
#...............................................................................
# validation data
ref_kspace_val = np.empty((args.num_reps,args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
nw_input_val = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

print('create training&loss masks and generate network inputs... ')
#train data
# for jj in range(args.num_reps):
#     trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(kspace_train,remainder_mask, rho=args.rho_train)
#     trn_mask_subspace[jj, ...], loss_mask_subspace[jj, ...] = utils.uniform_selection(kspace_train,remainder_mask, rho=args.rho_train)

#     # sub_kspace = kspace_train * np.tile(trn_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
#     sub_kspace = kspace_train * np.tile(trn_mask[jj][..., np.newaxis], (1, 1, 1, args.ncoil_GLOB))
#     sub_kspace_subspace = kspace_train * np.tile(trn_mask_subspace[jj][..., np.newaxis], (1, 1, 1, args.ncoil_GLOB))

#     # ref_kspace[jj, ...] = kspace_train * np.tile(loss_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
#     ref_kspace[jj, ...] = kspace_train * np.tile(loss_mask[jj][..., np.newaxis], (1, 1, 1, args.ncoil_GLOB))
#     ref_kspace_subspace[jj, ...] = kspace_train * np.tile(loss_mask_subspace[jj][..., np.newaxis], (1, 1, 1, args.ncoil_GLOB))

#     nw_input[jj, ...] = utils.sense1_multiecho(sub_kspace, sens_maps0)
#     nw_input_subspace[jj, ...] = utils.sense1_multiecho(sub_kspace_subspace, sens_maps0)

#..............................validation data.....................................
nw_input_val = utils.sense1_multiecho(kspace_train * np.tile(cv_trn_mask[:, :, :, np.newaxis], (1, 1, 1, args.ncoil_GLOB)),sens_maps0)[np.newaxis]
ref_kspace_val=kspace_train*np.tile(cv_val_mask[:, :, :, np.newaxis], (1, 1, 1, args.ncoil_GLOB))[np.newaxis]


# %%  zeropadded outer edges of k-space with no signal- check readme file for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency
# if args.data_opt=='Coronal_PD' :
#     trn_mask[:, :, 0:17] = np.ones((args.num_reps, args.nrow_GLOB, 17))
#     trn_mask[:, :, 352:args.ncol_GLOB] = np.ones((args.num_reps, args.nrow_GLOB, 16))

# %% Prepare the data for the training
sens_maps = np.tile(sens_maps0[np.newaxis],(args.num_reps,1,1,1))
sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))

# ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 4, 1, 2, 3)))
# ref_kspace_subspace = utils.complex2real(np.transpose(ref_kspace_subspace, (0, 4, 1, 2, 3)))

Uk_inv = np.linalg.pinv(Uk) # by heng
# nw_input_subspace = np.dot(nw_input_subspace, Uk_inv.T)
# nw_input_subspace = utils.complex2real(nw_input_subspace) 

# # nw_input = utils.complex2real(nw_input) # by heng
# nw_input = np.dot(nw_input, Uk_inv.T)
# nw_input = utils.complex2real(nw_input) 

# %% validation data 
ref_kspace_val = utils.complex2real(np.transpose(ref_kspace_val, (0, 4, 1, 2, 3)))

nw_input_val_subspace = np.dot(nw_input_val, Uk_inv.T) # by heng
nw_input_val_subspace = utils.complex2real(nw_input_val_subspace) 
nw_input_val = utils.complex2real(nw_input_val) # by heng


print('size of ref kspace: ', ref_kspace.shape, ', nw_input: ', nw_input.shape, ', maps: ', sens_maps.shape, ', mask: ', trn_mask.shape)

# %% set the batch size
total_batch = int(np.floor(np.float32(nw_input.shape[0]) / (args.batchSize)))
kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, args.n_echo, 2), name='refkspace')
kspaceP_subspace = tf.placeholder(tf.float32, shape=(None, None, None, None, args.n_echo, 2), name='refkspace_subspace')
sens_mapsP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='sens_maps')
trn_maskP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='trn_mask')
trn_maskP_subspace = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='trn_mask_subspace')
loss_maskP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='loss_mask')
loss_maskP_subspace = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='loss_mask_subspace')
# nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, args.n_echo, 2), name='nw_input') # by heng
nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, args.n_basis, 2), name='nw_input') # by heng
nw_inputP_subspace = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, args.n_basis, 2), name='nw_input_subspace')
# %% creating the dataset
# train_dataset = tf.data.Dataset.from_tensor_slices((kspaceP,nw_inputP,sens_mapsP,trn_maskP,loss_maskP)).shuffle(buffer_size= 10*args.batchSize).batch(args.batchSize) # by heng
train_dataset = tf.data.Dataset.from_tensor_slices((kspaceP,kspaceP_subspace,nw_inputP,nw_inputP_subspace,sens_mapsP,trn_maskP,trn_maskP_subspace,loss_maskP,loss_maskP_subspace)).shuffle(buffer_size= 10*args.batchSize).batch(args.batchSize) 
# cv_dataset = tf.data.Dataset.from_tensor_slices((kspaceP,nw_inputP,sens_mapsP,trn_maskP,loss_maskP)).shuffle(buffer_size=10*args.batchSize).batch(args.batchSize) # by heng
cv_dataset = tf.data.Dataset.from_tensor_slices((kspaceP,kspaceP_subspace,nw_inputP,nw_inputP_subspace,sens_mapsP,trn_maskP,trn_maskP_subspace,loss_maskP,loss_maskP_subspace)).shuffle(buffer_size=10*args.batchSize).batch(args.batchSize)
iterator=tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_iterator=iterator.make_initializer(train_dataset)
cv_iterator = iterator.make_initializer(cv_dataset)

ref_kspace_tensor,ref_kspace_tensor_subspace,nw_input_tensor,nw_input_tensor_subspace,sens_maps_tensor,trn_mask_tensor,trn_mask_tensor_subspace,loss_mask_tensor,loss_mask_tensor_subspace = iterator.get_next('getNext')
#%% make training model

Uk_inv = np.linalg.pinv(Uk)
UkT_tensor = tf.convert_to_tensor(Uk.T, dtype=tf.complex64)
Uk_invT_tensor = tf.convert_to_tensor(Uk_inv.T, dtype=tf.complex64)


# nw_output_img, nw_output_kspace, nw_output_kspace_inv, *_ = UnrollNet.UnrolledNet(nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor).model
nw_output_img, nw_output_kspace, nw_output_kspace_inv, *_ = UnrollNet.UnrolledNet(nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor, UkT_tensor, Uk_invT_tensor).model
nw_output_img_subspace, nw_output_kspace_subspace, nw_output_kspace_subspace_inv, *_ = UnrollNet_subspace.UnrolledNet(nw_input_tensor_subspace, sens_maps_tensor, trn_mask_tensor_subspace, loss_mask_tensor_subspace, UkT_tensor, Uk_invT_tensor).model

scalar = tf.constant(0.5, dtype=tf.float32)

# loss = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + \
#        tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1)) + \
#         tf.multiply(scalar, tf.norm(ref_kspace_tensor_subspace - nw_output_kspace_subspace) / tf.norm(ref_kspace_tensor_subspace)) + \
#         tf.multiply(scalar, tf.norm(ref_kspace_tensor_subspace - nw_output_kspace_subspace, ord=1) / tf.norm(ref_kspace_tensor_subspace, ord=1)) + \
#         0.05 * tf.multiply(scalar, tf.norm(nw_output_kspace_inv - nw_output_kspace_subspace_inv) / tf.norm(nw_output_kspace_inv)) + \
#         0.05 * tf.multiply(scalar, tf.norm(nw_output_kspace_inv - nw_output_kspace_subspace_inv, ord=1) / tf.norm(nw_output_kspace_inv, ord=1))

loss0 = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + \
       tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1)) + \
        tf.multiply(scalar, tf.norm(ref_kspace_tensor_subspace - nw_output_kspace_subspace) / tf.norm(ref_kspace_tensor_subspace)) + \
        tf.multiply(scalar, tf.norm(ref_kspace_tensor_subspace - nw_output_kspace_subspace, ord=1) / tf.norm(ref_kspace_tensor_subspace, ord=1)) 

loss = loss0 + \
        0.05 * tf.multiply(scalar, tf.norm(nw_output_kspace_inv - nw_output_kspace_subspace_inv) / tf.norm(nw_output_kspace_inv)) + \
        0.05 * tf.multiply(scalar, tf.norm(nw_output_kspace_inv - nw_output_kspace_subspace_inv, ord=1) / tf.norm(nw_output_kspace_inv, ord=1))

all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep=1) #only keep the model corresponding to lowest validation error
sess_trn_filename = os.path.join(directory, 'model')
totalLoss,totalTime=[],[]
total_val_loss = []
avg_cost = 0
print('training......................................................')
lowest_val_loss = np.inf

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print('Number of trainable parameters: ', sess.run(all_trainable_vars))

    # profiler = model_analyzer.Profiler(graph=sess.graph)
    # run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    # feedDict = {kspaceP: ref_kspace, kspaceP_subspace: ref_kspace_subspace, nw_inputP: nw_input, nw_inputP_subspace: nw_input_subspace,trn_maskP: trn_mask, trn_maskP_subspace: trn_mask_subspace, loss_maskP: loss_mask, loss_maskP_subspace: loss_mask_subspace, sens_mapsP: sens_maps}

    print('Training...')
    # if for args.stop_training consecutive epochs validation loss doesnt go below the lowest val loss,\
    #  stop the training
    if args.transfer_learning:
        print('transferring weights from pretrained model to the new model:')
        trainable_collection_test = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        initialize_model_weights = [v for v in trainable_collection_test]
        for ii in range(len(initialize_model_weights)):
            sess.run(initialize_model_weights[ii].assign(pretrained_model_weights[ii]))
    ep, val_loss_tracker = 0, 0 
    while ep<args.epochs and val_loss_tracker<args.stop_training:

        nw_input = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo), dtype=np.complex64)
        nw_input_subspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo), dtype=np.complex64)

        ref_kspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo, args.ncoil_GLOB), dtype=np.complex64)
        ref_kspace_subspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.n_echo, args.ncoil_GLOB), dtype=np.complex64)

        start_t = time.time()
        # mempool = cp.get_default_memory_pool()
        # pinned_mempool = cp.get_default_pinned_memory_pool()
        for jj in range(args.num_reps):
            
            trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(kspace_train,remainder_mask, rho=args.rho_train)
            trn_mask_subspace[jj, ...], loss_mask_subspace[jj, ...] = utils.uniform_selection(kspace_train,remainder_mask, rho=args.rho_train)

            # start_t = time.time()
            sub_kspace = kspace_train * np.tile(trn_mask[jj][..., None], (1, 1, 1, args.ncoil_GLOB))
            # sub_kspace = cp.multiply(cp.asarray(kspace_train), cp.asarray(np.tile(trn_mask[jj][..., None], (1, 1, 1, args.ncoil_GLOB))))
            # sub_kspace = cp.asnumpy(sub_kspace)
            # print ('gen mask iter cost1: ', time.time() - start_t)

            # kspace_train_device = cuda.to_device(kspace_train)
            # tmp_device = cuda.to_device(np.tile(trn_mask[jj][..., None], (1, 1, 1, args.ncoil_GLOB)))
            # sub_kspace = np.zeros_like(kspace_train)
            # sub_kspace_device = cuda.to_device(sub_kspace)
            # for ii in range(kspace_train.shape[-1]):
            #     for kk in range(kspace_train.shape[-2]):

            #         kspace_train_device = cuda.to_device(kspace_train[...,kk,ii])
            #         tmp_device = cuda.to_device(np.tile(trn_mask[jj][..., None], (1, 1, 1, args.ncoil_GLOB))[...,kk,ii])
            #         sub_kspace = np.zeros_like(kspace_train)
            #         sub_kspace_device = cuda.to_device(sub_kspace[...,kk,ii])

            #         mat_multiply[kspace_train.shape[:-2], (1,1)](kspace_train_device, tmp_device, sub_kspace_device)
            # print ('gen mask iter cost2: ', time.time() - start_t)
            # raise

            sub_kspace_subspace = kspace_train * np.tile(trn_mask_subspace[jj][..., None], (1, 1, 1, args.ncoil_GLOB))
            # sub_kspace_subspace = cp.multiply(cp.asarray(kspace_train), cp.asarray(np.tile(trn_mask_subspace[jj][..., None], (1, 1, 1, args.ncoil_GLOB))))
            # sub_kspace_subspace = cp.asnumpy(sub_kspace_subspace)

            ref_kspace[jj, ...] = kspace_train * np.tile(loss_mask[jj][..., None], (1, 1, 1, args.ncoil_GLOB))
            # tmp = cp.multiply(cp.asarray(kspace_train), cp.asarray(np.tile(loss_mask[jj][..., None], (1, 1, 1, args.ncoil_GLOB))))
            # ref_kspace[jj, ...] = cp.asnumpy(tmp)

            ref_kspace_subspace[jj, ...] = kspace_train * np.tile(loss_mask_subspace[jj][..., None], (1, 1, 1, args.ncoil_GLOB))
            # tmp = cp.multiply(cp.asarray(kspace_train), cp.asarray(np.tile(loss_mask_subspace[jj][..., None], (1, 1, 1, args.ncoil_GLOB))))
            # ref_kspace_subspace[jj, ...] = cp.asnumpy(tmp)

            nw_input[jj, ...] = utils.sense1_multiecho(sub_kspace, sens_maps0)
            nw_input_subspace[jj, ...] = utils.sense1_multiecho(sub_kspace_subspace, sens_maps0)

        # mempool.free_all_blocks()
        # pinned_mempool.free_all_blocks()
        print ('gen mask iter cost: ', time.time() - start_t)
        # raise

        # start_t = time.time()
        # for jj in range(args.num_reps):
            
        #     trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(kspace_train,remainder_mask, rho=args.rho_train)
        #     trn_mask_subspace[jj, ...], loss_mask_subspace[jj, ...] = utils.uniform_selection(kspace_train,remainder_mask, rho=args.rho_train)
 

        # sub_kspace = kspace_train[None, ...] * np.tile(trn_mask[..., None], (1, 1, 1, 1, args.ncoil_GLOB))
        # sub_kspace_subspace = kspace_train[None, ...] * np.tile(trn_mask_subspace[..., None], (1, 1, 1, 1, args.ncoil_GLOB))

        # print ('gen mask iter cost2: ', time.time() - start_t)     

        # # ref_kspace = kspace_train[None, ...] * np.tile(loss_mask[..., None], (1, 1, 1, 1, args.ncoil_GLOB))
        # ref_kspace = cp.multiply(cp.asarray(kspace_train[None, ...]), cp.asarray(np.tile(loss_mask[..., None], (1, 1, 1, 1, args.ncoil_GLOB))))
        # ref_kspace = cp.asnumpy(ref_kspace)
        # # tmp1 = tf.convert_to_tensor(kspace_train[None, ...], dtype=tf.float32)
        # # tmp2 = tf.convert_to_tensor(np.tile(loss_mask[..., None], (1, 1, 1, 1, args.ncoil_GLOB)), dtype=tf.float32)
        # # tmp = tmp1 * tmp2

        # ref_kspace_subspace = kspace_train[None, ...] * np.tile(loss_mask_subspace[..., None], (1, 1, 1, 1, args.ncoil_GLOB))

        # print ('gen mask iter cost3: ', time.time() - start_t)     

        # for jj in range(args.num_reps):
        #     nw_input[jj, ...] = utils.sense1_multiecho(sub_kspace[jj, ...], sens_maps0)
        #     nw_input_subspace[jj, ...] = utils.sense1_multiecho(sub_kspace_subspace[jj, ...], sens_maps0)

        # raise


        ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 4, 1, 2, 3)))
        ref_kspace_subspace = utils.complex2real(np.transpose(ref_kspace_subspace, (0, 4, 1, 2, 3)))

        nw_input_subspace = np.dot(nw_input_subspace, Uk_inv.T)
        nw_input_subspace = utils.complex2real(nw_input_subspace) 

        nw_input = np.dot(nw_input, Uk_inv.T)
        nw_input = utils.complex2real(nw_input) 

        feedDict = {kspaceP: ref_kspace, kspaceP_subspace: ref_kspace_subspace, nw_inputP: nw_input, nw_inputP_subspace: nw_input_subspace,trn_maskP: trn_mask, trn_maskP_subspace: trn_mask_subspace, loss_maskP: loss_mask, loss_maskP_subspace: loss_mask_subspace, sens_mapsP: sens_maps} # by heng

        sess.run(train_iterator, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        try:
            for jj in range(total_batch):
                # start_t = time.time()
                # tmp, _, _ = sess.run([loss, update_ops, optimizer], options=run_options, run_metadata=run_metadata)
                tmp, _, _ = sess.run([loss, update_ops, optimizer])
                # profiler.add_step(step=jj, run_meta=run_metadata)
                # print ('each iter cost: ', time.time() - start_t)
                avg_cost += tmp / total_batch  

            toc = time.time() - tic
            totalLoss.append(avg_cost)
        except tf.errors.OutOfRangeError:
            pass
        #%%..................................................................
        # perform validation
        sess.run(cv_iterator, feed_dict={kspaceP: ref_kspace_val, kspaceP_subspace: ref_kspace_val, nw_inputP: nw_input_val_subspace, nw_inputP_subspace: nw_input_val_subspace, trn_maskP: cv_trn_mask[np.newaxis], trn_maskP_subspace: cv_trn_mask[np.newaxis], loss_maskP: cv_val_mask[np.newaxis], loss_maskP_subspace: cv_val_mask[np.newaxis], sens_mapsP: sens_maps[0][np.newaxis]})
        val_loss = sess.run([loss])[0]
        # val_loss = sess.run([loss0])[0]
        total_val_loss.append(val_loss)
        # ..........................................................................................................
        print("Epoch:", ep, "elapsed_time =""{:f}".format(toc), "trn loss =", "{:.5f}".format(avg_cost)," val loss =", "{:.5f}".format(val_loss))        
        if val_loss<=lowest_val_loss:
            lowest_val_loss = val_loss    
            saver.save(sess, sess_trn_filename, global_step=ep)
            val_loss_tracker = 0 #reset the val loss tracker each time a new lowest val error is achieved
        else:
            val_loss_tracker += 1
        sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'trn_loss': totalLoss, 'val_loss': total_val_loss})
        ep += 1
    
end_time = time.time()
print('Training completed in  ', str(ep), ' epochs, ',((end_time - start_time) / 60), ' minutes')
