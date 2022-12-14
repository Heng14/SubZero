import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='ZS-SSL: Zero-Shot Self-Supervised Learning')

    # %% hyperparameters for the  network
    parser.add_argument('--data_name', type=str, default='T1', # ZS_SSL, spindata
                    help='name of dataset')
    parser.add_argument('--data_opt', type=str, default='subspace_slice5_2diff_2subspace_prll0.05_Ry8_9echo_basis4_augmask_online_se', # augmask_online_maskgenwhole_se, predefmask, augmask_online_se
                    help='type of dataset')
    parser.add_argument('--data_dir', type=str, default='T1_slice5_Ry8_9echo.mat',
                    help='data directory')                
    parser.add_argument('--nrow_GLOB', type=int, default=256, # 256
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol_GLOB', type=int, default=192, # 208
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--n_echo', type=int, default=9, # 16
                        help='number of echos of the slices in the dataset')
    parser.add_argument('--ncoil_GLOB', type=int, default=8, #12
                        help='number of coils of the slices in the dataset')   
    parser.add_argument('--n_basis', type=int, default=4, # default 3
                        help='number of basis')                
    parser.add_argument('--acc_rate', type=int, default=8, # 4
                        help='acceleration rate')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='batch size')
    parser.add_argument('--nb_unroll_blocks', type=int, default=5, #10
                        help='number of unrolled blocks')
    parser.add_argument('--nb_res_blocks', type=int, default=10, #15
                        help="number of residual blocks in ResNet")
    parser.add_argument('--CG_Iter', type=int, default=5, #10
                        help='number of Conjugate Gradient iterations for DC')

    # %% hyperparameters for the zs-ssl
    parser.add_argument('--rho_val', type=float, default=0.2,
                        help='cardinality of the validation mask (\Gamma)')                        
    parser.add_argument('--rho_train', type=float, default=0.4,
                        help='cardinality of the loss mask, \ rho = |\ Lambda| / |\ Omega|')
    parser.add_argument('--num_reps', type=int, default=25,
                        help='number of repetions for the remainder mask (\Omega \ \Gamma) ')
    parser.add_argument('--transfer_learning', type=bool, default=False,
                        help='transfer learning from pretrained model')
    parser.add_argument('--TL_path', type=str, default=None,
                        help='path to pretrained model')                                           
    parser.add_argument('--stop_training', type=int, default=25,
                        help='stop training if a new lowest validation loss hasnt been achieved in xx epochs')                    
    
    return parser
