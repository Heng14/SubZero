import tensorflow as tf
# import tensorflow.compat.v1 as tf
import math
import os
import numpy as np
import parser_ops
import UnrollNet_subspace_2diffmask_p1_se as UnrollNet_subspace
# tf.disable_v2_behavior()

parser = parser_ops.get_parser()
args = parser.parse_args()


def test_graph(directory, Uk):
    """
    This function creates a test graph for testing
    """

    tf.reset_default_graph()
    Uk_inv = np.linalg.pinv(Uk) # by heng
    UkT_tensor = tf.convert_to_tensor(Uk.T, dtype=tf.complex64)
    Uk_invT_tensor = tf.convert_to_tensor(Uk_inv.T, dtype=tf.complex64)

    # %% placeholders for the unrolled network
    sens_mapsP = tf.placeholder(tf.complex64, shape=(None, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB), name='sens_maps')
    trn_maskP = tf.placeholder(tf.complex64, shape=(None, args.nrow_GLOB, args.ncol_GLOB, args.n_echo), name='trn_mask_subspace')
    loss_maskP = tf.placeholder(tf.complex64, shape=(None, args.nrow_GLOB, args.ncol_GLOB, args.n_echo), name='loss_mask_subspace')
    # nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, args.n_echo, 2), name='nw_input') # by heng
    nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, args.n_basis, 2), name='nw_input_subspace')

    nw_output, nw_kspace_output, nw_kspace_output_inv, x0, all_intermediate_outputs, mu = \
               UnrollNet_subspace.UnrolledNet(nw_inputP, sens_mapsP, trn_maskP, loss_maskP, UkT_tensor, Uk_invT_tensor).model

    # %% unrolled network outputs
    nw_output = tf.identity(nw_output, name='nw_output_subspace')
    nw_kspace_output = tf.identity(nw_kspace_output, name='nw_kspace_output_subspace')
    nw_kspace_output_inv = tf.identity(nw_kspace_output_inv, name='nw_kspace_output_inv_subspace')
    all_intermediate_outputs = tf.identity(all_intermediate_outputs, name='all_intermediate_outputs_subspace')
    # x0 = tf.identity(x0, name='x0')
    mu = tf.identity(mu, name='mu')

    # %% saves computational graph for test
    saver = tf.train.Saver()
    sess_test_filename = os.path.join(directory, 'model_test_subspace')
    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(tf.global_variables_initializer())
        saved_test_model = saver.save(sess, sess_test_filename, latest_filename='checkpoint_test_subspace')

    print('\n Test graph is generated and saved at: ' + saved_test_model)

    return True


def tf_complex2real(input_data):
    """
    Parameters
    ----------
    input_data : nrow x ncol.

    Returns
    -------
    outputs concatenated real and imaginary parts as nrow x ncol x 2

    """

    return tf.stack([tf.real(input_data), tf.imag(input_data)], axis=-1)


def tf_real2complex(input_data):
    """
    Parameters
    ----------
    input_data : nrow x ncol x 2

    Returns
    -------
    merges concatenated channels and outputs complex image of size nrow x ncol.

    """

    return tf.complex(input_data[..., 0], input_data[..., 1])


def tf_fftshift_flip2D(input_data, axes=1):
    """
    Parameters
    ----------
    input_data : ncoil x nrow x ncol
    axes :  The default is 1.
    ------

    """

    nx = math.ceil(args.nrow_GLOB / 2)
    ny = math.ceil(args.ncol_GLOB / 2)

    if axes == 1:

        first_half = tf.identity(input_data[:, :nx, :])
        second_half = tf.identity(input_data[:, nx:, :])

    elif axes == 2:

        first_half = tf.identity(input_data[:, :, :ny])
        second_half = tf.identity(input_data[:, :, ny:])

    else:
        raise ValueError('Invalid axes for fftshift')

    return tf.concat([second_half, first_half], axis=axes)


def tf_ifftshift_flip2D(input_data, axes=1):
    """
    Parameters
    ----------
    input_data : ncoil x nrow x ncol
    axes :  The default is 1.
    ------

    """

    nx = math.floor(args.nrow_GLOB / 2)
    ny = math.floor(args.ncol_GLOB / 2)

    if axes == 1:

        first_half = tf.identity(input_data[:, :nx, :])
        second_half = tf.identity(input_data[:, nx:, :])

    elif axes == 2:

        first_half = tf.identity(input_data[:, :, :ny])
        second_half = tf.identity(input_data[:, :, ny:])

    else:
        raise ValueError('Invalid axes for ifftshift')

    return tf.concat([second_half, first_half], axis=axes)


def tf_fftshift(input_x, axes=1):
    """
    Parameters
    ----------
    input_x : ncoil x nrow x ncol
    axes : The default is 1.

    """

    return tf_fftshift_flip2D(tf_fftshift_flip2D(input_x, axes=1), axes=2)


def tf_ifftshift(input_x, axes=1):
    """
    Parameters
    ----------
    input_x : ncoil x nrow x ncol
    axes : The default is 1.

    """

    return tf_ifftshift_flip2D(tf_ifftshift_flip2D(input_x, axes=1), axes=2)
