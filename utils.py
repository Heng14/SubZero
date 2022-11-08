import numpy as np


# def uniform_selection_Ktotal(input_data, input_mask0, num_reps, rho=0.2, small_acs_block=(4, 4)):

#     if len(input_data.shape) == 4:
#         input_data = input_data[:,:,0,:]

#     if len(input_mask0.shape) == 3:
#         trn_mask_list = []
#         loss_mask_list = []
#         for i in range(input_mask0.shape[-1]):
#             input_mask_i = input_mask0[..., i]
#             input_mask_i = np.tile(input_mask_i[np.newaxis, :, :, ], (num_reps, 1, 1,))

#             nrow, ncol = input_data.shape[0], input_data.shape[1]

#             center_kx = int(find_center_ind(input_data, axes=(1, 2)))
#             center_ky = int(find_center_ind(input_data, axes=(0, 2)))

#             temp_mask = np.copy(input_mask_i)
#             temp_mask[:, center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
#             center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

#             pr = np.ndarray.flatten(temp_mask)
#             ind = np.random.choice(np.arange(num_reps * nrow * ncol),
#                                     size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

#             [indr, ind_x, ind_y] = index_flatten2nd(ind, (num_reps, nrow, ncol))

#             loss_mask = np.zeros_like(input_mask_i)
#             loss_mask[indr, ind_x, ind_y] = 1

#             trn_mask = input_mask_i - loss_mask            

#             trn_mask_list.append(trn_mask)
#             loss_mask_list.append(loss_mask)

#         trn_mask = np.stack(trn_mask_list, axis=-1)
#         loss_mask = np.stack(loss_mask_list, axis=-1)
#         return trn_mask, loss_mask



#     input_mask = np.tile(input_mask0[np.newaxis, :, :, ], (num_reps, 1, 1,))

#     nrow, ncol = input_data.shape[0], input_data.shape[1]

#     center_kx = int(find_center_ind(input_data, axes=(1, 2)))
#     center_ky = int(find_center_ind(input_data, axes=(0, 2)))

#     temp_mask = np.copy(input_mask)
#     temp_mask[:, center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
#     center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

#     pr = np.ndarray.flatten(temp_mask)
#     ind = np.random.choice(np.arange(num_reps * nrow * ncol),
#                             size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

#     [indr, ind_x, ind_y] = index_flatten2nd(ind, (num_reps, nrow, ncol))

#     loss_mask = np.zeros_like(input_mask)
#     loss_mask[indr, ind_x, ind_y] = 1

#     trn_mask = input_mask - loss_mask

#     return trn_mask, loss_mask


# def uniform_selection_bk(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)):

#     if len(input_data.shape) == 4:
#         input_data = input_data[:,:,0,:]


#     if len(input_mask.shape) == 3:
#         trn_mask_list = []
#         loss_mask_list = []


#         input_mask_i = input_mask[..., 0]

#         nrow, ncol = input_data.shape[0], input_data.shape[1]

#         center_kx = int(find_center_ind(input_data, axes=(1, 2)))
#         center_ky = int(find_center_ind(input_data, axes=(0, 2)))

#         temp_mask = np.copy(input_mask_i)
#         temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
#         center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

#         pr = np.ndarray.flatten(temp_mask)

#         ind = np.random.choice(np.arange(nrow * ncol),
#                                 size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

#         [ind_x, ind_y0] = index_flatten2nd(ind, (nrow, ncol))

#         for i in range(input_mask.shape[-1]):
#             input_mask_i = input_mask[..., i]

#             ind_y = [idy if 105 <= idy < 135 else idy + i for idy in ind_y0]

#             loss_mask = np.zeros_like(input_mask_i)
#             loss_mask[ind_x, ind_y] = 1

#             loss_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
#                 center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

#             trn_mask = input_mask_i - loss_mask            

#             trn_mask_list.append(trn_mask)
#             loss_mask_list.append(loss_mask)


#         trn_mask = np.stack(trn_mask_list, axis=-1)
#         loss_mask = np.stack(loss_mask_list, axis=-1)
#         return trn_mask, loss_mask

#     nrow, ncol = input_data.shape[0], input_data.shape[1]

#     center_kx = int(find_center_ind(input_data, axes=(1, 2)))
#     center_ky = int(find_center_ind(input_data, axes=(0, 2)))

#     temp_mask = np.copy(input_mask)
#     temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
#     center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

#     pr = np.ndarray.flatten(temp_mask)

#     ind = np.random.choice(np.arange(nrow * ncol),
#                             size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

#     [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

#     loss_mask = np.zeros_like(input_mask)
#     loss_mask[ind_x, ind_y] = 1

#     trn_mask = input_mask - loss_mask

#     return trn_mask, loss_mask


def uniform_selection(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)): # per slice gen mask

    if len(input_data.shape) == 4:
        input_data = input_data[:,:,0,:]


    if len(input_mask.shape) == 3:
        trn_mask_list = []
        loss_mask_list = []
        for i in range(input_mask.shape[-1]):
            input_mask_i = input_mask[..., i]

            nrow, ncol = input_data.shape[0], input_data.shape[1]

            center_kx = int(find_center_ind(input_data, axes=(1, 2)))
            center_ky = int(find_center_ind(input_data, axes=(0, 2)))

            temp_mask = np.copy(input_mask_i)
            temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
            center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

            pr = np.ndarray.flatten(temp_mask)

            ind = np.random.choice(np.arange(nrow * ncol),
                                    size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

            [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

            loss_mask = np.zeros_like(input_mask_i)
            loss_mask[ind_x, ind_y] = 1

            trn_mask = input_mask_i - loss_mask            

            trn_mask_list.append(trn_mask)
            loss_mask_list.append(loss_mask)

        trn_mask = np.stack(trn_mask_list, axis=-1)
        loss_mask = np.stack(loss_mask_list, axis=-1)
        return trn_mask, loss_mask

    nrow, ncol = input_data.shape[0], input_data.shape[1]

    center_kx = int(find_center_ind(input_data, axes=(1, 2)))
    center_ky = int(find_center_ind(input_data, axes=(0, 2)))

    temp_mask = np.copy(input_mask)
    temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
    center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

    pr = np.ndarray.flatten(temp_mask)

    ind = np.random.choice(np.arange(nrow * ncol),
                            size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

    [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

    loss_mask = np.zeros_like(input_mask)
    loss_mask[ind_x, ind_y] = 1

    trn_mask = input_mask - loss_mask

    return trn_mask, loss_mask


# def uniform_selection(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)): # whole 3d gen mask

#     if len(input_data.shape) == 4:
#         input_data = input_data[:,:,0,:]


#     if len(input_mask.shape) == 3:

#         nrow, ncol = input_data.shape[0], input_data.shape[1]
#         necho = input_mask.shape[-1]


#         center_kx = int(find_center_ind(input_data, axes=(1, 2)))
#         center_ky = int(find_center_ind(input_data, axes=(0, 2)))

#         temp_mask = np.copy(input_mask)
#         temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
#         center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2, :] = 0

#         pr = np.ndarray.flatten(temp_mask)

#         ind = np.random.choice(np.arange(nrow * ncol * necho),
#                                 size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

#         [ind_x, ind_y, ind_e] = index_flatten2nd(ind, (nrow, ncol, necho))

#         loss_mask = np.zeros_like(input_mask)
#         loss_mask[ind_x, ind_y, ind_e] = 1

#         trn_mask = input_mask - loss_mask            

#         return trn_mask, loss_mask

#     nrow, ncol = input_data.shape[0], input_data.shape[1]

#     center_kx = int(find_center_ind(input_data, axes=(1, 2)))
#     center_ky = int(find_center_ind(input_data, axes=(0, 2)))

#     temp_mask = np.copy(input_mask)
#     temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
#     center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

#     pr = np.ndarray.flatten(temp_mask)

#     ind = np.random.choice(np.arange(nrow * ncol),
#                             size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

#     [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

#     loss_mask = np.zeros_like(input_mask)
#     loss_mask[ind_x, ind_y] = 1

#     trn_mask = input_mask - loss_mask

#     return trn_mask, loss_mask


def getPSNR(ref, recon):
    """
    Measures PSNR between the reference and the reconstructed images
    """

    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    psnr = 20 * np.log10(np.abs(ref.max()) / (np.sqrt(mse) + 1e-10))

    return psnr


def fft(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : coil images of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform image space to k-space.

    """

    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ispace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * kspace.shape[axis]

        kspace = kspace / np.sqrt(fact)

    return kspace


def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : image space of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform k-space to image space.

    """

    ispace = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * ispace.shape[axis]

        ispace = ispace * np.sqrt(fact)

    return ispace


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm .

    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations

    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


def sense1(input_kspace, sens_maps, axes=(0, 1)):
    """
    Parameters
    ----------
    input_kspace : nrow x ncol x ncoil
    sens_maps : nrow x ncol x ncoil

    axes : The default is (0,1).

    Returns
    -------
    sense1 image

    """

    image_space = ifft(input_kspace, axes=axes, norm=None, unitary_opt=True)
    Eh_op = np.conj(sens_maps) * image_space
    sense1_image = np.sum(Eh_op, axis=axes[-1] + 1)

    return sense1_image

def sense1_multiecho(input_kspace, sens_maps, axes=(0, 1)):
    """
    Parameters
    ----------
    input_kspace : nrow x ncol x ncoil
    sens_maps : nrow x ncol x ncoil

    axes : The default is (0,1).

    Returns
    -------
    sense1 image

    """

    out = []
    for i in range(input_kspace.shape[-2]):
        input_kspace_i = input_kspace[:,:,i,:]
        image_space = ifft(input_kspace_i, axes=axes, norm=None, unitary_opt=True)
        Eh_op = np.conj(sens_maps) * image_space
        sense1_image = np.sum(Eh_op, axis=axes[-1] + 1)
        out.append(sense1_image)

    sense1_image = np.stack(out, axis=-1)
    return sense1_image


# def sense1_multiecho_reps(input_kspace, sens_maps, axes=(1, 2)):
#     """
#     Parameters
#     ----------
#     input_kspace : nrow x ncol x ncoil
#     sens_maps : nrow x ncol x ncoil

#     axes : The default is (0,1).

#     Returns
#     -------
#     sense1 image

#     """


#     out = []
#     for i in range(input_kspace.shape[-2]):
#         input_kspace_i = input_kspace[:,:,i,:]
#         image_space = ifft(input_kspace_i, axes=axes, norm=None, unitary_opt=True)
#         Eh_op = np.conj(sens_maps) * image_space
#         sense1_image = np.sum(Eh_op, axis=axes[-1] + 1)
#         out.append(sense1_image)

#     sense1_image = np.stack(out, axis=-1)
#     return sense1_image



def complex2real(input_data):
    """
    Parameters
    ----------
    input_data : row x col
    dtype :The default is np.float32.

    Returns
    -------
    output : row x col x 2

    """

    return np.stack((input_data.real, input_data.imag), axis=-1)


def real2complex(input_data):
    """
    Parameters
    ----------
    input_data : row x col x 2

    Returns
    -------
    output : row x col

    """

    return input_data[..., 0] + 1j * input_data[..., 1]
