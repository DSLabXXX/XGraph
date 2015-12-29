from scipy import *
from scipy.sparse import *
import numpy as np

__author__ = 'IanKuo'


def linear_combination(m, n, tensor_x, w):
    
    """
    Merge a tensor into a matrix by linear combination 
    :param m: number of rows
    :param n: number of columns
    :param tensor_x: the tensor to be combined to a matrix linearly
    :param w: the weight for linear combination
    :return: the combined matrix 
    """
    
    if len(w) != len(tensor_x):
        return 0

    ret_mat = csr_matrix((m, n), dtype=float)

    for i in range(len(w)):
        ret_mat = ret_mat + w[i] * tensor_x[i]

    return ret_mat


def gradient_mn(drv_mat_xn_drv_vec_w, drv_mat_yn_drv_vec_w, drv_mat_zn_drv_vec_w, 
                mat_xn, mat_yn, mat_zn, 
                x_num, y_num, z_num, 
                vec_w, inv_mat_d_n, vec_col_sum_n, tensor_w, index):

    """
    Output the gradient of a matrix with respect to vector vec_w
    This function is to implement Eq.(27).

    :param drv_mat_xn_drv_vec_w: gradient of matrix mat_xn w.r.t vec_w
    :param drv_mat_yn_drv_vec_w: gradient of matrix mat_yn w.r.t vec_w
    :param drv_mat_zn_drv_vec_w: gradient of matrix mat_zn w.r.t vec_w
    :param mat_xn: matrix mat_xn
    :param mat_yn: matrix mat_xn
    :param mat_zn: matrix mat_xn
    :param x_num: number of x (i.e., user)
    :param y_num: number of y (i.e., item)
    :param z_num: number of z (i.e., tag)
    :param vec_w: vector vec_w
    :param inv_mat_d_n: inverse matrix of mat_d_n
    :param vec_col_sum_n: 
    :param tensor_w: tensor of a matrix
    :param index: 1: drv -> mat_xn; 2: drv -> mat_yn; drv -> mat_zn
    :return:
    """
    w_len = len(vec_w)

    for i in range(w_len):

        if index == 1:
            drv_mat_a_drv_w = csr_matrix(mat_xn, copy=True)
            drv_mat_a_drv_w.data[:] = mat_xn.data * (1 - mat_xn.data) * tensor_w[i].data

            a = np.ones((1, x_num)) * drv_mat_a_drv_w
            b = np.power(vec_col_sum_n, 2) + ones(shape(a)) * 10e-50
            t = np.divide(a, b)
            inv_drv_mat_d_n_drv_w = - dia_matrix((t, array([0])), shape(inv_mat_d_n))

            drv_mat_xn_drv_vec_w.append(drv_mat_a_drv_w * inv_mat_d_n + mat_xn * inv_drv_mat_d_n_drv_w)
            drv_mat_yn_drv_vec_w.append(mat_yn * inv_drv_mat_d_n_drv_w)
            drv_mat_zn_drv_vec_w.append(mat_zn * inv_drv_mat_d_n_drv_w)

        elif index == 2:
            drv_mat_a_drv_w = csr_matrix(mat_yn, copy=True)
            drv_mat_a_drv_w.data[:] = mat_yn.data * (1 - mat_yn.data) * tensor_w[i].data

            a = np.ones((1, y_num)) * drv_mat_a_drv_w
            b = np.power(vec_col_sum_n, 2) + ones(shape(a)) * 10e-50
            t = np.divide(a, b)
            inv_drv_mat_d_n_drv_w = - dia_matrix((t, array([0])), shape(inv_mat_d_n))

            drv_mat_xn_drv_vec_w.append(mat_xn * inv_drv_mat_d_n_drv_w)
            drv_mat_yn_drv_vec_w.append(drv_mat_a_drv_w * inv_mat_d_n + mat_yn * inv_drv_mat_d_n_drv_w)
            drv_mat_zn_drv_vec_w.append(mat_zn * inv_drv_mat_d_n_drv_w)

        elif index == 3:
            drv_mat_a_drv_w = csr_matrix(mat_zn, copy=True)
            drv_mat_a_drv_w.data[:] = mat_zn.data * (1 - mat_zn.data) * tensor_w[i].data

            a = np.ones((1, z_num)) * drv_mat_a_drv_w
            b = np.power(vec_col_sum_n, 2) + ones(shape(a)) * 10e-50
            t = np.divide(a, b)
            inv_drv_mat_d_n_drv_w = - dia_matrix((t, array([0])), shape(inv_mat_d_n))

            drv_mat_xn_drv_vec_w.append(mat_xn * inv_drv_mat_d_n_drv_w)
            drv_mat_yn_drv_vec_w.append(mat_yn * inv_drv_mat_d_n_drv_w)
            drv_mat_zn_drv_vec_w.append(drv_mat_a_drv_w * inv_mat_d_n + mat_zn * inv_drv_mat_d_n_drv_w)

        else:
            return
