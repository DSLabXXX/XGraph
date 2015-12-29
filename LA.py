from scipy import *
from scipy.sparse import *
import numpy as np

__author__ = 'IanKuo'


def linear_combination(m, n, tensor_x, w):

    """
    Merge a tensor into a matrix by linear combination
    :param m: number of rows
    :param m: number of columns
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


def gradient_mn(drv_mat_xn, drv_mat_yn, drv_mat_zn, 
                mat_xn, mat_yn, mat_zn, 
                x_num, y_num, z_num, 
                w, inv_mat_d_n, col_sum_n, mat_w, index):

    w_len = len(w)

    for i in range(w_len):

        if index == 1:
            drv_mat_a_w = csr_matrix(mat_xn, copy=True)
            drv_mat_a_w.data[:] = mat_xn.data * (1 - mat_xn.data) * mat_w[i].data

            a = np.ones((1, x_num)) * drv_mat_a_w
            b = np.power(col_sum_n, 2) + ones(shape(a)) * 10e-50
            t = np.divide(a, b)
            inv_dD_n_w = - dia_matrix((t, array([0])), shape(inv_mat_d_n))

            drv_mat_xn.append(drv_mat_a_w * inv_mat_d_n + mat_xn * inv_dD_n_w)
            drv_mat_yn.append(mat_yn * inv_dD_n_w)
            drv_mat_zn.append(mat_zn * inv_dD_n_w)

        elif index == 2:
            drv_mat_a_w = csr_matrix(mat_yn, copy=True)
            drv_mat_a_w.data[:] = mat_yn.data * (1 - mat_yn.data) * mat_w[i].data

            a = np.ones((1, y_num)) * drv_mat_a_w
            b = np.power(col_sum_n, 2) + ones(shape(a)) * 10e-50
            t = np.divide(a, b)
            inv_dD_n_w = - dia_matrix((t, array([0])), shape(inv_mat_d_n))

            drv_mat_xn.append(mat_xn * inv_dD_n_w)
            drv_mat_yn.append(drv_mat_a_w * inv_mat_d_n + mat_yn * inv_dD_n_w)
            drv_mat_zn.append(mat_zn * inv_dD_n_w)

        elif index == 3:
            drv_mat_a_w = csr_matrix(mat_zn, copy=True)
            drv_mat_a_w.data[:] = mat_zn.data * (1 - mat_zn.data) * mat_w[i].data

            a = np.ones((1, z_num)) * drv_mat_a_w
            b = np.power(col_sum_n, 2) + ones(shape(a)) * 10e-50
            t = np.divide(a, b)
            inv_dD_n_w = - dia_matrix((t, array([0])), shape(inv_mat_d_n))

            drv_mat_xn.append(mat_xn * inv_dD_n_w)
            drv_mat_yn.append(mat_yn * inv_dD_n_w)
            drv_mat_zn.append(drv_mat_a_w * inv_mat_d_n + mat_zn * inv_dD_n_w)

        else:
            return
