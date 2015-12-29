import numpy as np
from scipy import *
from scipy.sparse import *

import ReadFile
import LA


__author__ = 'IanKuo'


val_alpha = 0.8
iter_num = 20
val_beta = 10000.
val_learning_rate = 10.

valConvergeThreshold_J = 10e-120

str_uit_graph_path = 'LastFm/UIT_GroundTruth.txt'
str_uu_graph_path = ''
str_ii_graph_path = ''
str_tt_graph_path = ''
str_train_instance_path = 'LastFm/UIT_Train.txt'

val_feature_num_uu = 0
val_feature_num_ui = 1
val_feature_num_ut = 1
val_feature_num_iu = val_feature_num_ui
val_feature_num_ii = 0
val_feature_num_it = 1
val_feature_num_tu = val_feature_num_ut
val_feature_num_ti = val_feature_num_it
val_feature_num_tt = 0

val_sizeof_xi_u = 1
val_sizeof_xi_i = 1

#
# dictionary of positive/negative instances
# (key: (vertex, vertex, vertex), value: [vertex])
#
list_training_instance = list()


#
# vector Theta denotes the feature weight
#
vec_theta_uu = np.ones(val_feature_num_uu) / val_feature_num_uu     # []
vec_theta_ui = np.ones(val_feature_num_ui) / val_feature_num_ui     # [-0.24718311]
vec_theta_ut = np.ones(val_feature_num_ut) / val_feature_num_ut     # [-157.3980413]
vec_theta_iu = np.ones(val_feature_num_ut) / val_feature_num_ut     # [-0.24718311]
vec_theta_ii = np.ones(val_feature_num_ii) / val_feature_num_ii     # []
vec_theta_it = np.ones(val_feature_num_it) / val_feature_num_it     # [0.67666965]
vec_theta_tu = np.ones(val_feature_num_it) / val_feature_num_it     # [-157.3980413]
vec_theta_ti = np.ones(val_feature_num_it) / val_feature_num_it     # [0.67666965]
vec_theta_tt = np.ones(val_feature_num_tt) / val_feature_num_tt     # []


#
# vector Xi denotes the query weight
#
vec_xi_u = np.ones(val_sizeof_xi_u) / val_sizeof_xi_u * 1000     # [4.5065864]
vec_xi_i = np.ones(val_sizeof_xi_i) / val_sizeof_xi_i * 1000     # [0.11826436]


#
# adjacent matrices definition
# [ UU  UI  UT ]
# [ IU  II  IT ]
# [ TU  TI  TT ]
#
tensor_x_uu = list()  # val_feature_num_uu
tensor_x_ui = list()  # val_feature_num_ui
tensor_x_ut = list()  # val_feature_num_ut
tensor_x_iu = list()
tensor_x_ii = list()  # val_feature_num_ii
tensor_x_it = list()  # val_feature_num_it
tensor_x_tu = list()
tensor_x_ti = list()
tensor_x_tt = list()  # val_feature_num_tt

drv_mat_a_uu_drv_vec_theta_uu = list()  # partial(mat_a_uu) / partial(theta_UU)
drv_mat_a_uu_drv_vec_theta_iu = list()
drv_mat_a_uu_drv_vec_theta_tu = list()
drv_mat_a_ui_drv_vec_theta_ui = list()
drv_mat_a_ui_drv_vec_theta_ii = list()
drv_mat_a_ui_drv_vec_theta_ti = list()
drv_mat_a_ut_drv_vec_theta_ut = list()
drv_mat_a_ut_drv_vec_theta_it = list()
drv_mat_a_ut_drv_vec_theta_tt = list()
drv_mat_a_iu_drv_vec_theta_uu = list()
drv_mat_a_iu_drv_vec_theta_iu = list()
drv_mat_a_iu_drv_vec_theta_tu = list()
drv_mat_a_ii_drv_vec_theta_ui = list()
drv_mat_a_ii_drv_vec_theta_ii = list()
drv_mat_a_ii_drv_vec_theta_ti = list()
drv_mat_a_it_drv_vec_theta_ut = list()
drv_mat_a_it_drv_vec_theta_it = list()
drv_mat_a_it_drv_vec_theta_tt = list()
drv_mat_a_tu_drv_vec_theta_uu = list()
drv_mat_a_tu_drv_vec_theta_iu = list()
drv_mat_a_tu_drv_vec_theta_tu = list()
drv_mat_a_ti_drv_vec_theta_ui = list()
drv_mat_a_ti_drv_vec_theta_ii = list()
drv_mat_a_ti_drv_vec_theta_ti = list()
drv_mat_a_tt_drv_vec_theta_ut = list()
drv_mat_a_tt_drv_vec_theta_it = list()
drv_mat_a_tt_drv_vec_theta_tt = list()

# read positive/negative instances
ReadFile.read_uit_file(str_uit_graph_path, tensor_x_ui, tensor_x_ut, tensor_x_it)

for i in tensor_x_ui:
    tensor_x_iu.append(i.transpose())

for i in tensor_x_ut:
    tensor_x_tu.append(i.transpose())

for i in tensor_x_it:
    tensor_x_ti.append(i.transpose())

print(shape(tensor_x_ui[0]))
print(shape(tensor_x_ut[0]))
print(shape(tensor_x_it[0]))

val_user_num = shape(tensor_x_ui[0])[0]
val_item_num = shape(tensor_x_ui[0])[1]
val_tag_num = shape(tensor_x_ut[0])[1]
#
# query vector definition
#     [ U ]
#     [ I ]
#     [ T ]
#

'''
# read UU graph
ReadFile.readXXFile(str_uu_graph_path, tensor_x_uu)

# read II graph
ReadFile.readXXFile(str_ii_graph_path, tensor_x_ii)

# read TT graph
ReadFile.readXXFile(str_tt_graph_path, tensor_x_tt)
'''

#
# read training instances
#
#  user \t item \t posIns1,posIns2,... \t negIns1,negIns2,...
#
with open(str_train_instance_path, 'r', encoding='utf8') as file_train_instances:
    for line in file_train_instances:
        list_training_instance.append(line)

#
# create matA
#

if len(tensor_x_uu) != 0:
    mat_a_uu = LA.linear_combination(val_user_num, val_user_num, tensor_x_uu, vec_theta_uu)
    mat_a_uu.data[:] = 2 / (1 + exp(-1 * mat_a_uu.data)) - 1
else:
    mat_a_uu = csr_matrix((val_user_num, val_user_num), dtype=float)

if len(tensor_x_ui) != 0:
    mat_a_ui = LA.linear_combination(val_user_num, val_item_num, tensor_x_ui, vec_theta_ui)
    mat_a_ui.data[:] = 2 / (1 + exp(-1 * mat_a_ui.data)) - 1
else:
    mat_a_ui = csr_matrix((val_user_num, val_item_num), dtype=float)

if len(tensor_x_ut) != 0:
    mat_a_ut = LA.linear_combination(val_user_num, val_tag_num, tensor_x_ut, vec_theta_ut)
    mat_a_ut.data[:] = 2 / (1 + exp(-1 * mat_a_ut.data)) - 1
else:
    mat_a_ut = csr_matrix((val_user_num, val_tag_num), dtype=float)

if len(tensor_x_ui) != 0:
    mat_a_iu = LA.linear_combination(val_item_num, val_user_num, tensor_x_iu, vec_theta_iu)
    mat_a_iu.data[:] = 2 / (1 + exp(-1 * mat_a_iu.data)) - 1
else:
    mat_a_iu = csr_matrix((val_item_num, val_user_num), dtype=float)

if len(tensor_x_ii) != 0:
    mat_a_ii = LA.linear_combination(val_item_num, val_item_num, tensor_x_ii, vec_theta_ii)
    mat_a_ii.data[:] = 2 / (1 + exp(-1 * mat_a_ii.data)) - 1
else:
    mat_a_ii = csr_matrix((val_item_num, val_item_num), dtype=float)

if len(tensor_x_it) != 0:
    mat_a_it = LA.linear_combination(val_item_num, val_tag_num, tensor_x_it, vec_theta_it)
    mat_a_it.data[:] = 2 / (1 + exp(-1 * mat_a_it.data)) - 1
else:
    mat_a_it = csr_matrix((val_item_num, val_tag_num), dtype=float)

if len(tensor_x_ut) != 0:
    mat_a_tu = LA.linear_combination(val_tag_num, val_user_num, tensor_x_tu, vec_theta_tu)
    mat_a_tu.data[:] = 2 / (1 + exp(-1 * mat_a_tu.data)) - 1
else:
    mat_a_tu = csr_matrix((val_tag_num, val_user_num), dtype=float)

if len(tensor_x_ti) != 0:
    mat_a_ti = LA.linear_combination(val_tag_num, val_item_num, tensor_x_ti, vec_theta_ti)
    mat_a_ti.data[:] = 2 / (1 + exp(-1 * mat_a_ti.data)) - 1
else:
    mat_a_ti = csr_matrix((val_tag_num, val_item_num), dtype=float)

if len(tensor_x_tt) != 0:
    mat_a_tt = LA.linear_combination(val_tag_num, val_tag_num, tensor_x_tt, vec_theta_tt)
    mat_a_tt.data[:] = 2 / (1 + exp(-1 * mat_a_tt.data)) - 1
else:
    mat_a_tt = csr_matrix((val_tag_num, val_tag_num), dtype=float)

print('mat_a_uu : ' + str(shape(mat_a_uu)))
print('mat_a_ui : ' + str(shape(mat_a_ui)))
print('mat_a_ut : ' + str(shape(mat_a_ut)))
print('mat_a_iu : ' + str(shape(mat_a_iu)))
print('mat_a_ii : ' + str(shape(mat_a_ii)))
print('mat_a_it : ' + str(shape(mat_a_it)))
print('mat_a_tu : ' + str(shape(mat_a_tu)))
print('mat_a_ti : ' + str(shape(mat_a_ti)))
print('mat_a_tt : ' + str(shape(mat_a_tt)))

#
# D_U^{-1} => Eq.(8)
#

vec_sum_col_u = csr_matrix(np.ones((1, val_user_num)) * mat_a_uu +
                           np.ones((1, val_item_num)) * mat_a_iu +
                           np.ones((1, val_tag_num)) * mat_a_tu)

vec_sum_col_i = csr_matrix(np.ones((1, val_user_num)) * mat_a_ui +
                           np.ones((1, val_item_num)) * mat_a_ii +
                           np.ones((1, val_tag_num)) * mat_a_ti)

vec_sum_col_t = csr_matrix(np.ones((1, val_user_num)) * mat_a_ut +
                           np.ones((1, val_item_num)) * mat_a_it +
                           np.ones((1, val_tag_num)) * mat_a_tt)

vec_inv_sum_col_t = vec_sum_col_u
vec_inv_sum_col_i = vec_sum_col_i
vec_inv_sum_col_t = vec_sum_col_t

vec_inv_sum_col_t.data[:] = power(vec_inv_sum_col_t.data, -1)
inv_mat_d_u = dia_matrix((vec_inv_sum_col_t.toarray(), array([0])), shape=(val_user_num, val_user_num))

vec_inv_sum_col_i.data[:] = power(vec_inv_sum_col_i.data, -1)
inv_mat_d_i = dia_matrix((vec_inv_sum_col_i.toarray(), array([0])), shape=(val_item_num, val_item_num))

vec_inv_sum_col_t.data[:] = power(vec_inv_sum_col_t.data, -1)
inv_mat_d_t = dia_matrix((vec_inv_sum_col_t.toarray(), array([0])), shape=(val_tag_num, val_tag_num))

vec_sum_col_u = vec_sum_col_u.todense()
vec_sum_col_i = vec_sum_col_i.todense()
vec_sum_col_t = vec_sum_col_t.todense()

vec_inv_sum_col_t = vec_inv_sum_col_t.todense()
vec_inv_sum_col_i = vec_inv_sum_col_i.todense()
vec_inv_sum_col_t = vec_inv_sum_col_t.todense()

print('val_user_num : ' + str(val_user_num))
print('val_item_num : ' + str(val_item_num))
print('val_tag_num : ' + str(val_tag_num))

print('inv_mat_d_u : ' + str(shape(inv_mat_d_u)))
print('inv_mat_d_i : ' + str(shape(inv_mat_d_i)))
print('inv_mat_d_t : ' + str(shape(inv_mat_d_t)))

#
# column normalization on transition probabilities matrices
#
mat_a_uu_col_norm = mat_a_uu * inv_mat_d_u
mat_a_ui_col_norm = mat_a_ui * inv_mat_d_i
mat_a_ut_col_norm = mat_a_ut * inv_mat_d_t
mat_a_iu_col_norm = mat_a_iu * inv_mat_d_u
mat_a_ii_col_norm = mat_a_ii * inv_mat_d_i
mat_a_it_col_norm = mat_a_it * inv_mat_d_t
mat_a_tu_col_norm = mat_a_tu * inv_mat_d_u
mat_a_ti_col_norm = mat_a_ti * inv_mat_d_i
mat_a_tt_col_norm = mat_a_tt * inv_mat_d_t

# Theta_UU
LA.gradient_mn(drv_mat_a_uu_drv_vec_theta_uu, drv_mat_a_iu_drv_vec_theta_uu, drv_mat_a_tu_drv_vec_theta_uu,
               mat_a_uu, mat_a_iu, mat_a_tu, val_user_num, val_item_num, val_tag_num,
               vec_theta_uu, inv_mat_d_u, vec_sum_col_u, tensor_x_uu, 1)

# Theta_UI
LA.gradient_mn(drv_mat_a_ui_drv_vec_theta_ui, drv_mat_a_ii_drv_vec_theta_ui, drv_mat_a_ti_drv_vec_theta_ui,
               mat_a_ui, mat_a_ii, mat_a_ti, val_user_num, val_item_num, val_tag_num,
               vec_theta_ui, inv_mat_d_i, vec_sum_col_i, tensor_x_ui, 1)

# Theta_UT
LA.gradient_mn(drv_mat_a_ut_drv_vec_theta_ut, drv_mat_a_it_drv_vec_theta_ut, drv_mat_a_tt_drv_vec_theta_ut,
               mat_a_ut, mat_a_it, mat_a_tt, val_user_num, val_item_num, val_tag_num,
               vec_theta_ut, inv_mat_d_t, vec_sum_col_t, tensor_x_ut, 1)

# Theta_IU
LA.gradient_mn(drv_mat_a_uu_drv_vec_theta_iu, drv_mat_a_iu_drv_vec_theta_iu, drv_mat_a_tu_drv_vec_theta_iu,
               mat_a_uu, mat_a_iu, mat_a_tu, val_user_num, val_item_num, val_tag_num,
               vec_theta_iu, inv_mat_d_u, vec_sum_col_u, tensor_x_ui, 2)

# Theta_II
LA.gradient_mn(drv_mat_a_ui_drv_vec_theta_ii, drv_mat_a_ii_drv_vec_theta_ii, drv_mat_a_ti_drv_vec_theta_ii,
               mat_a_ui, mat_a_ii, mat_a_ti, val_user_num, val_item_num, val_tag_num,
               vec_theta_ii, inv_mat_d_i, vec_sum_col_i, tensor_x_ii, 2)

# Theta_IT
LA.gradient_mn(drv_mat_a_ut_drv_vec_theta_it, drv_mat_a_it_drv_vec_theta_it, drv_mat_a_tt_drv_vec_theta_it,
               mat_a_ut, mat_a_it, mat_a_tt, val_user_num, val_item_num, val_tag_num,
               vec_theta_it, inv_mat_d_t, vec_sum_col_t, tensor_x_it, 2)

# Theta_TU
LA.gradient_mn(drv_mat_a_uu_drv_vec_theta_tu, drv_mat_a_iu_drv_vec_theta_tu, drv_mat_a_tu_drv_vec_theta_tu,
               mat_a_uu, mat_a_iu, mat_a_tu, val_user_num, val_item_num, val_tag_num,
               vec_theta_tu, inv_mat_d_u, vec_sum_col_u, tensor_x_ut, 3)

# Theta_TI
LA.gradient_mn(drv_mat_a_ui_drv_vec_theta_ti, drv_mat_a_ii_drv_vec_theta_ti, drv_mat_a_ti_drv_vec_theta_ti,
               mat_a_ui, mat_a_ii, mat_a_ti, val_user_num, val_item_num, val_tag_num,
               vec_theta_ti, inv_mat_d_i, vec_sum_col_i, tensor_x_it, 3)

# Theta_TT
LA.gradient_mn(drv_mat_a_ut_drv_vec_theta_tt, drv_mat_a_it_drv_vec_theta_tt, drv_mat_a_tt_drv_vec_theta_tt,
               mat_a_ut, mat_a_it, mat_a_tt, val_user_num, val_item_num, val_tag_num,
               vec_theta_tt, inv_mat_d_t, vec_sum_col_t, tensor_x_tt, 3)


#
# the derivatives of the parameter: Xi in Eq.(30)-(32)
#
drv_vec_p_u_drv_vec_xi_u = csr_matrix((val_user_num, val_sizeof_xi_u), dtype=float)
drv_vec_p_u_drv_vec_xi_i = csr_matrix((val_user_num, val_sizeof_xi_i), dtype=float)
drv_vec_p_i_drv_vec_xi_u = csr_matrix((val_item_num, val_sizeof_xi_u), dtype=float)
drv_vec_p_i_drv_vec_xi_i = csr_matrix((val_item_num, val_sizeof_xi_i), dtype=float)

isConverge = False
line_count = 0

for line in list_training_instance:

    line_count += 1

    if line_count < 0:
        continue

    if isConverge:
        break
    #
    # set query vector
    #
    l = line.strip('\n').split('\t')

    if len(l) < 4:
        continue

    user = int(l[0])
    item = int(l[1])
    list_positive_tags = list(map(int, l[2].split(',')))
    list_negative_tags = list(map(int, l[3].split(',')))
    val_sizeof_positive_tags = len(list_positive_tags)
    val_sizeof_negative_tags = len(list_negative_tags)

    #
    # probability distribution vector definition
    #     [ U ]
    #     [ I ]
    #     [ T ]
    #

    # Y
    vec_y_u = np.zeros((val_user_num, 1))
    vec_y_i = np.zeros((val_item_num, 1))

    ''''
    # preference vector
    vec_p_u = np.zeros((val_user_num, 1))
    vec_p_i = np.zeros((val_item_num, 1))
    '''
    vec_p_t = np.zeros((val_tag_num, 1))

    '''
    # query vector
    vec_q_u = np.zeros((val_user_num, 1))
    vec_q_i = np.zeros((val_item_num, 1))
    '''
    vec_q_t = np.zeros((val_tag_num, 1))

    if val_sizeof_positive_tags == 0 or val_sizeof_negative_tags == 0:
        continue

    print('\n' + str(line_count) + '\nuser: ' + str(user) + ', item: ' + str(item))

    vec_y_u[user, 0] = 1
    vec_y_i[item, 0] = 1

    # query vector
    vec_q_u = vec_y_u * vec_xi_u
    vec_q_i = vec_y_i * vec_xi_u
    vec_q_u = 2. / (1. + np.exp(-1 * vec_q_u)) - ones(shape(vec_q_u))
    vec_q_i = 2. / (1. + np.exp(-1 * vec_q_i)) - ones(shape(vec_q_i))
    vec_q_t = 2. / (1. + np.exp(-1 * vec_q_t)) - ones(shape(vec_q_t))

    # preference vector
    vec_p_u = vec_y_u * vec_xi_u
    vec_p_i = vec_y_i * vec_xi_u    
    vec_p_u = 2. / (1. + np.exp(-1 * vec_p_u)) - ones(shape(vec_q_u))
    vec_p_i = 2. / (1. + np.exp(-1 * vec_p_i)) - ones(shape(vec_q_i))
    vec_p_t = 2. / (1. + np.exp(-1 * vec_p_t)) - ones(shape(vec_q_t))

    val_sum_col_vec_p = np.sum(vec_p_u) + np.sum(vec_p_i) + np.sum(vec_p_t)

    vec_q_u /= val_sum_col_vec_p
    vec_q_i /= val_sum_col_vec_p
    vec_q_t /= val_sum_col_vec_p

    vec_p_u /= val_sum_col_vec_p
    vec_p_i /= val_sum_col_vec_p
    vec_p_t /= val_sum_col_vec_p

    vec_p_u = csr_matrix(vec_p_u, shape=(val_user_num, 1))
    vec_p_i = csr_matrix(vec_p_i, shape=(val_item_num, 1))
    vec_p_t = csr_matrix(vec_p_t, shape=(val_tag_num, 1))

    #
    # Calculate the distribution by random walk with restart
    #
    for r in range(iter_num):
        tmp = mat_a_uu_col_norm * vec_q_u + mat_a_ui_col_norm * vec_q_i + mat_a_ut_col_norm * vec_q_t
        vec_q_u = (1 - val_alpha) * tmp + val_alpha * vec_p_u
        
        tmp = mat_a_iu_col_norm * vec_q_u + mat_a_ii_col_norm * vec_q_i + mat_a_it_col_norm * vec_q_t
        vec_q_i = (1 - val_alpha) * tmp + val_alpha * vec_p_i
        
        tmp = mat_a_tu_col_norm * vec_q_u + mat_a_ti_col_norm * vec_q_i + mat_a_tt_col_norm * vec_q_t
        vec_q_t = (1 - val_alpha) * tmp + val_alpha * vec_p_t

    listQQQT = np.argsort(np.asarray(vec_q_t.transpose()), kind='heapsort')[0][::-1][0:10]

    count = 0

    set_results = set(listQQQT)
    set_ground_truth = set(list_positive_tags)
    set_retrive = set_ground_truth & set_results

    print('Precision@20 = ' + str(len(set_retrive) / 20))

    for i in range(val_sizeof_xi_u):
        t = (np.ones(shape(vec_p_u)) - vec_p_u)
        d = np.multiply(vec_p_u.todense(), t)
        dP_U_U = np.multiply(d, vec_y_u)
        inv_dD_U = -1 * sum(dP_U_U) / (val_sum_col_vec_p ** 2)
        drv_vec_p_u_drv_vec_xi_u[:, i] = dP_U_U / val_sum_col_vec_p + vec_p_u * inv_dD_U

        drv_vec_p_i_drv_vec_xi_u[:, i] = vec_q_i * inv_dD_U

    for i in range(val_sizeof_xi_i):
        t = (np.ones(shape(vec_p_i)) - vec_p_i)
        d = np.multiply(vec_p_i.todense(), t)
        dP_I_I = np.multiply(d, vec_y_i)
        inv_dD_I = -1 * sum(dP_I_I) / (val_sum_col_vec_p ** 2)
        drv_vec_p_i_drv_vec_xi_i[:, i] = dP_I_I / val_sum_col_vec_p + vec_p_i * inv_dD_I

        drv_vec_p_u_drv_vec_xi_i[:, i] = vec_q_u * inv_dD_I

    #
    # the gradients of vector p by vector Theta => Eq.(23)-(25)
    #
    mat_drv_vec_p_u_drv_vec_theta_uu = np.ones((val_user_num, val_feature_num_uu)) / val_user_num
    mat_drv_vec_p_u_drv_vec_theta_ui = np.ones((val_user_num, val_feature_num_ui)) / val_user_num
    mat_drv_vec_p_u_drv_vec_theta_ut = np.ones((val_user_num, val_feature_num_ut)) / val_user_num
    mat_drv_vec_p_u_drv_vec_theta_iu = np.ones((val_user_num, val_feature_num_ui)) / val_user_num
    mat_drv_vec_p_u_drv_vec_theta_ii = np.ones((val_user_num, val_feature_num_ii)) / val_user_num
    mat_drv_vec_p_u_drv_vec_theta_it = np.ones((val_user_num, val_feature_num_it)) / val_user_num
    mat_drv_vec_p_u_drv_vec_theta_tu = np.ones((val_user_num, val_feature_num_ut)) / val_user_num
    mat_drv_vec_p_u_drv_vec_theta_ti = np.ones((val_user_num, val_feature_num_it)) / val_user_num
    mat_drv_vec_p_u_drv_vec_theta_tt = np.ones((val_user_num, val_feature_num_tt)) / val_user_num

    mat_drv_vec_p_i_drv_vec_theta_uu = np.ones((val_item_num, val_feature_num_uu)) / val_item_num
    mat_drv_vec_p_i_drv_vec_theta_ui = np.ones((val_item_num, val_feature_num_ui)) / val_item_num
    mat_drv_vec_p_i_drv_vec_theta_ut = np.ones((val_item_num, val_feature_num_ut)) / val_item_num
    mat_drv_vec_p_i_drv_vec_theta_iu = np.ones((val_item_num, val_feature_num_ui)) / val_item_num
    mat_drv_vec_p_i_drv_vec_theta_ii = np.ones((val_item_num, val_feature_num_ii)) / val_item_num
    mat_drv_vec_p_i_drv_vec_theta_it = np.ones((val_item_num, val_feature_num_it)) / val_item_num
    mat_drv_vec_p_i_drv_vec_theta_tu = np.ones((val_item_num, val_feature_num_ut)) / val_item_num
    mat_drv_vec_p_i_drv_vec_theta_ti = np.ones((val_item_num, val_feature_num_it)) / val_item_num
    mat_drv_vec_p_i_drv_vec_theta_tt = np.ones((val_item_num, val_feature_num_tt)) / val_item_num

    mat_drv_vec_p_t_drv_vec_theta_uu = np.ones((val_tag_num, val_feature_num_uu)) / val_item_num
    mat_drv_vec_p_t_drv_vec_theta_ui = np.ones((val_tag_num, val_feature_num_ui)) / val_item_num
    mat_drv_vec_p_t_drv_vec_theta_ut = np.ones((val_tag_num, val_feature_num_ut)) / val_item_num
    mat_drv_vec_p_t_drv_vec_theta_iu = np.ones((val_tag_num, val_feature_num_ui)) / val_item_num
    mat_drv_vec_p_t_drv_vec_theta_ii = np.ones((val_tag_num, val_feature_num_ii)) / val_item_num
    mat_drv_vec_p_t_drv_vec_theta_it = np.ones((val_tag_num, val_feature_num_it)) / val_item_num
    mat_drv_vec_p_t_drv_vec_theta_tu = np.ones((val_tag_num, val_feature_num_ut)) / val_item_num
    mat_drv_vec_p_t_drv_vec_theta_ti = np.ones((val_tag_num, val_feature_num_it)) / val_item_num
    mat_drv_vec_p_t_drv_vec_theta_tt = np.ones((val_tag_num, val_feature_num_tt)) / val_item_num

    #
    # the gradients vector p by vector Theta => Eq.(23)-(25)
    #
    mat_drv_vec_p_u_drv_vec_xi_u = csr_matrix(np.ones((val_user_num, val_sizeof_xi_u)) / val_user_num,
                                              shape=(val_user_num, val_sizeof_xi_u))
    mat_drv_vec_p_u_drv_vec_xi_i = csr_matrix(np.ones((val_user_num, val_sizeof_xi_i)) / val_user_num,
                                              shape=(val_user_num, val_sizeof_xi_i))
    mat_drv_vec_p_i_drv_vec_xi_u = csr_matrix(np.ones((val_item_num, val_sizeof_xi_u)) / val_item_num,
                                              shape=(val_item_num, val_sizeof_xi_u))
    mat_drv_vec_p_i_drv_vec_xi_i = csr_matrix(np.ones((val_item_num, val_sizeof_xi_i)) / val_item_num,
                                              shape=(val_item_num, val_sizeof_xi_i))
    mat_drv_vec_p_t_drv_vec_xi_u = csr_matrix(np.ones((val_tag_num, val_sizeof_xi_u)) / val_tag_num,
                                              shape=(val_tag_num, val_sizeof_xi_u))
    mat_drv_vec_p_t_drv_vec_xi_i = csr_matrix(np.ones((val_tag_num, val_sizeof_xi_i)) / val_tag_num,
                                              shape=(val_tag_num, val_sizeof_xi_i))

    #
    # calculate the derivatives of Xi by markovian process
    #
    for itr in range(iter_num):
        for i in range(val_sizeof_xi_u):
            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_xi_u[:, i]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_xi_u[:, i]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_xi_u[:, i]
            mat_drv_vec_p_u_drv_vec_xi_u[:, i] = (1 - val_alpha) * tmp + val_alpha * drv_vec_p_u_drv_vec_xi_u[:, i]

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_xi_u[:, i]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_xi_u[:, i]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_xi_u[:, i]
            mat_drv_vec_p_i_drv_vec_xi_u[:, i] = (1 - val_alpha) * tmp + val_alpha * drv_vec_p_i_drv_vec_xi_u[:, i]

            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_xi_u[:, i]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_xi_u[:, i]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_xi_u[:, i]
            mat_drv_vec_p_t_drv_vec_xi_u[:, i] = (1 - val_alpha) * tmp

        for i in range(val_sizeof_xi_i):
            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_xi_i[:, i]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_xi_i[:, i]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_xi_i[:, i]
            mat_drv_vec_p_u_drv_vec_xi_i[:, i] = (1 - val_alpha) * tmp + val_alpha * drv_vec_p_u_drv_vec_xi_i[:, i]

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_xi_i[:, i]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_xi_i[:, i]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_xi_i[:, i]
            mat_drv_vec_p_i_drv_vec_xi_i[:, i] = (1 - val_alpha) * tmp + val_alpha * drv_vec_p_i_drv_vec_xi_i[:, i]

            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_xi_i[:, i]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_xi_i[:, i]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_xi_i[:, i]
            mat_drv_vec_p_t_drv_vec_xi_i[:, i] = (1 - val_alpha) * tmp

    #
    # calculate the derivatives of Theta by markovian process
    #
    for r in range(iter_num):

        for i in range(val_feature_num_uu):
            vecPU_UUt = csr_matrix(mat_drv_vec_p_u_drv_vec_theta_uu[:, i]).transpose()
            vecPI_UUt = csr_matrix(mat_drv_vec_p_i_drv_vec_theta_uu[:, i]).transpose()
            vecPT_UUt = csr_matrix(mat_drv_vec_p_t_drv_vec_theta_uu[:, i]).transpose()

            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_theta_uu[:, i]
            tmp += (drv_mat_a_uu_drv_vec_theta_uu[i] * vec_p_u).toarray()[:, 0]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_theta_uu[:, i]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_theta_uu[:, i]
            mat_drv_vec_p_u_drv_vec_theta_uu[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_theta_uu[:, i]
            tmp += (drv_mat_a_iu_drv_vec_theta_uu[i] * vec_p_u).toarray()[:, 0]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_theta_uu[:, i]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_theta_uu[:, i]
            mat_drv_vec_p_i_drv_vec_theta_uu[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_theta_uu[:, i]
            tmp += (drv_mat_a_tu_drv_vec_theta_uu[i] * vec_p_u).toarray()[:, 0]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_theta_uu[:, i]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_theta_uu[:, i]
            mat_drv_vec_p_t_drv_vec_theta_uu[:, i] = (1 - val_alpha) * tmp

        for i in range(val_feature_num_ui):
            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ui[:, i]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_theta_ui[:, i]
            tmp += (drv_mat_a_ui_drv_vec_theta_ui[i] * vec_p_i).toarray()[:, 0]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_theta_ui[:, i]
            mat_drv_vec_p_u_drv_vec_theta_ui[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ui[:, i]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_theta_ui[:, i]
            tmp += (drv_mat_a_ii_drv_vec_theta_ui[i] * vec_p_i).toarray()[:, 0]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_theta_ui[:, i]
            mat_drv_vec_p_i_drv_vec_theta_ui[:, i] = (1 - val_alpha) * tmp

            t = (drv_mat_a_ti_drv_vec_theta_ui[i] * vec_p_i).toarray()
            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ui[:, i]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_theta_ui[:, i]
            tmp += (drv_mat_a_ti_drv_vec_theta_ui[i] * vec_p_i).toarray()[:, 0]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_theta_ui[:, i]
            mat_drv_vec_p_t_drv_vec_theta_ui[:, i] = (1 - val_alpha) * tmp

        for i in range(val_feature_num_ut):
            vecPU_UTt = csr_matrix(mat_drv_vec_p_u_drv_vec_theta_ut[:, i]).transpose()
            vecPI_UTt = csr_matrix(mat_drv_vec_p_i_drv_vec_theta_ut[:, i]).transpose()
            vecPT_UTt = csr_matrix(mat_drv_vec_p_t_drv_vec_theta_ut[:, i]).transpose()

            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ut[:, i]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_theta_ut[:, i]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_theta_ut[:, i]
            tmp += (drv_mat_a_ut_drv_vec_theta_ut[i] * vec_p_t).toarray()[:, 0]
            mat_drv_vec_p_u_drv_vec_theta_ut[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ut[:, i]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_theta_ut[:, i]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_theta_ut[:, i]
            tmp += (drv_mat_a_it_drv_vec_theta_ut[i] * vec_p_t).toarray()[:, 0]
            mat_drv_vec_p_i_drv_vec_theta_ut[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ut[:, i]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_theta_ut[:, i]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_theta_ut[:, i]
            tmp += (drv_mat_a_tt_drv_vec_theta_ut[i] * vec_p_t).toarray()[:, 0]
            mat_drv_vec_p_t_drv_vec_theta_ut[:, i] = (1 - val_alpha) * tmp

        for i in range(val_feature_num_ui):
            vecPU_IUt = csr_matrix(mat_drv_vec_p_u_drv_vec_theta_iu[:, i]).transpose()
            vecPI_IUt = csr_matrix(mat_drv_vec_p_i_drv_vec_theta_iu[:, i]).transpose()
            vecPT_IUt = csr_matrix(mat_drv_vec_p_t_drv_vec_theta_iu[:, i]).transpose()

            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_theta_iu[:, i]
            tmp += (drv_mat_a_uu_drv_vec_theta_iu[i] * vec_p_u).toarray()[:, 0]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_theta_iu[:, i]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_theta_iu[:, i]
            mat_drv_vec_p_u_drv_vec_theta_iu[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_theta_iu[:, i]
            tmp += (drv_mat_a_iu_drv_vec_theta_iu[i] * vec_p_u).toarray()[:, 0]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_theta_iu[:, i]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_theta_iu[:, i]
            mat_drv_vec_p_i_drv_vec_theta_iu[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_theta_iu[:, i]
            tmp += (drv_mat_a_tu_drv_vec_theta_iu[i] * vec_p_u).toarray()[:, 0]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_theta_iu[:, i]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_theta_iu[:, i]
            mat_drv_vec_p_t_drv_vec_theta_iu[:, i] = (1 - val_alpha) * tmp

        for i in range(val_feature_num_ii):
            vecPU_IIt = csr_matrix(mat_drv_vec_p_u_drv_vec_theta_ii[:, i]).transpose()
            vecPI_IIt = csr_matrix(mat_drv_vec_p_i_drv_vec_theta_ii[:, i]).transpose()
            vecPT_IIt = csr_matrix(mat_drv_vec_p_t_drv_vec_theta_ii[:, i]).transpose()

            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ii[:, i]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_theta_ii[:, i]
            tmp += (drv_mat_a_ui_drv_vec_theta_ii[i] * vec_p_i).toarray()[:, 0]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_theta_ii[:, i]
            mat_drv_vec_p_u_drv_vec_theta_ii[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ii[:, i]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_theta_ii[:, i]
            tmp += (drv_mat_a_ii_drv_vec_theta_ii[i] * vec_p_i).toarray()[:, 0]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_theta_ii[:, i]
            mat_drv_vec_p_i_drv_vec_theta_ii[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ii[:, i]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_theta_ii[:, i]
            tmp += (drv_mat_a_ti_drv_vec_theta_ii[i] * vec_p_i).toarray()[:, 0]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_theta_ii[:, i]
            mat_drv_vec_p_t_drv_vec_theta_ii[:, i] = (1 - val_alpha) * tmp

        for i in range(val_feature_num_it):
            vecPU_ITt = csr_matrix(mat_drv_vec_p_u_drv_vec_theta_it[:, i]).transpose()
            vecPI_ITt = csr_matrix(mat_drv_vec_p_i_drv_vec_theta_it[:, i]).transpose()
            vecPT_ITt = csr_matrix(mat_drv_vec_p_t_drv_vec_theta_it[:, i]).transpose()

            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_theta_it[:, i]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_theta_it[:, i]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_theta_it[:, i]
            tmp += (drv_mat_a_ut_drv_vec_theta_it[i] * vec_p_t).toarray()[:, 0]
            mat_drv_vec_p_u_drv_vec_theta_it[:, i] = (1 - val_alpha) * ()

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_theta_it[:, i]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_theta_it[:, i]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_theta_it[:, i]
            tmp += (drv_mat_a_it_drv_vec_theta_it[i] * vec_p_t).toarray()[:, 0]
            mat_drv_vec_p_i_drv_vec_theta_it[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_theta_it[:, i]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_theta_it[:, i]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_theta_it[:, i]
            tmp += (drv_mat_a_tt_drv_vec_theta_it[i] * vec_p_t).toarray()[:, 0]
            mat_drv_vec_p_t_drv_vec_theta_it[:, i] = (1 - val_alpha) * ()

        for i in range(val_feature_num_ut):
            vecPU_TUt = csr_matrix(mat_drv_vec_p_u_drv_vec_theta_tu[:, i]).transpose()
            vecPI_TUt = csr_matrix(mat_drv_vec_p_i_drv_vec_theta_tu[:, i]).transpose()
            vecPT_TUt = csr_matrix(mat_drv_vec_p_t_drv_vec_theta_tu[:, i]).transpose()

            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_theta_tu[:, i]
            tmp += (drv_mat_a_uu_drv_vec_theta_tu[i] * vec_p_u).toarray()[:, 0]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_theta_tu[:, i]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_theta_tu[:, i]
            mat_drv_vec_p_u_drv_vec_theta_tu[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_theta_tu[:, i]
            tmp += (drv_mat_a_iu_drv_vec_theta_tu[i] * vec_p_u).toarray()[:, 0]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_theta_tu[:, i]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_theta_tu[:, i]
            mat_drv_vec_p_i_drv_vec_theta_tu[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_theta_tu[:, i]
            tmp += (drv_mat_a_tu_drv_vec_theta_tu[i] * vec_p_u).toarray()[:, 0]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_theta_tu[:, i]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_theta_tu[:, i]
            mat_drv_vec_p_t_drv_vec_theta_tu[:, i] = (1 - val_alpha) * tmp

        for i in range(val_feature_num_it):
            vecPU_TIt = csr_matrix(mat_drv_vec_p_u_drv_vec_theta_ti[:, i]).transpose()
            vecPI_TIt = csr_matrix(mat_drv_vec_p_i_drv_vec_theta_ti[:, i]).transpose()
            vecPT_TIt = csr_matrix(mat_drv_vec_p_t_drv_vec_theta_ti[:, i]).transpose()

            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ti[:, i]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_theta_ti[:, i]
            tmp += (drv_mat_a_ui_drv_vec_theta_ti[i] * vec_p_i).toarray()[:, 0]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_theta_ti[:, i]
            mat_drv_vec_p_u_drv_vec_theta_ti[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ti[:, i]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_theta_ti[:, i]
            tmp += (drv_mat_a_ii_drv_vec_theta_ti[i] * vec_p_i).toarray()[:, 0]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_theta_ti[:, i]
            mat_drv_vec_p_i_drv_vec_theta_ti[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_theta_ti[:, i]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_theta_ti[:, i]
            tmp += (drv_mat_a_ti_drv_vec_theta_ti[i] * vec_p_i).toarray()[:, 0]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_theta_ti[:, i]
            mat_drv_vec_p_t_drv_vec_theta_ti[:, i] = (1 - val_alpha) * tmp

        for i in range(val_feature_num_tt):
            vecPU_TTt = csr_matrix(mat_drv_vec_p_u_drv_vec_theta_tt[:, i]).transpose()
            vecPI_TTt = csr_matrix(mat_drv_vec_p_i_drv_vec_theta_tt[:, i]).transpose()
            vecPT_TTt = csr_matrix(mat_drv_vec_p_t_drv_vec_theta_tt[:, i]).transpose()

            tmp = mat_a_uu_col_norm * mat_drv_vec_p_u_drv_vec_theta_tt[:, i]
            tmp += mat_a_ui_col_norm * mat_drv_vec_p_i_drv_vec_theta_tt[:, i]
            tmp += mat_a_ut_col_norm * mat_drv_vec_p_t_drv_vec_theta_tt[:, i]
            tmp += (drv_mat_a_ut_drv_vec_theta_tt[i] * vec_p_t).toarray()[:, 0]
            mat_drv_vec_p_u_drv_vec_theta_tt[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_iu_col_norm * mat_drv_vec_p_u_drv_vec_theta_tt[:, i]
            tmp += mat_a_ii_col_norm * mat_drv_vec_p_i_drv_vec_theta_tt[:, i]
            tmp += mat_a_it_col_norm * mat_drv_vec_p_t_drv_vec_theta_tt[:, i]
            tmp += (drv_mat_a_it_drv_vec_theta_tt[i] * vec_p_t).toarray()[:, 0]
            mat_drv_vec_p_i_drv_vec_theta_tt[:, i] = (1 - val_alpha) * tmp

            tmp = mat_a_tu_col_norm * mat_drv_vec_p_u_drv_vec_theta_tt[:, i]
            tmp += mat_a_ti_col_norm * mat_drv_vec_p_i_drv_vec_theta_tt[:, i]
            tmp += mat_a_tt_col_norm * mat_drv_vec_p_t_drv_vec_theta_tt[:, i]
            tmp += (drv_mat_a_tt_drv_vec_theta_tt[i] * vec_p_t).toarray()[:, 0]
            mat_drv_vec_p_t_drv_vec_theta_tt[:, i] = (1 - val_alpha) * tmp

    vec_drv_val_j_drv_vec_theta_uu = 0
    vec_drv_val_j_drv_vec_theta_ui = 0
    vec_drv_val_j_drv_vec_theta_ut = 0
    vec_drv_val_j_drv_vec_theta_iu = 0
    vec_drv_val_j_drv_vec_theta_ii = 0
    vec_drv_val_j_drv_vec_theta_it = 0
    vec_drv_val_j_drv_vec_theta_tu = 0
    vec_drv_val_j_drv_vec_theta_ti = 0
    vec_drv_val_j_drv_vec_theta_tt = 0

    vec_drv_val_j_drv_vec_xi_u = 0
    vec_drv_val_j_drv_vec_xi_i = 0

    J = 0

    for p_ins in list_positive_tags:
        for n_ins in list_negative_tags:
            t = vec_q_t[n_ins, 0] - vec_q_t[p_ins, 0]
            pp = (-1) * val_beta * t
            r = math.exp(pp)
            s = 1 / (1 + r)

            J += s

            s_Theta_UU = mat_drv_vec_p_t_drv_vec_theta_uu[n_ins, :] - mat_drv_vec_p_t_drv_vec_theta_uu[p_ins, :]
            s_Theta_UI = mat_drv_vec_p_t_drv_vec_theta_ui[n_ins, :] - mat_drv_vec_p_t_drv_vec_theta_ui[p_ins, :]
            s_Theta_UT = mat_drv_vec_p_t_drv_vec_theta_ut[n_ins, :] - mat_drv_vec_p_t_drv_vec_theta_ut[p_ins, :]
            s_Theta_IU = mat_drv_vec_p_t_drv_vec_theta_iu[n_ins, :] - mat_drv_vec_p_t_drv_vec_theta_iu[p_ins, :]
            s_Theta_II = mat_drv_vec_p_t_drv_vec_theta_ii[n_ins, :] - mat_drv_vec_p_t_drv_vec_theta_ii[p_ins, :]
            s_Theta_IT = mat_drv_vec_p_t_drv_vec_theta_it[n_ins, :] - mat_drv_vec_p_t_drv_vec_theta_it[p_ins, :]
            s_Theta_TU = mat_drv_vec_p_t_drv_vec_theta_tu[n_ins, :] - mat_drv_vec_p_t_drv_vec_theta_tu[p_ins, :]
            s_Theta_TI = mat_drv_vec_p_t_drv_vec_theta_ti[n_ins, :] - mat_drv_vec_p_t_drv_vec_theta_ti[p_ins, :]
            s_Theta_TT = mat_drv_vec_p_t_drv_vec_theta_tt[n_ins, :] - mat_drv_vec_p_t_drv_vec_theta_tt[p_ins, :]

            vec_drv_val_j_drv_vec_theta_uu += val_beta * s * (1 - s) * s_Theta_UU
            vec_drv_val_j_drv_vec_theta_ui += val_beta * s * (1 - s) * s_Theta_UI
            vec_drv_val_j_drv_vec_theta_ut += val_beta * s * (1 - s) * s_Theta_UT
            vec_drv_val_j_drv_vec_theta_iu += val_beta * s * (1 - s) * s_Theta_IU
            vec_drv_val_j_drv_vec_theta_ii += val_beta * s * (1 - s) * s_Theta_II
            vec_drv_val_j_drv_vec_theta_it += val_beta * s * (1 - s) * s_Theta_IT
            vec_drv_val_j_drv_vec_theta_tu += val_beta * s * (1 - s) * s_Theta_TU
            vec_drv_val_j_drv_vec_theta_ti += val_beta * s * (1 - s) * s_Theta_TI
            vec_drv_val_j_drv_vec_theta_tt += val_beta * s * (1 - s) * s_Theta_TT

            s_Xi_U = mat_drv_vec_p_t_drv_vec_xi_u[n_ins, :] - mat_drv_vec_p_t_drv_vec_xi_u[p_ins, :]
            s_Xi_I = mat_drv_vec_p_t_drv_vec_xi_i[n_ins, :] - mat_drv_vec_p_t_drv_vec_xi_i[p_ins, :]

            vec_drv_val_j_drv_vec_xi_u += val_beta * s * (1 - s) * s_Xi_U
            vec_drv_val_j_drv_vec_xi_i += val_beta * s * (1 - s) * s_Xi_I

    J /= val_sizeof_positive_tags * val_sizeof_negative_tags

    vec_drv_val_j_drv_vec_theta_uu /= val_sizeof_positive_tags * val_sizeof_negative_tags
    vec_drv_val_j_drv_vec_theta_ui /= val_sizeof_positive_tags * val_sizeof_negative_tags
    vec_drv_val_j_drv_vec_theta_ut /= val_sizeof_positive_tags * val_sizeof_negative_tags
    vec_drv_val_j_drv_vec_theta_iu /= val_sizeof_positive_tags * val_sizeof_negative_tags
    vec_drv_val_j_drv_vec_theta_ii /= val_sizeof_positive_tags * val_sizeof_negative_tags
    vec_drv_val_j_drv_vec_theta_it /= val_sizeof_positive_tags * val_sizeof_negative_tags
    vec_drv_val_j_drv_vec_theta_it /= val_sizeof_positive_tags * val_sizeof_negative_tags
    vec_drv_val_j_drv_vec_theta_ti /= val_sizeof_positive_tags * val_sizeof_negative_tags
    vec_drv_val_j_drv_vec_theta_tt /= val_sizeof_positive_tags * val_sizeof_negative_tags

    vec_theta_uu -= val_learning_rate * vec_drv_val_j_drv_vec_theta_uu
    vec_theta_ui -= val_learning_rate * vec_drv_val_j_drv_vec_theta_ui
    vec_theta_ut -= val_learning_rate * vec_drv_val_j_drv_vec_theta_ut
    vec_theta_iu -= val_learning_rate * vec_drv_val_j_drv_vec_theta_iu
    vec_theta_ii -= val_learning_rate * vec_drv_val_j_drv_vec_theta_ii
    vec_theta_it -= val_learning_rate * vec_drv_val_j_drv_vec_theta_it
    vec_theta_tu -= val_learning_rate * vec_drv_val_j_drv_vec_theta_tu
    vec_theta_ti -= val_learning_rate * vec_drv_val_j_drv_vec_theta_ti
    vec_theta_tt -= val_learning_rate * vec_drv_val_j_drv_vec_theta_tt

    vec_drv_val_j_drv_vec_xi_u /= val_sizeof_positive_tags * val_sizeof_negative_tags
    vec_drv_val_j_drv_vec_xi_i /= val_sizeof_positive_tags * val_sizeof_negative_tags

    vec_xi_u -= val_learning_rate * vec_drv_val_j_drv_vec_xi_u
    vec_xi_i -= val_learning_rate * vec_drv_val_j_drv_vec_xi_i

    val_diff_theta = 0

    if len(vec_drv_val_j_drv_vec_theta_uu) > 0:
        val_diff_theta += transpose(vec_drv_val_j_drv_vec_theta_uu) * vec_drv_val_j_drv_vec_theta_uu
        
    if len(vec_drv_val_j_drv_vec_theta_ui) > 0:
        val_diff_theta += transpose(vec_drv_val_j_drv_vec_theta_ui) * vec_drv_val_j_drv_vec_theta_ui
        
    if len(vec_drv_val_j_drv_vec_theta_ut) > 0:
        val_diff_theta += transpose(vec_drv_val_j_drv_vec_theta_ut) * vec_drv_val_j_drv_vec_theta_ut
        
    if len(vec_drv_val_j_drv_vec_theta_iu) > 0:
        val_diff_theta += transpose(vec_drv_val_j_drv_vec_theta_iu) * vec_drv_val_j_drv_vec_theta_iu
        
    if len(vec_drv_val_j_drv_vec_theta_ii) > 0:
        val_diff_theta += transpose(vec_drv_val_j_drv_vec_theta_ii) * vec_drv_val_j_drv_vec_theta_ii
        
    if len(vec_drv_val_j_drv_vec_theta_it) > 0:
        val_diff_theta += transpose(vec_drv_val_j_drv_vec_theta_it) * vec_drv_val_j_drv_vec_theta_it
        
    if len(vec_drv_val_j_drv_vec_theta_tu) > 0:
        val_diff_theta += transpose(vec_drv_val_j_drv_vec_theta_tu) * vec_drv_val_j_drv_vec_theta_tu
        
    if len(vec_drv_val_j_drv_vec_theta_ti) > 0:
        val_diff_theta += transpose(vec_drv_val_j_drv_vec_theta_ti) * vec_drv_val_j_drv_vec_theta_ti
        
    if len(vec_drv_val_j_drv_vec_theta_tt) > 0:
        val_diff_theta += transpose(vec_drv_val_j_drv_vec_theta_tt) * vec_drv_val_j_drv_vec_theta_tt

    val_diff_theta = val_diff_theta

    val_diff_xi = transpose(vec_drv_val_j_drv_vec_xi_u) * vec_drv_val_j_drv_vec_xi_u
    val_diff_xi += transpose(vec_drv_val_j_drv_vec_xi_u) * vec_drv_val_j_drv_vec_xi_u
    val_diff_xi = val_diff_xi[0, 0]

    print('J = ' + str(J))
    print('val_diff_xi      = ' + str(val_diff_xi))
    print('vec_xi_u     = ' + str(vec_xi_u)     + '   vec_xi_i     = ' + str(vec_xi_i))
    print('vec_drv_val_j_drv_vec_xi_u      = ' + str(vec_drv_val_j_drv_vec_xi_u[0, 0]) +
          '   vec_drv_val_j_drv_vec_xi_i      = ' + str(vec_drv_val_j_drv_vec_xi_i[0, 0]))
    print('val_diff_theta   = ' + str(val_diff_theta))
    print('vec_theta_ui = ' + str(vec_theta_ui) + '   vec_theta_ut = ' + str(vec_theta_ut))

    '''
    if J < valConvergeThreshold_J:
        isConverge = True
    '''
    '''
    if val_diff_theta < 10e-40 and val_diff_xi < 10e-40:
        isConverge = True
    '''

    if line_count == 2000:
        isConverge = True

print('\nvec_theta_uu = ' + str(vec_theta_uu))
print('vec_theta_ui = ' + str(vec_theta_ui))
print('vec_theta_ut = ' + str(vec_theta_ut))
print('vec_theta_iu = ' + str(vec_theta_iu))
print('vec_theta_ii = ' + str(vec_theta_ii))
print('vec_theta_it = ' + str(vec_theta_it))
print('vec_theta_tu = ' + str(vec_theta_tu))
print('vec_theta_ti = ' + str(vec_theta_ti))
print('vec_theta_tt = ' + str(vec_theta_tt))

print('vec_xi_u = ' + str(vec_xi_u[0]))
print('vec_xi_i = ' + str(vec_xi_i[0]))
