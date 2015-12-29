import numpy as np
from scipy import *
from scipy.sparse import *

import ReadFile
import LA


__author__ = 'IanKuo'


val_alpha = 0.8
iter_num = 20
val_beta = 10000.
val_lr = 10.

topK = 20

str_uit_graph_path = 'LastFm/UIT_GroundTruth.txt'
str_uu_graph_path = ''
str_ii_graph_path = ''
str_tt_graph_path = ''
str_test_instance_path = 'LastFm/UIT_Train.txt'

vec_theta_uu = []
vec_theta_ui = [-0.1738571]
vec_theta_ut = [1.]
vec_theta_iu = [6030.06074788]
vec_theta_ii = []
vec_theta_it = [1.]
vec_theta_tu = [-5149459.83916038]
vec_theta_ti = [933.83925912]
vec_theta_tt = []

val_feature_num_uu = len(vec_theta_uu)
val_feature_num_ui = len(vec_theta_ui)
val_feature_num_ut = len(vec_theta_ut)
val_feature_num_iu = len(vec_theta_iu)
val_feature_num_ii = len(vec_theta_ii)
val_feature_num_it = len(vec_theta_it)
val_feature_num_tu = len(vec_theta_tu)
val_feature_num_ti = len(vec_theta_ti)
val_feature_num_tt = len(vec_theta_tt)

vec_xi_u = [-20.10886058]     # [-6.12912391]
vec_xi_i = [-354.34245751]    # [-22.7446508]

#
# dictionary of positive/negative instances
# (key: (vertex, vertex, vertex), value: [vertex])
#
list_test_instance = list()


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

# read positive/negative instances
ReadFile.read_uit_file(str_uit_graph_path, tensor_x_ui, tensor_x_ut, tensor_x_it)

for i in tensor_x_ui:
    tensor_x_iu.append(i.transpose())

for i in tensor_x_ut:
    tensor_x_tu.append(i.transpose())

for i in tensor_x_it:
    tensor_x_ti.append(i.transpose())

print('\n ----- Reading the UIT graph completes -----')
print(shape(tensor_x_ui[0]))
print(shape(tensor_x_ut[0]))
print(shape(tensor_x_it[0]))

valUserNum = shape(tensor_x_ui[0])[0]
valItemNum = shape(tensor_x_ui[0])[1]
valTagNum = shape(tensor_x_ut[0])[1]

#
# read training instances
#
#  user \t item \t posIns1,posIns2,... \t negIns1,negIns2,...
#
f_TestInstances = open(str_test_instance_path, 'r')
for line in f_TestInstances:
    list_test_instance.append(line)

#
# create matA
#
if len(tensor_x_uu) != 0:
    matA_UU = LA.linear_combination(valUserNum, valUserNum, tensor_x_uu, vec_theta_uu)
    matA_UU.data[:] = 1 / (1 + exp(-1 * matA_UU.data))
else:
    matA_UU = csr_matrix((valUserNum, valUserNum), dtype=float)

if len(tensor_x_ui) != 0:
    matA_UI = LA.linear_combination(valUserNum, valItemNum, tensor_x_ui, vec_theta_ui)
    matA_UI.data[:] = 1 / (1 + exp(-1 * matA_UI.data))
else:
    matA_UI = csr_matrix((valUserNum, valItemNum), dtype=float)

if len(tensor_x_ut) != 0:
    matA_UT = LA.linear_combination(valUserNum, valTagNum, tensor_x_ut, vec_theta_ut)
    matA_UT.data[:] = 1 / (1 + exp(-1 * matA_UT.data))
else:
    matA_UT = csr_matrix((valUserNum, valTagNum), dtype=float)

if len(tensor_x_ui) != 0:
    matA_IU = LA.linear_combination(valItemNum, valUserNum, tensor_x_iu, vec_theta_iu)
    matA_IU.data[:] = 1 / (1 + exp(-1 * matA_IU.data))
else:
    matA_IU = csr_matrix((valItemNum, valUserNum), dtype=float)

if len(tensor_x_ii) != 0:
    matA_II = LA.linear_combination(valItemNum, valItemNum, tensor_x_ii, vec_theta_ii)
    matA_II.data[:] = 1 / (1 + exp(-1 * matA_II.data))
else:
    matA_II = csr_matrix((valItemNum, valItemNum), dtype=float)

if len(tensor_x_it) != 0:
    matA_IT = LA.linear_combination(valItemNum, valTagNum, tensor_x_it, vec_theta_it)
    matA_IT.data[:] = 1 / (1 + exp(-1 * matA_IT.data))
else:
    matA_IT = csr_matrix((valItemNum, valTagNum), dtype=float)

if len(tensor_x_ut) != 0:
    matA_TU = LA.linear_combination(valTagNum, valUserNum, tensor_x_tu, vec_theta_tu)
    matA_TU.data[:] = 1 / (1 + exp(-1 * matA_TU.data))
else:
    matA_TU = csr_matrix((valTagNum, valUserNum), dtype=float)

if len(tensor_x_ti) != 0:
    matA_TI = LA.linear_combination(valTagNum, valItemNum, tensor_x_ti, vec_theta_ti)
    matA_TI.data[:] = 1 / (1 + exp(-1 * matA_TI.data))
else:
    matA_TI = csr_matrix((valTagNum, valItemNum), dtype=float)

if len(tensor_x_tt) != 0:
    matA_TT = LA.linear_combination(valTagNum, valTagNum, tensor_x_tt, vec_theta_tt)
    matA_TT.data[:] = 1 / (1 + exp(-1 * matA_TT.data))
else:
    matA_TT = csr_matrix((valTagNum, valTagNum), dtype=float)

print('matA_UU : ' + str(shape(matA_UU)))
print('matA_UI : ' + str(shape(matA_UI)))
print('matA_UT : ' + str(shape(matA_UT)))
print('matA_IU : ' + str(shape(matA_IU)))
print('matA_II : ' + str(shape(matA_II)))
print('matA_IT : ' + str(shape(matA_IT)))
print('matA_TU : ' + str(shape(matA_TU)))
print('matA_TI : ' + str(shape(matA_TI)))
print('matA_TT : ' + str(shape(matA_TT)))

#
# D_U^{-1}
#
colSum_U = csr_matrix(np.ones((1, valUserNum)) * matA_UU + np.ones((1, valItemNum)) * matA_IU + np.ones((1, valTagNum)) * matA_TU)
colSum_I = csr_matrix(np.ones((1, valUserNum)) * matA_UI + np.ones((1, valItemNum)) * matA_II + np.ones((1, valTagNum)) * matA_TI)
colSum_T = csr_matrix(np.ones((1, valUserNum)) * matA_UT + np.ones((1, valItemNum)) * matA_IT + np.ones((1, valTagNum)) * matA_TT)

colSumInv_U = csr_matrix(np.ones((1, valUserNum)) * matA_UU + np.ones((1, valItemNum)) * matA_IU + np.ones((1, valTagNum)) * matA_TU)
colSumInv_I = csr_matrix(np.ones((1, valUserNum)) * matA_UI + np.ones((1, valItemNum)) * matA_II + np.ones((1, valTagNum)) * matA_TI)
colSumInv_T = csr_matrix(np.ones((1, valUserNum)) * matA_UT + np.ones((1, valItemNum)) * matA_IT + np.ones((1, valTagNum)) * matA_TT)

colSumInv_U.data[:] = power(colSumInv_U.data, -1)
inv_D_U = dia_matrix((colSumInv_U.toarray(), array([0])), shape=(valUserNum, valUserNum))

colSumInv_I.data[:] = power(colSumInv_I.data, -1)
inv_D_I = dia_matrix((colSumInv_I.toarray(), array([0])), shape=(valItemNum, valItemNum))

colSumInv_T.data[:] = power(colSumInv_T.data, -1)
inv_D_T = dia_matrix((colSumInv_T.toarray(), array([0])), shape=(valTagNum, valTagNum))

#
# column normalization on transition probabilities matrices
#
matA_UU_CN = matA_UU * inv_D_U
matA_UI_CN = matA_UI * inv_D_I
matA_UT_CN = matA_UT * inv_D_T
matA_IU_CN = matA_IU * inv_D_U
matA_II_CN = matA_II * inv_D_I
matA_IT_CN = matA_IT * inv_D_T
matA_TU_CN = matA_TU * inv_D_U
matA_TI_CN = matA_TI * inv_D_I
matA_TT_CN = matA_TT * inv_D_T

valAUC = 0
valPrecision20 = 0
valPrecision10 = 0
valPrecision5 = 0
valRecall20 = 0
valRecall10 = 0
valRecall5 = 0
valDCG20 = 0
valDCG10 = 0
valDCG5 = 0
instanceCount = 0

count = 0
for line in list_test_instance:

    count += 1

    if count <= 2000:
        continue
    #
    # set query vector
    #
    l = line.strip('\n').split('\t')

    if len(l) < 4:
        continue

    user = int(l[0])
    item = int(l[1])
    Ptags = list(map(int, l[2].split(',')))
    Ntags = list(map(int, l[3].split(',')))
    lengthPT = len(Ptags)


    if lengthPT == 0:
        continue

    #
    # probability distribution vector definition
    #     [ U ]
    #     [ I ]
    #     [ T ]
    #

    # Y
    vecYU = np.zeros((valUserNum, 1))
    vecYI = np.zeros((valItemNum, 1))

    # preference vector
    vecPU = np.zeros((valUserNum, 1))
    vecPI = np.zeros((valItemNum, 1))
    vecPT = np.zeros((valTagNum, 1))

    # query vector
    vecQU = np.zeros((valUserNum, 1))
    vecQI = np.zeros((valItemNum, 1))
    vecQT = np.zeros((valTagNum, 1))

    print('\nuser: ' + str(user) + ', item: ' + str(item))

    '''
    vecQU[user] = 1 / (1 + exp(1))
    vecPU[user] = 1 / (1 + exp(1))

    vecQI[item] = 1 / (1 + exp(1))
    vecPI[item] = 1 / (1 + exp(1))
    '''

    vecYU[user, 0] = 1
    vecYI[item, 0] = 1

    vecQU = vecYU * vec_xi_u
    vecQI = vecYI * vec_xi_i

    vecPU = vecYU * vec_xi_u
    vecPI = vecYI * vec_xi_i


    vecQU = 2. / (1. + np.exp(-1 * vecQU)) - ones(shape(vecQU))
    vecQI = 2. / (1. + np.exp(-1 * vecQI)) - ones(shape(vecQI))
    vecQT = 2. / (1. + np.exp(-1 * vecQT)) - ones(shape(vecQT))

    vecPU = 2. / (1. + np.exp(-1 * vecPU)) - ones(shape(vecQU))
    vecPI = 2. / (1. + np.exp(-1 * vecPI)) - ones(shape(vecQI))
    vecPT = 2. / (1. + np.exp(-1 * vecPT)) - ones(shape(vecQT))

    '''
    vecQU = 1. / (1. + np.exp(-1 * vecQU))
    vecQI = 1. / (1. + np.exp(-1 * vecQI))
    vecQT = 1. / (1. + np.exp(-1 * vecQT))

    vecPU = 1. / (1. + np.exp(-1 * vecPU))
    vecPI = 1. / (1. + np.exp(-1 * vecPI))
    vecPT = 1. / (1. + np.exp(-1 * vecPT))
    '''

    colSumQ = np.sum(vecQU) + np.sum(vecQI) + np.sum(vecQT)

    vecQU /= colSumQ
    vecQI /= colSumQ

    vecPU /= colSumQ
    vecPI /= colSumQ

    #
    # calculate the distribution by random walk with restart
    #
    for itr in range(iter_num):
        vecQU = (1 - val_alpha) * (
            matA_UU_CN * vecQU + matA_UI_CN * vecQI + matA_UT_CN * vecQT) + val_alpha * vecPU
        vecQI = (1 - val_alpha) * (
            matA_IU_CN * vecQU + matA_II_CN * vecQI + matA_IT_CN * vecQT) + val_alpha * vecPI
        vecQT = (1 - val_alpha) * (
            matA_TU_CN * vecQU + matA_TI_CN * vecQI + matA_TT_CN * vecQT) + val_alpha * vecPT


    listQQQT_20 = np.argsort(np.asarray(vecQT.transpose()), kind='heapsort')[0][::-1][0:20]
    listQQQT_10 = np.argsort(np.asarray(vecQT.transpose()), kind='heapsort')[0][::-1][0:10]
    listQQQT_5 = np.argsort(np.asarray(vecQT.transpose()), kind='heapsort')[0][::-1][0:5]

    setResults = set(listQQQT_5)
    setGroundTruth = set(Ptags)
    setRetrive = setGroundTruth & setResults

    listRR = list(listQQQT_5)

    DCG = 0
    for t in setRetrive:
        indx = listRR.index(t) + 2
        DCG += 1 / math.log2(indx)

    valPrecision5 += len(setRetrive) / 5
    valRecall5 += len(setRetrive) / len(setGroundTruth)
    valDCG5 += DCG

    print('Precision@5 = ' + str(len(setRetrive) / 5) + ', ' + str(len(setRetrive) / len(setGroundTruth)))

    setResults = set(listQQQT_10)
    setGroundTruth = set(Ptags)
    setRetrive = setGroundTruth & setResults

    listRR = list(listQQQT_10)

    DCG = 0
    for t in setRetrive:
        indx = listRR.index(t) + 2
        DCG += 1 / math.log2(indx)

    valPrecision10 += len(setRetrive) / 10
    valRecall10 += len(setRetrive) / len(setGroundTruth)
    valDCG10 += DCG

    print('Precision@10 = ' + str(len(setRetrive) / 10) + ', ' + str(len(setRetrive) / len(setGroundTruth)))

    setResults = set(listQQQT_20)
    setGroundTruth = set(Ptags)
    setRetrive = setGroundTruth & setResults

    listRR = list(listQQQT_20)

    DCG = 0
    for t in setRetrive:
        indx = listRR.index(t) + 2
        DCG += 1 / math.log2(indx)

    valPrecision20 += len(setRetrive) / 20
    valRecall10 += len(setRetrive) / len(setGroundTruth)
    valDCG20 += DCG
    print('Precision@20 = ' + str(len(setRetrive) / 20) + ', ' + str(len(setRetrive) / len(setGroundTruth)))

    instanceCount += 1

    J = 0

    for p_ins in Ptags:
        for n_ins in Ntags:
            if vecQT[p_ins, 0] > vecQT[n_ins, 0]:
                J += 1

    J /= len(Ptags) * len(Ntags)

    print('AUC = ' + str(J))
    valAUC += J

    if count > 5000:
        break

print('\nAvg. Precision @5 = ' + str(valPrecision5 / instanceCount))
print('Avg. DCG @5 = ' + str(valDCG5 / instanceCount))
print('Avg. Precision @10 = ' + str(valPrecision10 / instanceCount))
print('Avg. DCG @10 = ' + str(valDCG10 / instanceCount))
print('Avg. Precision @20 = ' + str(valPrecision20 / instanceCount))
print('Avg. DCG @20 = ' + str(valDCG20 / instanceCount))
print('Avg. AUC = ' + str(valAUC / instanceCount))