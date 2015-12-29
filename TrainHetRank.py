__author__ = 'iankuoli'

import numpy as np
from scipy import *
from scipy.sparse import *
from sklearn.preprocessing import normalize

import ReadFile
import LA

valAlpha = 0.8
iterNum = 20
valBeta = 10000.
valLR = 10.

valConvergeThreshold_J = 10e-120

strUITGraphPath = 'lastfm/UIT_GroundTruth.txt'
strUUGraphPath = ''
strIIGraphPath = ''
strTTGraphPath = ''
strTrainInstancePath = 'lastfm/UIT_Train.txt'

valFeatureNum_UU = 0
valFeatureNum_UI = 1
valFeatureNum_UT = 1
valFeatureNum_IU = valFeatureNum_UI
valFeatureNum_II = 0
valFeatureNum_IT = 1
valFeatureNum_TU = valFeatureNum_UT
valFeatureNum_TI = valFeatureNum_IT
valFeatureNum_TT = 0

valXiNum_U = 1
valXiNum_I = 1

valUserNum = 0
valTagNum = 0
valItemNum = 0

#
# dictionary of positive/negative instances
# (key: (vertex, vertex, vertex), value: [vertex])
#
listTrainingInstance = list()

#
# vector Theta denotes the feature weight
#

vecTheta_UU = np.ones(valFeatureNum_UU) / valFeatureNum_UU
vecTheta_UI = np.ones(valFeatureNum_UI) / valFeatureNum_UI * 0.30805685
vecTheta_UT = np.ones(valFeatureNum_UT) / valFeatureNum_UT
vecTheta_IU = np.ones(valFeatureNum_UT) / valFeatureNum_UT * 6030.06074788
vecTheta_II = np.ones(valFeatureNum_II) / valFeatureNum_II
vecTheta_IT = np.ones(valFeatureNum_IT) / valFeatureNum_IT
vecTheta_TU = np.ones(valFeatureNum_IT) / valFeatureNum_IT * -5149459.83916038
vecTheta_TI = np.ones(valFeatureNum_IT) / valFeatureNum_IT * 933.83925912
vecTheta_TT = np.ones(valFeatureNum_TT) / valFeatureNum_TT
'''
vecTheta_UU = []
vecTheta_UI = [ 0.30805685]
vecTheta_UT = [ 1.]
vecTheta_IU = [ 6030.06074788]
vecTheta_II = []
vecTheta_IT = [ 1.]
vecTheta_TU = [-5149459.83916038]
vecTheta_TI = [ 933.83925912]
vecTheta_TT = []
'''
#
# vector Xi denotes the query weight
#

vecXi_U = np.ones(valXiNum_U) / valXiNum_U * -27.5103425
vecXi_I = np.ones(valXiNum_I) / valXiNum_I * -308.54507162
'''
vecXi_U = [-27.5103425]
vecXi_I = [-308.54507162]
'''


#
# adjacent matrices definition
# [ UU  UI  UT ]
# [ IU  II  IT ]
# [ TU  TI  TT ]
#
matX_UU = list()  #valFeatureNum_UU
matX_UI = list()  #valFeatureNum_UI
matX_UT = list()  #valFeatureNum_UT
matX_IU = list()
matX_II = list()  #valFeatureNum_II
matX_IT = list()  #valFeatureNum_IT
matX_TU = list()
matX_TI = list()
matX_TT = list()  #valFeatureNum_TT

gradientA_UU_UU = list()  # partial(matA_UU)/partial(theta_UU)
gradientA_UU_IU = list()
gradientA_UU_TU = list()
gradientA_UI_UI = list()
gradientA_UI_II = list()
gradientA_UI_TI = list()
gradientA_UT_UT = list()
gradientA_UT_IT = list()
gradientA_UT_TT = list()
gradientA_IU_UU = list()
gradientA_IU_IU = list()
gradientA_IU_TU = list()
gradientA_II_UI = list()
gradientA_II_II = list()
gradientA_II_TI = list()
gradientA_IT_UT = list()
gradientA_IT_IT = list()
gradientA_IT_TT = list()
gradientA_TU_UU = list()
gradientA_TU_IU = list()
gradientA_TU_TU = list()
gradientA_TI_UI = list()
gradientA_TI_II = list()
gradientA_TI_TI = list()
gradientA_TT_UT = list()
gradientA_TT_IT = list()
gradientA_TT_TT = list()

# read positive/negative instances
ReadFile.readUITFile(strUITGraphPath, matX_UI, matX_UT, matX_IT)

for i in matX_UI:
    matX_IU.append(i.transpose())

for i in matX_UT:
    matX_TU.append(i.transpose())

for i in matX_IT:
    matX_TI.append(i.transpose())

print(shape(matX_UI[0]))
print(shape(matX_UT[0]))
print(shape(matX_IT[0]))

valUserNum = shape(matX_UI[0])[0]
valItemNum = shape(matX_UI[0])[1]
valTagNum = shape(matX_UT[0])[1]
#
# query vector definition
#     [ U ]
#     [ I ]
#     [ T ]
#

'''
# read UU graph
ReadFile.readXXFile(strUUGraphPath, matX_UU)

# read II graph
ReadFile.readXXFile(strIIGraphPath, matX_II)

# read TT graph
ReadFile.readXXFile(strTTGraphPath, matX_TT)
'''

#
# read training instances
#
#  user \t item \t posIns1,posIns2,... \t negIns1,negIns2,...
#
f_TrainInstances = open(strTrainInstancePath, 'r')
for line in f_TrainInstances:
    listTrainingInstance.append(line)


isConverge = False
line_count = 0

for line in listTrainingInstance:

    line_count += 1

    if line_count < 1000:
        continue

    if isConverge:
        break

    #
    # create matA
    #

    if len(matX_UU) != 0:
        matA_UU = LA.linear_combination(valUserNum, valUserNum, matX_UU, vecTheta_UU)
        matA_UU.data[:] = 1 / (1 + exp(-1 * matA_UU.data))
    else:
        matA_UU = csr_matrix((valUserNum, valUserNum), dtype=float)

    if len(matX_UI) != 0:
        matA_UI = LA.linear_combination(valUserNum, valItemNum, matX_UI, vecTheta_UI)
        matA_UI.data[:] = 1 / (1 + exp(-1 * matA_UI.data))
    else:
        matA_UI = csr_matrix((valUserNum, valItemNum), dtype=float)

    if len(matX_UT) != 0:
        matA_UT = LA.linear_combination(valUserNum, valTagNum, matX_UT, vecTheta_UT)
        matA_UT.data[:] = 1 / (1 + exp(-1 * matA_UT.data))
    else:
        matA_UT = csr_matrix((valUserNum, valTagNum), dtype=float)

    if len(matX_UI) != 0:
        matA_IU = LA.linear_combination(valItemNum, valUserNum, matX_IU, vecTheta_IU)
        matA_IU.data[:] = 1 / (1 + exp(-1 * matA_IU.data))
    else:
        matA_IU = csr_matrix((valItemNum, valUserNum), dtype=float)

    if len(matX_II) != 0:
        matA_II = LA.linear_combination(valItemNum, valItemNum, matX_II, vecTheta_II)
        matA_II.data[:] = 1 / (1 + exp(-1 * matA_II.data))
    else:
        matA_II = csr_matrix((valItemNum, valItemNum), dtype=float)

    if len(matX_IT) != 0:
        matA_IT = LA.linear_combination(valItemNum, valTagNum, matX_IT, vecTheta_IT)
        matA_IT.data[:] = 1 / (1 + exp(-1 * matA_IT.data))
    else:
        matA_IT = csr_matrix((valItemNum, valTagNum), dtype=float)

    if len(matX_UT) != 0:
        matA_TU = LA.linear_combination(valTagNum, valUserNum, matX_TU, vecTheta_TU)
        matA_TU.data[:] = 1 / (1 + exp(-1 * matA_TU.data))
    else:
        matA_TU = csr_matrix((valTagNum, valUserNum), dtype=float)

    if len(matX_TI) != 0:
        matA_TI = LA.linear_combination(valTagNum, valItemNum, matX_TI, vecTheta_TI)
        matA_TI.data[:] = 1 / (1 + exp(-1 * matA_TI.data))
    else:
        matA_TI = csr_matrix((valTagNum, valItemNum), dtype=float)

    if len(matX_TT) != 0:
        matA_TT = LA.linear_combination(valTagNum, valTagNum, matX_TT, vecTheta_TT)
        matA_TT.data[:] = 1 / (1 + exp(-1 * matA_TT.data))
    else:
        matA_TT = csr_matrix((valTagNum, valTagNum), dtype=float)

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

    colSum_U = colSum_U.todense()
    colSum_I = colSum_I.todense()
    colSum_T = colSum_T.todense()

    colSumInv_U = colSumInv_U.todense()
    colSumInv_I = colSumInv_I.todense()
    colSumInv_T = colSumInv_T.todense()

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

    # Theta_UU
    LA.gradient_mn(gradientA_UU_UU, gradientA_IU_UU, gradientA_TU_UU,
                   matA_UU, matA_IU, matA_TU, valUserNum, valItemNum, valTagNum, vecTheta_UU, inv_D_U, colSum_U, matX_UU, 1)

    # Theta_UI
    LA.gradient_mn(gradientA_UI_UI, gradientA_II_UI, gradientA_TI_UI,
                   matA_UI, matA_II, matA_TI, valUserNum, valItemNum, valTagNum, vecTheta_UI, inv_D_I, colSum_I, matX_UI, 1)

    # Theta_UT
    LA.gradient_mn(gradientA_UT_UT, gradientA_IT_UT, gradientA_TT_UT,
                   matA_UT, matA_IT, matA_TT, valUserNum, valItemNum, valTagNum, vecTheta_UT, inv_D_T, colSum_T, matX_UT, 1)

    # Theta_IU
    LA.gradient_mn(gradientA_UU_IU, gradientA_IU_IU, gradientA_TU_IU,
                   matA_UU, matA_IU, matA_TU, valUserNum, valItemNum, valTagNum, vecTheta_IU, inv_D_U, colSum_U, matX_UI, 2)

    # Theta_II
    LA.gradient_mn(gradientA_UI_II, gradientA_II_II, gradientA_TI_II,
                   matA_UI, matA_II, matA_TI, valUserNum, valItemNum, valTagNum, vecTheta_II, inv_D_I, colSum_I, matX_II, 2)

    # Theta_IT
    LA.gradient_mn(gradientA_UT_IT, gradientA_IT_IT, gradientA_TT_IT,
                   matA_UT, matA_IT, matA_TT, valUserNum, valItemNum, valTagNum, vecTheta_IT, inv_D_T, colSum_T, matX_IT, 2)

    # Theta_TU
    LA.gradient_mn(gradientA_UU_TU, gradientA_IU_TU, gradientA_TU_TU,
                   matA_UU, matA_IU, matA_TU, valUserNum, valItemNum, valTagNum, vecTheta_TU, inv_D_U, colSum_U, matX_UT, 3)

    # Theta_TI
    LA.gradient_mn(gradientA_UI_TI, gradientA_II_TI, gradientA_TI_TI,
                   matA_UI, matA_II, matA_TI, valUserNum, valItemNum, valTagNum, vecTheta_TI, inv_D_I, colSum_I, matX_IT, 3)

    # Theta_TT
    LA.gradient_mn(gradientA_UT_TT, gradientA_IT_TT, gradientA_TT_TT,
                   matA_UT, matA_IT, matA_TT, valUserNum, valItemNum, valTagNum, vecTheta_TT, inv_D_T, colSum_T, matX_TT, 3)


    #
    # the derivatives of the parameter: Xi
    #
    gradientP_U_U = csr_matrix((valUserNum, valXiNum_U), dtype=float)
    gradientP_U_I = csr_matrix((valUserNum, valXiNum_I), dtype=float)
    gradientP_I_U = csr_matrix((valItemNum, valXiNum_U), dtype=float)
    gradientP_I_I = csr_matrix((valItemNum, valXiNum_I), dtype=float)

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
    lengthNT = len(Ntags)

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

    if lengthPT == 0 or lengthNT == 0:
        continue

    print('\n' + str(line_count) + '\nuser: ' + str(user) + ', item: ' + str(item))

    vecYU[user, 0] = 1
    vecYI[item, 0] = 1


    vecQU = vecYU * vecXi_U
    vecQI = vecYI * vecXi_U

    vecPU = vecYU * vecXi_U
    vecPI = vecYI * vecXi_U

    vecQU = 2. / (1. + np.exp(-1 * vecQU)) - ones(shape(vecQU))
    vecQI = 2. / (1. + np.exp(-1 * vecQI)) - ones(shape(vecQI))
    vecQT = 2. / (1. + np.exp(-1 * vecQT)) - ones(shape(vecQT))

    vecPU = 2. / (1. + np.exp(-1 * vecPU)) - ones(shape(vecQU))
    vecPI = 2. / (1. + np.exp(-1 * vecPI)) - ones(shape(vecQI))
    vecPT = 2. / (1. + np.exp(-1 * vecPT)) - ones(shape(vecQT))

    colSumP = np.sum(vecPU) + np.sum(vecPI) + np.sum(vecPT)

    vecQU /= colSumP
    vecQI /= colSumP
    vecQT /= colSumP

    vecPU /= colSumP
    vecPI /= colSumP
    vecPT /= colSumP

    vecPU = csr_matrix(vecPU, shape=(valUserNum, 1))
    vecPI = csr_matrix(vecPI, shape=(valItemNum, 1))
    vecPT = csr_matrix(vecPT, shape=(valTagNum, 1))

    #
    # calculate the distribution by random walk with restart
    #
    for itr in range(iterNum):
        vecQU = (1 - valAlpha) * (
            matA_UU_CN * vecQU + matA_UI_CN * vecQI + matA_UT_CN * vecQT) + valAlpha * vecPU
        vecQI = (1 - valAlpha) * (
            matA_IU_CN * vecQU + matA_II_CN * vecQI + matA_IT_CN * vecQT) + valAlpha * vecPI
        vecQT = (1 - valAlpha) * (
            matA_TU_CN * vecQU + matA_TI_CN * vecQI + matA_TT_CN * vecQT) + valAlpha * vecPT


    listQQQT = np.argsort(np.asarray(vecQT.transpose()), kind='heapsort')[0][::-1][0:10]

    count = 0

    setResults = set(listQQQT)
    setGroundTruth = set(Ptags)
    setRetrive = setGroundTruth & setResults

    p = len(setRetrive) / len(setGroundTruth)

    print('Precision@20 = ' + str(len(setRetrive) / 20) + ', ' + str(p))

    for i in range(valXiNum_U):
        t = (np.ones(shape(vecPU)) - vecPU)
        d = np.multiply(vecPU.todense(), t)
        dP_U_U = np.multiply(d, vecYU)
        inv_dD_U = -1 * sum(dP_U_U) / (colSumP ** 2)
        gradientP_U_U[:, i] = dP_U_U / colSumP + vecPU * inv_dD_U

        gradientP_I_U[:, i] = vecQI * inv_dD_U

    for i in range(valXiNum_I):
        t = (np.ones(shape(vecPI)) - vecPI)
        d = np.multiply(vecPI.todense(), t)
        dP_I_I = np.multiply(d, vecYI)
        inv_dD_I = -1 * sum(dP_I_I) / (colSumP ** 2)
        gradientP_I_I[:, i] = dP_I_I / colSumP + vecPI * inv_dD_I

        gradientP_U_I[:, i] = vecQU * inv_dD_I

    #
    # the gradients by Theta
    #
    vecPU_UU = np.ones((valUserNum, valFeatureNum_UU)) / valUserNum
    vecPU_UI = np.ones((valUserNum, valFeatureNum_UI)) / valUserNum
    vecPU_UT = np.ones((valUserNum, valFeatureNum_UT)) / valUserNum
    vecPU_IU = np.ones((valUserNum, valFeatureNum_UI)) / valUserNum
    vecPU_II = np.ones((valUserNum, valFeatureNum_II)) / valUserNum
    vecPU_IT = np.ones((valUserNum, valFeatureNum_IT)) / valUserNum
    vecPU_TU = np.ones((valUserNum, valFeatureNum_UT)) / valUserNum
    vecPU_TI = np.ones((valUserNum, valFeatureNum_IT)) / valUserNum
    vecPU_TT = np.ones((valUserNum, valFeatureNum_TT)) / valUserNum

    vecPI_UU = np.ones((valItemNum, valFeatureNum_UU)) / valItemNum
    vecPI_UI = np.ones((valItemNum, valFeatureNum_UI)) / valItemNum
    vecPI_UT = np.ones((valItemNum, valFeatureNum_UT)) / valItemNum
    vecPI_IU = np.ones((valItemNum, valFeatureNum_UI)) / valItemNum
    vecPI_II = np.ones((valItemNum, valFeatureNum_II)) / valItemNum
    vecPI_IT = np.ones((valItemNum, valFeatureNum_IT)) / valItemNum
    vecPI_TU = np.ones((valItemNum, valFeatureNum_UT)) / valItemNum
    vecPI_TI = np.ones((valItemNum, valFeatureNum_IT)) / valItemNum
    vecPI_TT = np.ones((valItemNum, valFeatureNum_TT)) / valItemNum

    vecPT_UU = np.ones((valTagNum, valFeatureNum_UU)) / valItemNum
    vecPT_UI = np.ones((valTagNum, valFeatureNum_UI)) / valItemNum
    vecPT_UT = np.ones((valTagNum, valFeatureNum_UT)) / valItemNum
    vecPT_IU = np.ones((valTagNum, valFeatureNum_UI)) / valItemNum
    vecPT_II = np.ones((valTagNum, valFeatureNum_II)) / valItemNum
    vecPT_IT = np.ones((valTagNum, valFeatureNum_IT)) / valItemNum
    vecPT_TU = np.ones((valTagNum, valFeatureNum_UT)) / valItemNum
    vecPT_TI = np.ones((valTagNum, valFeatureNum_IT)) / valItemNum
    vecPT_TT = np.ones((valTagNum, valFeatureNum_TT)) / valItemNum

    #
    # the gradients by Xi
    #
    vecPU_Xi_U = csr_matrix(np.ones((valUserNum, valXiNum_U)) / valUserNum, shape=(valUserNum, valXiNum_U))
    vecPU_Xi_I = csr_matrix(np.ones((valUserNum, valXiNum_I)) / valUserNum, shape=(valUserNum, valXiNum_I))
    vecPI_Xi_U = csr_matrix(np.ones((valItemNum, valXiNum_U)) / valItemNum, shape=(valItemNum, valXiNum_U))
    vecPI_Xi_I = csr_matrix(np.ones((valItemNum, valXiNum_I)) / valItemNum, shape=(valItemNum, valXiNum_I))
    vecPT_Xi_U = csr_matrix(np.ones((valTagNum, valXiNum_U)) / valTagNum, shape=(valTagNum, valXiNum_U))
    vecPT_Xi_I = csr_matrix(np.ones((valTagNum, valXiNum_I)) / valTagNum, shape=(valTagNum, valXiNum_I))

    #
    # calculate the derivatives of Xi by markovian process
    #
    for itr in range(iterNum):
        for i in range(valXiNum_U):
            vecPU_Xi_U[:, i] = (1 - valAlpha) * (matA_UU_CN * vecPU_Xi_U[:, i] + matA_UI_CN * vecPI_Xi_U[:, i]
                                                 + matA_UT_CN * vecPT_Xi_U[:, i]) + valAlpha * gradientP_U_U[:, i]

            vecPI_Xi_U[:, i] = (1 - valAlpha) * (matA_IU_CN * vecPU_Xi_U[:, i] + matA_II_CN * vecPI_Xi_U[:, i]
                                                 + matA_IT_CN * vecPT_Xi_U[:, i]) + valAlpha * gradientP_I_U[:, i]

            vecPT_Xi_U[:, i] = (1 - valAlpha) * (matA_TU_CN * vecPU_Xi_U[:, i] + matA_TI_CN * vecPI_Xi_U[:, i]
                                                 + matA_TT_CN * vecPT_Xi_U[:, i])

        for i in range(valXiNum_I):
            vecPU_Xi_I[:, i] = (1 - valAlpha) * (matA_UU_CN * vecPU_Xi_I[:, i] + matA_UI_CN * vecPI_Xi_I[:, i]
                                                 + matA_UT_CN * vecPT_Xi_I[:, i]) + valAlpha * gradientP_U_I[:, i]

            vecPI_Xi_I[:, i] = (1 - valAlpha) * (matA_IU_CN * vecPU_Xi_I[:, i] + matA_II_CN * vecPI_Xi_I[:, i]
                                                 + matA_IT_CN * vecPT_Xi_I[:, i]) + valAlpha * gradientP_I_I[:, i]

            vecPT_Xi_I[:, i] = (1 - valAlpha) * (matA_TU_CN * vecPU_Xi_I[:, i] + matA_TI_CN * vecPI_Xi_I[:, i]
                                                 + matA_TT_CN * vecPT_Xi_I[:, i])

    #
    # calculate the derivatives of Theta by markovian process
    #
    for itr in range(iterNum):

        for i in range(valFeatureNum_UU):
            vecPU_UUt = csr_matrix(vecPU_UU[:, i]).transpose()
            vecPI_UUt = csr_matrix(vecPI_UU[:, i]).transpose()
            vecPT_UUt = csr_matrix(vecPT_UU[:, i]).transpose()
            vecPU_UU[:, i] = (1 - valAlpha) * (
                matA_UU_CN * vecPU_UU[:, i] + (gradientA_UU_UU[i] * vecPU).toarray()[:, 0] +
                matA_UI_CN * vecPI_UU[:, i] +
                matA_UT_CN * vecPT_UU[:, i])

            vecPI_UU[:, i] = (1 - valAlpha) * (
                matA_IU_CN * vecPU_UU[:, i] + (gradientA_IU_UU[i] * vecPU).toarray()[:, 0] +
                matA_II_CN * vecPI_UU[:, i] +
                matA_IT_CN * vecPT_UU[:, i])

            vecPT_UU[:, i] = (1 - valAlpha) * (
                matA_TU_CN * vecPU_UU[:, i] + (gradientA_TU_UU[i] * vecPU).toarray()[:, 0] +
                matA_TI_CN * vecPI_UU[:, i] +
                matA_TT_CN * vecPT_UU[:, i])

        for i in range(valFeatureNum_UI):
            vecPU_UI[:, i] = (1 - valAlpha) * (matA_UU_CN * vecPU_UI[:, i] +
                                               matA_UI_CN * vecPI_UI[:, i] + (gradientA_UI_UI[i] * vecPI).toarray()[:,
                                                                             0] +
                                               matA_UT_CN * vecPT_UI[:, i])

            vecPI_UI[:, i] = (1 - valAlpha) * (matA_IU_CN * vecPU_UI[:, i] +
                                               matA_II_CN * vecPI_UI[:, i] + (gradientA_II_UI[i] * vecPI).toarray()[:,
                                                                             0] +
                                               matA_IT_CN * vecPT_UI[:, i])

            t = (gradientA_TI_UI[i] * vecPI).toarray()
            vecPT_UI[:, i] = (1 - valAlpha) * (matA_TU_CN * vecPU_UI[:, i] +
                                               matA_TI_CN * vecPI_UI[:, i] + (gradientA_TI_UI[i] * vecPI).toarray()[:,
                                                                             0] +
                                               matA_TT_CN * vecPT_UI[:, i])

        for i in range(valFeatureNum_UT):
            vecPU_UTt = csr_matrix(vecPU_UT[:, i]).transpose()
            vecPI_UTt = csr_matrix(vecPI_UT[:, i]).transpose()
            vecPT_UTt = csr_matrix(vecPT_UT[:, i]).transpose()
            vecPU_UT[:, i] = (1 - valAlpha) * (matA_UU_CN * vecPU_UT[:, i] +
                                               matA_UI_CN * vecPI_UT[:, i] +
                                               matA_UT_CN * vecPT_UT[:, i] + (gradientA_UT_UT[i] * vecPT).toarray()[:,
                                                                             0])

            vecPI_UT[:, i] = (1 - valAlpha) * (matA_IU_CN * vecPU_UT[:, i] +
                                               matA_II_CN * vecPI_UT[:, i] +
                                               matA_IT_CN * vecPT_UT[:, i] + (gradientA_IT_UT[i] * vecPT).toarray()[:,
                                                                             0])

            vecPT_UT[:, i] = (1 - valAlpha) * (matA_TU_CN * vecPU_UT[:, i] +
                                               matA_TI_CN * vecPI_UT[:, i] +
                                               matA_TT_CN * vecPT_UT[:, i] + (gradientA_TT_UT[i] * vecPT).toarray()[:,
                                                                             0])

        for i in range(valFeatureNum_UI):
            vecPU_IUt = csr_matrix(vecPU_IU[:, i]).transpose()
            vecPI_IUt = csr_matrix(vecPI_IU[:, i]).transpose()
            vecPT_IUt = csr_matrix(vecPT_IU[:, i]).transpose()
            vecPU_IU[:, i] = (1 - valAlpha) * (
                matA_UU_CN * vecPU_IU[:, i] + (gradientA_UU_IU[i] * vecPU).toarray()[:, 0] +
                matA_UI_CN * vecPI_IU[:, i] +
                matA_UT_CN * vecPT_IU[:, i])

            vecPI_IU[:, i] = (1 - valAlpha) * (
                matA_IU_CN * vecPU_IU[:, i] + (gradientA_IU_IU[i] * vecPU).toarray()[:, 0] +
                matA_II_CN * vecPI_IU[:, i] +
                matA_IT_CN * vecPT_IU[:, i])

            vecPT_IU[:, i] = (1 - valAlpha) * (
                matA_TU_CN * vecPU_IU[:, i] + (gradientA_TU_IU[i] * vecPU).toarray()[:, 0] +
                matA_TI_CN * vecPI_IU[:, i] +
                matA_TT_CN * vecPT_IU[:, i])

        for i in range(valFeatureNum_II):
            vecPU_IIt = csr_matrix(vecPU_II[:, i]).transpose()
            vecPI_IIt = csr_matrix(vecPI_II[:, i]).transpose()
            vecPT_IIt = csr_matrix(vecPT_II[:, i]).transpose()
            vecPU_II[:, i] = (1 - valAlpha) * (matA_UU_CN * vecPU_II[:, i] +
                                               matA_UI_CN * vecPI_II[:, i] + (gradientA_UI_II[i] * vecPI).toarray()[:,
                                                                             0] +
                                               matA_UT_CN * vecPT_II[:, i])

            vecPI_II[:, i] = (1 - valAlpha) * (matA_IU_CN * vecPU_II[:, i] +
                                               matA_II_CN * vecPI_II[:, i] + (gradientA_II_II[i] * vecPI).toarray()[:,
                                                                             0] +
                                               matA_IT_CN * vecPT_II[:, i])

            vecPT_II[:, i] = (1 - valAlpha) * (matA_TU_CN * vecPU_II[:, i] +
                                               matA_TI_CN * vecPI_II[:, i] + (gradientA_TI_II[i] * vecPI).toarray()[:,
                                                                             0] +
                                               matA_TT_CN * vecPT_II[:, i])

        for i in range(valFeatureNum_IT):
            vecPU_ITt = csr_matrix(vecPU_IT[:, i]).transpose()
            vecPI_ITt = csr_matrix(vecPI_IT[:, i]).transpose()
            vecPT_ITt = csr_matrix(vecPT_IT[:, i]).transpose()
            vecPU_IT[:, i] = (1 - valAlpha) * (matA_UU_CN * vecPU_IT[:, i] +
                                               matA_UI_CN * vecPI_IT[:, i] +
                                               matA_UT_CN * vecPT_IT[:, i] + (gradientA_UT_IT[i] * vecPT).toarray()[:,
                                                                             0])

            vecPI_IT[:, i] = (1 - valAlpha) * (matA_IU_CN * vecPU_IT[:, i] +
                                               matA_II_CN * vecPI_IT[:, i] +
                                               matA_IT_CN * vecPT_IT[:, i] + (gradientA_IT_IT[i] * vecPT).toarray()[:,
                                                                             0])

            vecPT_IT[:, i] = (1 - valAlpha) * (matA_TU_CN * vecPU_IT[:, i] +
                                               matA_TI_CN * vecPI_IT[:, i] +
                                               matA_TT_CN * vecPT_IT[:, i] + (gradientA_TT_IT[i] * vecPT).toarray()[:,
                                                                             0])

        for i in range(valFeatureNum_UT):
            vecPU_TUt = csr_matrix(vecPU_TU[:, i]).transpose()
            vecPI_TUt = csr_matrix(vecPI_TU[:, i]).transpose()
            vecPT_TUt = csr_matrix(vecPT_TU[:, i]).transpose()
            vecPU_TU[:, i] = (1 - valAlpha) * (
                matA_UU_CN * vecPU_TU[:, i] + (gradientA_UU_TU[i] * vecPU).toarray()[:, 0] +
                matA_UI_CN * vecPI_TU[:, i] +
                matA_UT_CN * vecPT_TU[:, i])

            vecPI_TU[:, i] = (1 - valAlpha) * (
                matA_IU_CN * vecPU_TU[:, i] + (gradientA_IU_TU[i] * vecPU).toarray()[:, 0] +
                matA_II_CN * vecPI_TU[:, i] +
                matA_IT_CN * vecPT_TU[:, i])

            vecPT_TU[:, i] = (1 - valAlpha) * (
                matA_TU_CN * vecPU_TU[:, i] + (gradientA_TU_TU[i] * vecPU).toarray()[:, 0] +
                matA_TI_CN * vecPI_TU[:, i] +
                matA_TT_CN * vecPT_TU[:, i])

        for i in range(valFeatureNum_IT):
            vecPU_TIt = csr_matrix(vecPU_TI[:, i]).transpose()
            vecPI_TIt = csr_matrix(vecPI_TI[:, i]).transpose()
            vecPT_TIt = csr_matrix(vecPT_TI[:, i]).transpose()
            vecPU_TI[:, i] = (1 - valAlpha) * (matA_UU_CN * vecPU_TI[:, i] +
                                               matA_UI_CN * vecPI_TI[:, i] + (gradientA_UI_TI[i] * vecPI).toarray()[:,
                                                                             0] +
                                               matA_UT_CN * vecPT_TI[:, i])

            vecPI_TI[:, i] = (1 - valAlpha) * (matA_IU_CN * vecPU_TI[:, i] +
                                               matA_II_CN * vecPI_TI[:, i] + (gradientA_II_TI[i] * vecPI).toarray()[:,
                                                                             0] +
                                               matA_IT_CN * vecPT_TI[:, i])

            vecPT_TI[:, i] = (1 - valAlpha) * (matA_TU_CN * vecPU_TI[:, i] +
                                               matA_TI_CN * vecPI_TI[:, i] + (gradientA_TI_TI[i] * vecPI).toarray()[:,
                                                                             0] +
                                               matA_TT_CN * vecPT_TI[:, i])

        for i in range(valFeatureNum_TT):
            vecPU_TTt = csr_matrix(vecPU_TT[:, i]).transpose()
            vecPI_TTt = csr_matrix(vecPI_TT[:, i]).transpose()
            vecPT_TTt = csr_matrix(vecPT_TT[:, i]).transpose()
            vecPU_TT[:, i] = (1 - valAlpha) * (matA_UU_CN * vecPU_TT[:, i] +
                                               matA_UI_CN * vecPI_TT[:, i] +
                                               matA_UT_CN * vecPT_TT[:, i] + (gradientA_UT_TT[i] * vecPT).toarray()[:,
                                                                             0])

            vecPI_TT[:, i] = (1 - valAlpha) * (matA_IU_CN * vecPU_TT[:, i] +
                                               matA_II_CN * vecPI_TT[:, i] +
                                               matA_IT_CN * vecPT_TT[:, i] + (gradientA_IT_TT[i] * vecPT).toarray()[:,
                                                                             0])

            vecPT_TT[:, i] = (1 - valAlpha) * (matA_TU_CN * vecPU_TT[:, i] +
                                               matA_TI_CN * vecPI_TT[:, i] +
                                               matA_TT_CN * vecPT_TT[:, i] + (gradientA_TT_TT[i] * vecPT).toarray()[:,
                                                                             0])

    J_Theta_UU = 0
    J_Theta_UI = 0
    J_Theta_UT = 0
    J_Theta_IU = 0
    J_Theta_II = 0
    J_Theta_IT = 0
    J_Theta_TU = 0
    J_Theta_TI = 0
    J_Theta_TT = 0

    J_Xi_U = 0
    J_Xi_I = 0

    J = 0

    for p_ins in Ptags:
        for n_ins in Ntags:
            t = vecQT[n_ins, 0] - vecQT[p_ins, 0]
            pp = (-1) * valBeta * t

            if pp > 709:
                #overflow
                s = 0
            else:
                r = math.exp(pp)
                s = 1 / (1 + r)

            J += s

            s_Theta_UU = vecPT_UU[n_ins, :] - vecPT_UU[p_ins, :]
            s_Theta_UI = vecPT_UI[n_ins, :] - vecPT_UI[p_ins, :]
            s_Theta_UT = vecPT_UT[n_ins, :] - vecPT_UT[p_ins, :]
            s_Theta_IU = vecPT_IU[n_ins, :] - vecPT_IU[p_ins, :]
            s_Theta_II = vecPT_II[n_ins, :] - vecPT_II[p_ins, :]
            s_Theta_IT = vecPT_IT[n_ins, :] - vecPT_IT[p_ins, :]
            s_Theta_TU = vecPT_TU[n_ins, :] - vecPT_TU[p_ins, :]
            s_Theta_TI = vecPT_TI[n_ins, :] - vecPT_TI[p_ins, :]
            s_Theta_TT = vecPT_TT[n_ins, :] - vecPT_TT[p_ins, :]

            J_Theta_UU += valBeta * s * (1 - s) * s_Theta_UU
            J_Theta_UI += valBeta * s * (1 - s) * s_Theta_UI
            J_Theta_UT += valBeta * s * (1 - s) * s_Theta_UT
            J_Theta_IU += valBeta * s * (1 - s) * s_Theta_IU
            J_Theta_II += valBeta * s * (1 - s) * s_Theta_II
            J_Theta_IT += valBeta * s * (1 - s) * s_Theta_IT
            J_Theta_TU += valBeta * s * (1 - s) * s_Theta_TU
            J_Theta_TI += valBeta * s * (1 - s) * s_Theta_TI
            J_Theta_TT += valBeta * s * (1 - s) * s_Theta_TT

            s_Xi_U = vecPT_Xi_U[n_ins, :] - vecPT_Xi_U[p_ins, :]
            s_Xi_I = vecPT_Xi_I[n_ins, :] - vecPT_Xi_I[p_ins, :]

            J_Xi_U = J_Xi_U + valBeta * s * (1 - s) * s_Xi_U
            J_Xi_I = J_Xi_I + valBeta * s * (1 - s) * s_Xi_I

    J /= lengthPT * lengthNT

    J_Theta_UU /= lengthPT * lengthNT
    J_Theta_UI /= lengthPT * lengthNT
    J_Theta_UT /= lengthPT * lengthNT
    J_Theta_IU /= lengthPT * lengthNT
    J_Theta_II /= lengthPT * lengthNT
    J_Theta_IT /= lengthPT * lengthNT
    J_Theta_IT /= lengthPT * lengthNT
    J_Theta_TI /= lengthPT * lengthNT
    J_Theta_TT /= lengthPT * lengthNT

    p = 1

    vecTheta_UU -= p * valLR * J_Theta_UU
    vecTheta_UI -= p * valLR * J_Theta_UI
    vecTheta_UT -= p * valLR * J_Theta_UT
    vecTheta_IU -= p * valLR * J_Theta_IU
    vecTheta_II -= p * valLR * J_Theta_II
    vecTheta_IT -= p * valLR * J_Theta_IT
    vecTheta_TU -= p * valLR * J_Theta_TU
    vecTheta_TI -= p * valLR * J_Theta_TI
    vecTheta_TT -= p * valLR * J_Theta_TT

    J_Xi_U /= lengthPT * lengthNT
    J_Xi_I /= lengthPT * lengthNT

    vecXi_U = vecXi_U - p * valLR * J_Xi_U
    vecXi_I = vecXi_I - p * valLR * J_Xi_I

    diffTheta = 0

    if len(J_Theta_UU) > 0:
        diffTheta += transpose(J_Theta_UU) * J_Theta_UU
    if len(J_Theta_UI) > 0:
        diffTheta += transpose(J_Theta_UI) * J_Theta_UI
    if len(J_Theta_UT) > 0:
        diffTheta += transpose(J_Theta_UT) * J_Theta_UT
    if len(J_Theta_IU) > 0:
        diffTheta += transpose(J_Theta_IU) * J_Theta_IU
    if len(J_Theta_II) > 0:
        diffTheta += transpose(J_Theta_II) * J_Theta_II
    if len(J_Theta_IT) > 0:
        diffTheta += transpose(J_Theta_IT) * J_Theta_IT
    if len(J_Theta_TU) > 0:
        diffTheta += transpose(J_Theta_TU) * J_Theta_TU
    if len(J_Theta_TI) > 0:
        diffTheta += transpose(J_Theta_TI) * J_Theta_TI
    if len(J_Theta_TT) > 0:
        diffTheta += transpose(J_Theta_TT) * J_Theta_TT

    diffTheta = diffTheta[0]

    diffXi = transpose(J_Xi_U) * J_Xi_U + transpose(J_Xi_U) * J_Xi_U
    diffXi = diffXi[0, 0]

    print('J = ' + str(J))
    print('diffXi      = ' + str(diffXi))
    print('vecXi_U     = ' + str(vecXi_U)     + '   vecXi_I     = ' + str(vecXi_I))
    print('J_Xi_U      = ' + str(J_Xi_U[0, 0])      + '   J_Xi_I      = ' + str(J_Xi_I[0, 0]))
    print('diffTheta   = ' + str(diffTheta))
    print('vecTheta_UU = ' + str(vecTheta_UU))
    print('vecTheta_UI = ' + str(vecTheta_UI))
    print('vecTheta_UT = ' + str(vecTheta_UT))
    print('vecTheta_IU = ' + str(vecTheta_IU))
    print('vecTheta_II = ' + str(vecTheta_II))
    print('vecTheta_IT = ' + str(vecTheta_IT))
    print('vecTheta_TU = ' + str(vecTheta_TU))
    print('vecTheta_TI = ' + str(vecTheta_TI))
    print('vecTheta_TT = ' + str(vecTheta_TT))


    '''
    if J < valConvergeThreshold_J:
        isConverge = True
    '''
    '''
    if diffTheta < 10e-40 and diffXi < 10e-40:
        isConverge = True
    '''

    if line_count >= 5000:
        isConverge = True

print('\nvecTheta_UU = ' + str(vecTheta_UU))
print('vecTheta_UI = ' + str(vecTheta_UI))
print('vecTheta_UT = ' + str(vecTheta_UT))
print('vecTheta_IU = ' + str(vecTheta_IU))
print('vecTheta_II = ' + str(vecTheta_II))
print('vecTheta_IT = ' + str(vecTheta_IT))
print('vecTheta_TU = ' + str(vecTheta_TU))
print('vecTheta_TI = ' + str(vecTheta_TI))
print('vecTheta_TT = ' + str(vecTheta_TT))

print('vecXi_U = ' + str(vecXi_U[0]))
print('vecXi_I = ' + str(vecXi_I[0]))
