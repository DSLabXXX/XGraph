__author__ = 'iankuoli'

#
# generate UU graph
#
#  user \t user \t <UU>
#
#  <UU> = f_UU_1,f_UU_2,...
#
from numpy import *
from scipy import *
from scipy.sparse import *

valUserNum = 0
valItemNum = 0
valTagNum = 0

f_Uid2Uindex = open('lastfm/map_Uid2Uindex.txt', 'r')
for line in f_Uid2Uindex:
    index = int(line.strip('\n').split('\t')[1])
    valUserNum = max([valUserNum, index])
f_Uid2Uindex.close()

f_Iid2Iindex = open('lastfm/map_Iid2Iindex.txt', 'r')
for line in f_Iid2Iindex:
    index = int(line.strip('\n').split('\t')[1])
    valItemNum = max([valItemNum, index])
f_Iid2Iindex.close()

f_Tid2Tindex = open('lastfm/map_Tid2Tindex.txt', 'r')
for line in f_Tid2Tindex:
    index = int(line.strip('\n').split('\t')[1])
    valTagNum = max([valTagNum, index])
f_Tid2Tindex.close()

valUserNum += 1
valItemNum += 1
valTagNum += 1

f_readUU = open('lastfm/UIT_Graph.txt', 'r')

list_U = list()
list_I = list()
list_T = list()
list_UI = list()
list_UT = list()
list_IT = list()

for line in f_readUU:
    l = line.strip('\n').split('\t')

    u = int(l[0])
    i = int(l[1])
    t = int(l[2])
    w_ui = float(l[3])
    w_ut = float(l[4])
    w_it = float(l[5])

    list_U.append(u)
    list_I.append(i)
    list_T.append(t)

    list_UI.append(w_ui)
    list_UT.append(w_ut)
    list_IT.append(w_it)
f_readUU.close()

matUI = csr_matrix((array(list_UI), (array(list_U), array(list_I))), shape=(valUserNum, valItemNum), dtype=float)
matUT = csr_matrix((array(list_UT), (array(list_U), array(list_T))), shape=(valUserNum, valTagNum), dtype=float)
matIT = csr_matrix((array(list_IT), (array(list_I), array(list_T))), shape=(valItemNum, valTagNum), dtype=float)

mat_UU_I = matUI * transpose(matUI)
mat_UU_T = matUT * transpose(matUT)

mat_II_U = transpose(matUI) * matUI
mat_II_T = matIT * transpose(matIT)

mat_TT_U = transpose(matUT) * matUT
mat_TT_I = transpose(matIT) * matIT

dictUU = dict()
n = mat_UU_I.nonzero()
for e in range(len(n[0])):
    x = n[0][e]
    y = n[1][e]

    if x < y:
        continue

    dictUU[(x, y)] = [mat_UU_I[x, y], 0]

n = mat_UU_T.nonzero()
for e in range(len(n[0])):
    x = n[0][e]
    y = n[1][e]

    if x < y:
        continue

    if (x, y) in dictUU:
        dictUU[(x, y)][1] = mat_UU_T[x, y]
    else:
        dictUU[(x, y)] = [0, mat_UU_T[x, y]]

f_writeUU = open('lastfm/UU_Graph.txt', 'w')
for key in dictUU.keys():
    f_writeUU.write(str(key[0]) + '\t' + str(key[1]) + '\t' + str(dictUU[key][0]) + ',' + str(dictUU[key][1]) + '\n')
f_writeUU.close()
print(len(dictUU))

dictII = dict()
n = mat_II_U.nonzero()
for e in range(len(n[0])):
    x = n[0][e]
    y = n[1][e]

    if x < y:
        continue

    dictII[(x, y)] = [mat_II_U[x, y], 0]

n = mat_II_T.nonzero()
for e in range(len(n[0])):
    x = n[0][e]
    y = n[1][e]

    if x < y:
        continue

    if (x, y) in dictII:
        dictII[(x, y)][1] = mat_II_T[x, y]
    else:
        dictII[(x, y)] = [0, mat_II_T[x, y]]

f_writeII = open('lastfm/II_Graph.txt', 'w')
for key in dictII.keys():
    f_writeII.write(str(key[0]) + '\t' + str(key[1]) + '\t' + str(dictII[key][0]) + ',' + str(dictII[key][1]) + '\n')
f_writeII.close()
print(len(dictII))

dictTT = dict()
n = mat_TT_U.nonzero()
for e in range(len(n[0])):
    x = n[0][e]
    y = n[1][e]

    if x < y:
        continue

    dictTT[(x, y)] = [mat_TT_U[x, y], 0]

n = mat_TT_I.nonzero()
for e in range(len(n[0])):
    x = n[0][e]
    y = n[1][e]

    if x < y:
        continue

    if (x, y) in dictTT:
        dictTT[(x, y)][1] = mat_TT_I[x, y]
    else:
        dictTT[(x, y)] = [0, mat_TT_I[x, y]]

f_writeTT = open('lastfm/TT_Graph.txt', 'w')
for key in dictTT.keys():
    f_writeTT.write(str(key[0]) + '\t' + str(key[1]) + '\t' + str(dictTT[key][0]) + ',' + str(dictTT[key][1]) + '\n')
f_writeTT.close()
print(len(dictTT))