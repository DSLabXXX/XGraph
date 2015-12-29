__author__ = 'iankuoli'

import numpy as np
from scipy import *
from scipy.sparse import *
from sklearn.preprocessing import normalize


#
# [ 1  0  2
# [
# [
#
row = array([0, 0, 1, 2, 2, 2])
col = array([0, 2, 2, 0, 1, 2])
data = array([1., 2., 1., 3., 5., 6.])
A = csr_matrix((data, (row, col)), shape=(3, 3))
B = csr_matrix((3, 3), dtype=float)

vec = A.dot(ones(3))
n = A.nonzero()
for i in range(len(n[0])):
    print(str(n[0][i]) + ',' + str(n[1][i]))

print(A.todense())

B = B + 2 * A

print(B.todense())

B = B + 2 * A

print(B.todense())

'''
dictUU = dict()
dictUU[(1, 2)] = [11, 12]
dictUU[(3, 4)] = [13, 14]

for key in dictUU.keys():
    print(str(key[0]) + '\t' + str(key[1]) + '\t' + str(dictUU[key][0]) + ',' + str(dictUU[key][1]))

dictUU[(3, 4)][1] = 24

for key in dictUU.keys():
    print(str(key[0]) + '\t' + str(key[1]) + '\t' + str(dictUU[key][0]) + ',' + str(dictUU[key][1]))
'''

'''
k = list()

for i in range(3):
    k.append(list())

for i in range(3):
    for j in range(5):
        k[i].append(j)

print(k)

s = '123456,'
s = s[:-1] + 'abcd'
print(s[:-1])
'''

'''

dictT = dict()
dictT['a'] = 1
dictT['b'] = 2
dictT['c'] = 3
dictT['d'] = 4

print(dictT)

print(list(dictT.keys()))

print(len(dictT))
dictT.pop('a')

print(dictT)
print(list(dictT.keys()))
print(len(dictT))


print(int(1.22))
'''

#inv_D_U = dia_matrix(np.divide(np.ones(3), [[1, 2, 3]]))
inv_D_U = dia_matrix((array([1, 2, 3, 4]), array([0])), shape=(4, 4))
print(shape(inv_D_U))
print(inv_D_U.todense())

Z = np.multiply(array([1, 2, 3]), array([1, 2, 3]))
print(Z)

print(A.todense())
A.data[:] = A.data * (1 - A.data) * 3
print(A.todense())

print((3*A).todense())

A.data[:] += 100
print(A.todense())

C = np.ones((3, 1))
B = A * C
print(B)

print(np.divide([[3, 9]], [[3, 3]]))
#print(1 / (1 + np.exp(-1*A)))
#print(np.sign(A))

colSum_U = np.ones((1, 3)) * A

print(colSum_U)


temp = csr_matrix(colSum_U)
temp.data[:] = 1 / temp.data
inv_D_U = dia_matrix((temp.data, array([0])), shape=(3, 3))
Z = A * inv_D_U
print(Z.todense())

aaa = ones((1,3)) * Z

print(math.exp((-1.) * 100000. * -0.00023))

kkkk = 20
kkkk /= 2*5
print(kkkk)


tttt = array([0.5, 0])

print(str(power(tttt, -1)))

rrrr = 1. / (1. + np.exp(tttt))
print(rrrr)


r = math.exp(-1000000000)
print(r)