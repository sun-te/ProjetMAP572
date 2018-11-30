#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Part 4'''

__author__ = "Te SUN et Enlin ZHU"
__copyright__ = "Copyright 2018, MAP572"
__credits__ = ["Te SUN", "Enlin ZHU", "Prof",]
__license__ = "GPL"
__version__ = "0.5"
__maintainer__ = "Te SUN"
__email__ = "te.sun@polytechnique.edu"
__status__ = "Test"

import numpy as np
import TreePy.Adjacency as Ad

# from numba import int64, float64
# spec = [
#     ('n', int64),               
#     ('adjacnce', int64[:,:]),
#     ('distance', float64[:,:]),
#     ('pos',float64[:,:])
# ]
# @jitclass(spec)
def RenormalizedMatrix(A):
    degree=np.sum(A,0)
    n=len(A)
    ans=np.zeros((n,n))
    for i in range(n):
        if(degree[i]!=0):
            ans[:,i]=A[:,i]/degree[i]
    return ans
        
def Scores(M,eps=0.15,tol=1.e-6):
    A=RenormalizedMatrix(M.copy())
    n=len(A)
    Pe=(1-eps)*A+eps/n
#     #print(Pe)
#     tup=np.linalg.eig(Pe)
#     values=np.real(tup[0])
#     vectors=np.real(tup[1].T)
#     for i in range(len(values)):
#         if(np.abs(values[i]-1)<=tol):
#             index=i;
#     ans=vectors[index]/np.sum(vectors[index])
    return Resolve(Pe-np.eye(n))

def PageRank(M, eps=1.0e-8, d=1):
    M=Ad.RenomalizedMatrix(M.copy())
    N=M.shape[1]
    v=np.random.rand(N, 1)
    v/=np.linalg.norm(v, 1)
    last_v=np.ones((N, 1),dtype=np.float32) * 100
    M_hat=(d*M) + (((1-d)/N)*np.ones((N, N), dtype=np.float32))
    while np.linalg.norm(v-last_v,2) > eps:
        last_v=v
        v=np.matmul(M_hat,v)
    return v

def Resolve(A):
    l = len(A)
    M = A.copy()
    Order = np.array([list(range(l))])
    M = np.concatenate((M, Order), axis=0)
    ans = np.zeros(l)
    for i in range(l):
        for j in range(l):
            if abs(M[i][j]) > abs(M[i][i]):
                M[:,[i, j]] = M[:,[j, i]]
        if abs(M[i,i]) < 1e-8:
            tmp = np.array([-M[x][l-1] for x in range(l-1)])
            for k in range(l-1):
                ans[int(M[l][k])]=tmp[k]
            ans[int(M[l][l-1])]=1    
            ans = ans/np.sum(ans)
            return ans    
        M[i]/=M[i,i]
        for j in range(l):     
            if j != i:
                M[j]-=(M[j,i]/M[i,i])*M[i]
    return ans

'''
M = np.array([[1,0,0,0,0,0,0,0,0,0,0],
              [0,0,1,0,0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0,0,0,0,0],
              [1,1,0,0,0,0,0,0,0,0,0],
              [0,1,0,1,0,1,0,0,0,0,0],
              [0,1,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,1,0,0,0,0,0,0]
            ]).T
    
v = pagerank(M, 0.001, 0.2)


k=Scores(M,0.15,0.001)
'''