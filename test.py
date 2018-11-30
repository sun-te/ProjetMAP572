#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    test
    Created on Tue Nov 20 14:50:21 2018
    @author: TeTe & Zeal
'''

import numpy as np
import os
#Change Your file name:
os.chdir("C:\\Users\\TeTe\\Desktop\\ProjetMAP572-master")
#import scipy as sp
#from scipy.optimize import  minimize
from TreePy import Cluster,Graph,Adjacency,PageRank
from bokeh.io import  show
from bokeh.plotting import figure
from bokeh.layouts import column,row
import pandas as pd
import time
from importlib import reload
#%%
'''
Part1
Question1.2: P(degre de s=k) \approx ck^{-\alpha} 
'''
time0=time.time()
n=100
M=40
dict_counter={}
for i in range(M):
    graph=Graph.grapheG()
    graph.reload(Adjacency.Matrix_delta(n,0))
    Adj=graph.adjacence
    degree=np.sum(Adj,1)
    degree.sort()
    degrees=np.unique(degree)
    
    for d in degree:
        if(d in dict_counter.keys()):
            dict_counter[d]+=1.0
        else:
            dict_counter[d]=1.0
df_res=pd.DataFrame(dict_counter, index=['num']).T
print(df_res)
degrees=np.array(df_res.index.tolist())
total=df_res['num'].sum()
count_d=np.array((df_res['num']/total).tolist())
c=1.5
fig=figure(x_axis_type="log",y_axis_type="log",y_range=[count_d.min()/2,1],plot_width=800,plot_height=800,title='Recheche de alpha')
fig.line(degrees,degrees**(-2)/c, line_width=2, legend="slope -2", color="red", line_dash='dotted')
fig.line(degrees,degrees**(-2.2)/c, line_width=2, legend="slope -2.2", color="green", line_dash='dotted')
fig.line(degrees,degrees**(-2.5)/c, line_width=2, legend="slope -2.5", color="orange", line_dash='dotted')
fig.line(degrees,degrees**(-3)/c, line_width=2, legend="slope -3", color="blue", line_dash='dotted')
fig.x(degrees, count_d,line_width=5)
show(fig)
print("time: ", time.time()-time0)
#%%
'''
Partie2
Question2.1-2.2: Optimisation de la representation du graphe par methode de gradient
'''
N =200
graph=Graph.grapheG()
graph.reload(Adjacency.Matrix_delta(N,10))
graph.Visual()
a=graph.Distance()
graph.Energy(graph.pos)
graph.GPO_d(itermax=200,tol=1.e-3)
graph.GPO_Armijio(itermax=800,tol=1.e-3)
#%%
'''
Question2.2 3d :
'''
N=200
graph=Graph.grapheG()
graph.reload(Adjacency.Matrix_delta(N,1),dim=3)
graph.Visual3D()
a=graph.Distance()
graph.Energy(graph.pos)
graph.GPO_d(itermax=300,tol=1.e-3)
graph.GPO_Armijio(itermax=800,tol=1.e-3)
print("Trois graphe: 1.Graphe en 3D; 2. Projection sur Vect(x,y); 3.Projection sur PCA")
#%%

'''Part 3'''
'''Question 3.1'''
N =100
graph=Graph.grapheG()
graph.reload(Adjacency.Matrix_delta(N,1))
graph.Visual()
a=graph.Distance()
graph.Energy(graph.pos)
graph.GPO_d(itermax=500,tol=1.e-3)
graph.GPO_Armijio(itermax=800,tol=1.e-3)
graph.Visual()
#%%
k=4
cluster_un=Cluster.Unnormalized(k,graph)
cluster_n=Cluster.Normalized_sym(k,graph)
graph.VisualCluster(cluster_un)
graph.VisualCluster(cluster_n)
#%%
'''Question 3.2'''
'''teste sur l'exemple sur Moodle (N'excute pas avec impatience)'''
MatriceAdjacence=np.loadtxt('StochasticBlockModel.txt')
graph=Graph.grapheG()
graph.reload(MatriceAdjacence)
graph.Visual()
graph.GPO_Armijio(itermax=800,tol=0.01)
#%%
k=2
cluster_n=Cluster.Normalized_sym(k,graph)
graph.VisualCluster(cluster_n)
#%%
'''Construction une SBM, puis optimiser le graphe, et representer le clustering'''
reload(Graph)
n=200
K=4
sommets=np.array(range(n))
partition=np.zeros(n)
for i in range(n):
    partition[i]=np.random.randint(0,K)
tmp=np.random.uniform(0,0.5,[K,K])
Q=tmp+tmp.T
Q=np.array([[0.8,0.02,0.2],[0.02,0.8,0.1],[0.2,0.1,0.8]])
eps=0.01
Q=np.ones([K,K])*eps
Q=Q+np.eye(K)*(1-eps)
print("Matrice Q: ")
print(Q)
graph=Graph.grapheG()
graph.reload(Adjacency.Matrix_SBM(partition,Q))

#%%
graph.GPO_Armijio(itermax=400)                                            #Optimisation
#%%
print("Original clustering")
graph.VisualCluster(partition)                                            #CLustering Original
cluster_n=Cluster.Normalized_sym(3,graph)                                 #Clustering detecter
print("detected one：")
graph.VisualCluster(cluster_n)
#%%
'''comparaison avec K_means'''
reload(Graph)
k=4
print("Spectral Clustering")
cluster_n=Cluster.Normalized_sym(k,graph) 
graph.VisualCluster(cluster_n) 
print("kmeans")
cluster=Cluster.K_MEANS(k,graph.pos) 
graph.VisualCluster(cluster)  
#%%
'''Part4'''
M = np.array([[0,  0, 0, 0, 1],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0,   1, 1, 0, 0],
             [0,   0, 1, 1, 0]])
'''Question4.1: teste avec l'exemple de Wiki'''
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
graph=Graph.grapheG()
graph.reload(M)

scores=PageRank.Scores(graph.adjacence,eps=0.15)

graph.reload(Adjacency.CompleteMatrix(M))
graph.GPO_Armijio()

graph.Visual(pg=scores)
print(scores)
#%%   
# Q4.3
reload(Graph)
reload(Adjacency)
reload(PageRank)

M = np.array([[0,1,1],
              [0,0,0],
              [0,0,0]
            ])
trustset=[1,1,1,0,0,0,0]
M = np.array([[0,0,0,0,0,0],
              [0,0,1,0,0,0],
              [0,1,0,0,0,0],
              [1,1,0,1,1,1],
              [1,0,1,1,0,1],
              [1,0,1,1,1,0]
            ])


# Q4.2
MM = Adjacency.PaddleMatrix(M,trustset)
graph=Graph.grapheG()
graph.reload(MM.T)
scores=PageRank.Scores(graph.adjacence,eps=0.15)
graph.reload(Adjacency.CompleteMatrix(M))
graph.GPO_Armijio()

graph.VisualArrow(pg=scores,M0=M)
print(scores)

reload(Graph)
reload(Adjacency)

M = Adjacency.Matrix_delta(600,1)
graph=Graph.grapheG()
graph.reload(M)
scores=PageRank.Scores(graph.adjacence,eps=0.15)
graph.reload(Adjacency.CompleteMatrix(M))
graph.GPO_Armijio()

graph.VisualArrow()
graph.Visual(pg=scores)
print(scores)
#%%   
