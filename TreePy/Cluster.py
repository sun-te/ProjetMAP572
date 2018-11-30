#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
       Graphe Clustering
'''

__author__ = "Te SUN et Enlin ZHU"
__copyright__ = "Copyright 2018, MAP572"
__credits__ = ["Te SUN", "Enlin ZHU", "Prof",]
__license__ = "GPL"
__version__ = "0.5"
__maintainer__ = "Te SUN"
__email__ = "te.sun@polytechnique.edu"
__status__ = "Test"



import time
#from Visual import grapheG
import numpy as np
import matplotlib.pyplot as plt
import random


def k_means(k,data,iter_max=10000):
    n,p=data.shape
    #choix=np.random.randint(low=0,high=n,size=k)
    center0=data[:k]
    it=0
    while it<=iter_max:
        err_now=0
        it+=1
        dict_clus={}
        dict_clus_index={}
        for i in range(k):
            dict_clus[i]=[]
            dict_clus_index[i]=[]
        for i in range(n):
            cluster=-1
            min_dis=np.inf
            for j in range(len(center0)):
                dis=np.linalg.norm(data[i]-center0[j])
                if(dis<min_dis):
                    min_dis=dis
                    cluster=j
            err_now+=min_dis
            dict_clus_index[cluster].append(i)
            dict_clus[cluster].append(data[i])
        center1=[]
        for key in dict_clus:
            len_cluster=len(dict_clus[key])
            if(len_cluster==0):
                continue
            center1+=[sum(dict_clus[key])/len_cluster]
        if(len(center0)==len(center1) and np.linalg.norm(np.array(center0)-np.array(center1))==0.0):
            print("OK")
            break;
        center0=center1.copy()
        print(len(center0))
    return dict_clus,dict_clus_index    

def K_MEANS(k,data):
    n,p=data.shape
    #choix=np.random.randint(low=0,high=n,size=k)
    center0=random.choices(data,k=k)
    #center0=data[:k]
    err=np.inf
    cluster=np.zeros(n)
    while(err>=1.e-7):
        for i in range(n):
            min_dis=np.inf
            for j in range(k):
                dis=np.linalg.norm(data[i]-center0[j])             
                if(dis<min_dis):
                    min_dis=dis
                    cluster[i]=j
        center1=np.zeros([k,p])
        for c in range(k):
            group=data[np.where(cluster==c)]
            if(len(group)==0):
                print("The cluster for group "+str(c)+" is empty!")
                continue;
            else:
                center1[c]=np.mean(group,0)
        err=np.linalg.norm(center1-center0)
        center0=center1
    global_distance=0
    for i in range(n):
        global_distance+=np.linalg.norm(data[i]-center1[int(cluster[i])])
    print(global_distance)
    return cluster



def Unnormalized(k,graph):
    W=graph.adjacence
    n=len(W)
    s=np.sum(W,0)
    D=np.zeros((n,n))
    for i in range(n):
        D[i,i]=s[i]
    L=D-W
    temp=np.linalg.eig(L)
    value,vector= np.real(temp[0]),np.real(temp[1])
    for i in range(n):
        for j in range(i+1,n):
            if(value[i]>value[j]):
                temp=value[i]
                value[i]=value[j]
                value[j]=temp
                vector[:,i],vector[:,j]=vector[:,j],vector[:,i]
    U=vector[:,:k]
    cluster=K_MEANS(k,U)
#    colors=['red','blue','green','brown','purple','black','yellow','orange','pink']
#    plt.figure(figsize=[10,10])
#    for i in range(graph.n):
#        plt.scatter(graph.pos[i][0],graph.pos[i][1],color=colors[int(cluster[i])])
#    plt.show()   
    return cluster;

def Normalized_sym(k,graph):
#    time0=time.time()
    W=graph.adjacence
    n=len(W)    
    s=np.sum(W,0) 
    tmp=s**(-1/2)
    D=np.zeros((n,n))
    for i in range(n):
        D[i,i]=tmp[i]    
    L=np.eye(n) - np.dot(np.dot(D,W),D)
    temp=np.linalg.eig(L)
    value,vector= np.real(temp[0]),np.real(temp[1])
    for i in range(n):
        for j in range(i+1,n):
            if(value[i]>value[j]):
                temp=value[i]
                value[i]=value[j]
                value[j]=temp
                vector[:,i],vector[:,j]=vector[:,j],vector[:,i]
    U=vector[:,:k]
    norm=np.array([np.sqrt(np.sum(U*U,1))])
    T=U/norm.T
    cluster=K_MEANS(k,T)
#    print(time.time()-time0)
#    colors=['red','blue','green','brown','purple','black','yellow','orange','pink']
#    plt.figure(figsize=[8,8])
#    for i in range(graph.n):
#        plt.scatter(graph.pos[i][0],graph.pos[i][1],color=colors[int(cluster[i])])
#    plt.show() 
    return cluster;
    

def UnnormSpectralCluster(k,graph):
    W=graph.adjacence
    n=len(W)
    s=np.sum(W,0)
    D=np.zeros((n,n))
    for i in range(n):
        D[i,i]=s[i]
    L=D-W
    temp=np.linalg.eig(L)
    value,vector= np.real(temp[0]),np.real(temp[1])
    for i in range(n):
        for j in range(i+1,n):
            if(value[i]>value[j]):
                temp=value[i]
                value[i]=value[j]
                value[j]=temp
                vector[:,i],vector[:,j]=vector[:,j],vector[:,i]
    U=vector[:,:k]
    dict_cluster_y, dict_cluster_index=k_means(k,U)
    for k in dict_cluster_index.keys():
        print(len(dict_cluster_index[k]))
    colors=['red','blue','green','brown','purple','black','yellow','orange','pink']
    plt.figure(figsize=[10,10])
    pos = graph.pos#np.array([[np.random.uniform(0,1),np.random.uniform(0,1)] for i in range(self.n)])
    edge = []
    for i in range(graph.n):
        for j in range(i,graph.n):
            if (graph.adjacence[i,j] == 1):
                edge.append(([pos[i][0],pos[j][0]],[pos[i][1],pos[j][1]]))
    for i in range(len(edge)):
        plt.plot(*edge[i],color='black')    
    i=0   
    for key in dict_cluster_index.keys():
        for p in dict_cluster_index[key]:
            plt.scatter(graph.pos[p][0],graph.pos[p][1],color=colors[i])
        i+=1;
    plt.show()   
    return;