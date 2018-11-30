#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
       Graphe Structure Definition ans Visualization 
'''

__author__ = "Te SUN et Enlin ZHU"
__copyright__ = "Copyright 2018, MAP572"
__credits__ = ["Te SUN", "Enlin ZHU", "Prof",]
__license__ = "GPL"
__version__ = "0.5"
__maintainer__ = "Te SUN"
__email__ = "te.sun@polytechnique.edu"
__status__ = "Test"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import random
from sklearn.decomposition import PCA
class grapheG:
    def __init__(self,dim=2,n=1):
        self.n=n
        self.dim=dim
        self.adjacence=np.zeros((n,n))   
        #self.edge=np.zeros((n,1))
        if dim==2:
            self.pos=np.array([[np.random.uniform(0,1),np.random.uniform(0,1)] for i in range(self.n)])
        if dim==3:
            self.pos=self.pos=np.array([[np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)] for i in range(self.n)])
        self.distance=np.zeros((n,n))

    def reload(self,A,dim=2):
        self.__init__(dim,len(A))
        self.adjacence=A
        self.distance = self.Distance()
        

    def Distance(self):
        n=self.n
        A=self.adjacence.copy()
        tmp=self.adjacence.copy()
        matrix_dis=tmp.copy()
        dis=1
        for i in range(n):
            matrix_dis[i,i]=0
        for it in range(n):
            dis+=1
            tmp=np.dot(tmp,A)
            flag=1
            for i in range(n):
                for j in range(n):
                    if(i!=j and matrix_dis[i,j]==0 and tmp[i,j]!=0):
                        matrix_dis[i,j]=dis
                        flag=0
            if(flag==1):
                break;
        for i in range(n):
            for j in range(n):
                if matrix_dis[i,j]==0:
                    matrix_dis[i,j]=-1
        return matrix_dis

    def Energy(self,p):
        n=self.n
        matrix_distance=self.distance
        max_d=np.max(matrix_distance)
        matrix_distance=matrix_distance/max_d
        for i in range(n):
            matrix_distance[i,i]=1
        x=np.array([p[:,0]])
        Xminus=(x.T-x)**2
        y=np.array([p[:,1]])
        Yminus=(y.T-y)**2
        Zminus=0
        if self.dim==3:
            z=np.array([p[:,2]])
            Zminus=(z.T-z)**2
        B=np.sqrt(Xminus+Yminus+Zminus)/np.sqrt(self.dim)
        M=matrix_distance
        temp=(B-M)[np.where(M!=0)]/M[np.where(M!=0)]
        E=np.sum(temp*temp)
        return E

    def Gradiant(self,pos,epsilon=1.e-4):
        l=len(pos)
        res=np.zeros([l,2])
        E0=self.Energy(pos)
        for i in range(l):
            increment=np.zeros([l,2])
            increment[i,0]=epsilon
            res[i,0]=(self.Energy(pos+increment)-E0)/epsilon
            increment[i,0]=0
            increment[i,1]=epsilon
            res[i,1]=(self.Energy(pos+increment)-E0)/epsilon
        #res=
        return res 

    def GPF(self,delta=1.e-2,itermax=1000,tol=0.1):
        pos0=self.pos
        residu=np.inf
        iteration=0
        while (iteration<itermax and residu>tol):
            iteration+=1
            pos1=pos0-delta*(self.Gradiant(pos0))
            residu=self.Energy(pos1)
            pos0=pos1
            if(iteration%10==0):
                print(iteration, residu)
                self.pos=pos1
                self.Visual()
        self.pos=pos1
        self.Visual()


    def Gradiant_1d(self,x,epsilon=1.e-4):
        n=self.n
        l=n*self.dim
        grad=np.zeros(l)
        
        increment=np.zeros(l)
        E0=self.Energy(x.reshape((n,self.dim)))
        for i in range(l):
            increment[i]=epsilon
            grad[i]=(self.Energy((x+increment).reshape((n,self.dim)))-E0)/epsilon
            increment[i]=0
        return grad
    
    def Gradiant_exact(self,pos):
        n=self.n
        l=n*self.dim
        matrix_distance=self.distance/np.max(self.distance)+np.eye(n)
        x=np.array([pos[:,0]])
        Xminus=(x.T-x)
        y=np.array([pos[:,1]])
        Yminus=(y.T-y)
        
        Xtrick=Xminus+np.eye(n)
        Ytrick=Yminus+np.eye(n)
        Zminus=0
        Ztrick=0
        if(self.dim==3):
            z=np.array([pos[:,2]])
            Zminus=(z.T-z)
            Ztrick=Zminus+np.eye(n)
            
        
        inverseM=1/matrix_distance
        com=2*np.sqrt(2) *inverseM*(1/np.sqrt(2) *inverseM -1/ np.sqrt(Xtrick**2+Ytrick**2+Ztrick**2))
        tmp= com*Xminus
        gradX=np.sum(tmp,1)
        tmp= com *Yminus
        gradY=np.sum(tmp,1)
        grad2D=np.array([gradX,gradY]).T
        if(self.dim==3):
            tmp=com*Zminus
            gradZ=np.sum(tmp,1)
            grad3D=np.array([gradX,gradY,gradZ]).T
            ans=grad3D.reshape(l)
            return ans
        ans=grad2D.reshape(l)
        return ans
        
        
    def Visual3D(self,draw_edge=True):
        fig = plt.figure(figsize=[10,10])
        ax = fig.gca(projection='3d')
        pos = self.pos
        x=pos.T[0]
        y=pos.T[1]
        z=pos.T[2]
        ax.scatter(x,y,z,c='r')
        
        edge = set()
        for i in range(self.n):
            for j in range(self.n):
                if (i!=j and self.adjacence[i,j] >1.e-5):
                    x1,x2=pos[i][0],pos[j][0]
                    y1,y2=pos[i][1],pos[j][1]
                    z1,z2=pos[i][2],pos[j][2]
                    if(x1>x2):
                        x1,x2=x2,x1
                        y1,y2=y2,y1
                        z1,z2=z2,z1
                    edge.add(((x1,x2),(y1,y2),(z1,z2)))
        edge=list(edge)
        if(draw_edge):
            for i in range(len(edge)):
                ax.plot(*edge[i])
        plt.show()
        
        #Projection2D
        edge = set()
        fig = plt.figure(figsize=[10,10])
        for i in range(self.n):
            for j in range(self.n):
                if (i!=j and self.adjacence[i,j] >1.e-5):
                    x1,x2=pos[i][0],pos[j][0]
                    y1,y2=pos[i][1],pos[j][1]
                   
                    if(x1>x2):
                        x1,x2=x2,x1
                        y1,y2=y2,y1
                    edge.add(((x1,x2),(y1,y2)))
        edge=list(edge)
        if(draw_edge):
            for i in range(len(edge)):
                plt.plot(*edge[i],color='black',linewidth=0.3)
        
        point_c=pos[:,2]/np.max(pos[:,2])
        colors = cm.rainbow(point_c)       
        plt.scatter(x,y,color=colors)
        plt.show()
        #PCA Verion
        fig = plt.figure(figsize=[10,10])
        edge = set()
        pca=PCA(2)
        pca.fit(self.pos)
        pos=pca.transform(self.pos)
        for i in range(self.n):
            for j in range(self.n):
                if (i!=j and self.adjacence[i,j] >1.e-5):
                    x1,x2=pos[i][0],pos[j][0]
                    y1,y2=pos[i][1],pos[j][1]
                   
                    if(x1>x2):
                        x1,x2=x2,x1
                        y1,y2=y2,y1
                    edge.add(((x1,x2),(y1,y2)))
        edge=list(edge)
        for i in range(len(edge)):
            plt.plot(*edge[i],color='black',linewidth=0.3)
        colors = cm.GnBu(point_c)       
        plt.scatter(pos.T[0],pos.T[1],color=colors)
        plt.show()
        
    def Visual(self,draw_edge=True,pg=[]):
        marksize = 2000
        if len(pg)==0:
            pg = np.ones(self.n)*(1/self.n)
            marksize = 50
        pg1 = pg / np.max(pg)
        colors = cm.rainbow(pg1)
        pg2 = pg + 1/self.n
        pg2 = pg2 / np.max(pg2)
        sizes = [i*marksize for i in pg2]
        plt.figure(figsize=[10,10])
        pos = self.pos
        edge = set()
        for i in range(self.n):
            for j in range(self.n):
                if (i!=j and self.adjacence[i,j] >1.e-5):
                    x1,x2=pos[i][0],pos[j][0]
                    y1,y2=pos[i][1],pos[j][1]
                    if(x1>x2):
                        x1,x2=x2,x1
                        y1,y2=y2,y1
                    edge.add(((x1,x2),(y1,y2)))
        edge=list(edge)
        if(len(edge)<=1000):
            for i in range(len(edge)):
                plt.plot(*edge[i],color='b',zorder=1)
            for i in range(self.n):
                plt.annotate(s=round(pg[i]*100,1) ,xy=(pos[i]))
        else:
            edge=random.sample(edge,int(len(edge)/20))
            print(len(edge))
            for i in range(len(edge)):
                plt.plot(*edge[i],color='black',linewidth=0.3,zorder=1)
        plt.scatter(pos.T[0],pos.T[1],color=colors,s=sizes,zorder=2)
        plt.show()
        
    def VisualArrow(self,draw_edge=True,pg=[],M0=np.array(1)):
        marksize = 8000
        if len(M0)==1:
            M0 = self.adjacence[i,j]
        if len(pg)==0:
            pg = np.ones(self.n)*(1/self.n)
            marksize = 50
        pg1 = pg / np.max(pg)
        colors = cm.rainbow(pg1)
        pg2 = pg
        pg2 = pg2 / np.max(pg2)
        sizes = [i*marksize for i in pg2]
        plt.figure(figsize=[10,10])
        pos = self.pos
        edge = set()
        for i in range(self.n):
            for j in range(self.n):
                if (i!=j and M0[i,j] >1.e-5):
                    x1,x2=pos[i][0],pos[i][1]
                    y1,y2=pos[j][0]-pos[i][0],pos[j][1]-pos[i][1]
                    edge.add((x1,x2,y1,y2))
        edge=list(edge)
        if(len(edge)<=1000):
            for i in range(len(edge)):
                plt.arrow(*edge[i],head_width=0.02, head_length=0.1,length_includes_head=True,shape='left',fc='grey',ec='grey',overhang=-0.1)
            for i in range(self.n):
                plt.annotate(s=chr(i+65)+"_"+str(round(pg[i]*100,1)) ,xy=(pos[i]),size=20)
        else:
            edge=random.sample(edge,int(len(edge)/20))
            print(len(edge))
            for i in range(len(edge)):
                plt.arrow(*edge[i],head_width=0.02, head_length=0.1,length_includes_head=True,color='black',shape='left',linewidth=0.3)
        plt.scatter(pos.T[0],pos.T[1],color=colors,s=sizes)
        plt.show()
        
        
    def VisualCluster(self,cluster):
        plt.figure(figsize=[8,8])
        pos = self.pos
        colors=['red','blue','green','brown','purple','black','yellow','orange','pink']
        for i in range(len(pos)):    
            plt.scatter(pos[i][0],pos[i][1],color=colors[int(cluster[i])])
        edge = set()
        for i in range(self.n):
            for j in range(self.n):
                if (i!=j and self.adjacence[i,j] >1.e-5):
                    x1,x2=pos[i][0],pos[j][0]
                    y1,y2=pos[i][1],pos[j][1]
                    if(x1>x2):
                        x1,x2=x2,x1
                        y1,y2=y2,y1
                    edge.add(((x1,x2),(y1,y2)))
        edge=list(edge)
        if(len(edge)<=1000):
            for i in range(len(edge)):
                plt.plot(*edge[i],color='black',linewidth=0.3)
            for i in range(self.n):
                plt.annotate(s=i ,xy=(pos[i]))    
        plt.show()

    def GPO_d(self,itermax=1000,tol=1.e-4):
        x0=self.pos.reshape(self.n*self.dim)
        residu=np.inf
        iteration=0
        d=1.e-4
        #grad0=self.Gradiant_1d(x0)
        grad0=self.Gradiant_exact(self.pos)
        E0=self.Energy(self.pos)
        while (iteration<itermax and residu>tol):
            iteration+=1
            x1=x0-d*(grad0)
            self.pos=x1.reshape((self.n, self.dim))
            grad1=self.Gradiant_exact(self.pos)
            E1=self.Energy(x1.reshape((self.n, self.dim)))
            residu=np.abs(E1-E0)
            dp=(x1-x0)
            dD=(grad1-grad0)
            d=np.dot(dp,dD)/(np.linalg.norm(dD))**2
            grad0=grad1
            x0=x1
            E0=E1
            if(iteration%int((100)**2*500/self.n**2)==0):
                print("iteration, residu, d(步长): ",iteration, residu,d)
                #if(iteration%200==0):
                #self.pos=x1.reshape((self.n, self.dim))
                if(self.dim==3):
                    self.Visual3D()
                else:
                    self.Visual()
        print(iteration, residu,d)        
        self.pos=x1.reshape((self.n, self.dim))
        if(self.dim==3):
            self.Visual3D()
        else:
            self.Visual()
            
    def GPO_Armijio(self,itermax=1000,tol=1.e-4):
        n=self.n
        iteration=0
        residu=np.inf
        u=self.pos.reshape(n*self.dim)
        while (iteration<itermax and residu>tol):
            #w=self.Gradiant_1d(u)
            w=self.Gradiant_exact(self.pos)
            residu=w.dot(w)
            E0=self.Energy(u.reshape((n,self.dim)))            
            rho=1
            while self.Energy((u-rho*w).reshape((n,self.dim)))>E0-rho*0.0001*residu:#armijio critère, recherche linéaire
                rho*=0.8
            u-=rho*w
            iteration+=1
            self.pos=u.reshape((n,self.dim))
            if(iteration%int((100)**2*400/self.n**2)==0):
                print("iteration, residu, d(步长): ",iteration, residu,rho)
                #if(iteration%200==0):
                if(self.dim==3):
                    self.Visual3D()
                else:
                    self.Visual()
        print(iteration,residu,rho)        
        self.pos=u.reshape((n,self.dim))
        if(self.dim==3):
            self.Visual3D()
        else:
            self.Visual()
        
