# -*- coding: utf-8 -*-
"""
This code reproduces Figure 3 from the paper:

    Daniel Koch, Ulrike Feudel, Aneta Koseska (2025):
    Criticality governs response dynamics and entrainment of periodically forced ghost cycles.
    Physical Review E, XX: XXXX-XXXX.
    
Copyright: Daniel Koch
"""

# Import packages and modules

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.integrate import solve_ivp

import os
import sys

#paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join( os.path.dirname( __file__ ), '..' ))

import functions as fun
import models as mod

# plotting settings
pylab.rcParams.update(fun.get_rcparams())
plt.rcParams.update({'font.family':'Arial'})
inCm = 1/2.54
colors = ['#0000FF','m','#FF5555']

# window limits
xmin=-2.3;xmax=2.3
ymin=-2;ymax=2

xmid=0; ymid=0

# grid resolution
Ng=100
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss


# models and parameters

eps = 0.02; tau = 16.5
a_bif = [7.131, 3.145] # positions of SNIC bifurcations
eps_bif = 0.01
ampVect = [0.2,0,-0.2]

jacobians = [mod.jac_vdp,mod.jac_vdp1g, mod.jac_vdp2g]

#%%  Figure 3a

# Van der Pol
def flow_model(t,z):
    return mod.vanDerPol_constForce(t,z,para)

m_idx=0

# parameters
for i in range(3):
    A = ampVect[i]

    para = [eps,tau,A,0]
    
    plt.figure(figsize=(8.6*2/3*inCm*0.7,5.5*inCm*0.7))
    ax = plt.gca()

    # Flow
    U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
    ax.streamplot(Xg,Yg,U,V,density=0.7,color=[0.75,0.75,0.75,1],arrowsize=1,linewidth=0.85)
            
    #x-NC
    ax.plot(x_range,mod.xNC(x_range),'-k',lw=4,alpha=0.3)
    
    #y-NC default
    ax.plot(mod.yNC_vdp(y_range,A),y_range,'--',color=colors[0],lw=2)
    
    #intersection points
    xNC_arr = np.column_stack((x_range,mod.xNC(x_range)))
    yNC_arr = np.column_stack((mod.yNC_vdp(y_range,A),y_range))
    intersecPts = fun.intersections(xNC_arr,yNC_arr)
        
    for ii in range(intersecPts.shape[1]):
        x,y=intersecPts[:,ii]
        
        eigenvalues, eigenvectors = np.linalg.eig(jacobians[m_idx](x,y,para[0],para[1]))

        if any(eigenvalues<0):
            if any(eigenvalues>0): #saddle
                ax.plot(x,y,'o', color='grey', mec='k', ms=7)
            else: #stable FP
                ax.plot(x,y,'o', color='black', mec='k',ms=7)
        else: #stable FP
            ax.plot(x,y,'o', color='white', mec='k',ms=7)
    
    dt = 0.05
    t_plot = int(50/0.05); t_end = 100; npts = int(t_end/dt)
    time = np.linspace(0,t_end,npts+1)  

    solution = solve_ivp(mod.vanDerPol_constForce, (0,t_end), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para]), method='LSODA')  

    ax.plot(solution.y[0,t_plot:],solution.y[1,t_plot:],'-',color=colors[0],lw=0.9,alpha=0.5)
    
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    ax.set_xticks([xmin,xmid,xmax]); ax.set_yticks([ymin,ymid,ymax])
    ax.set_xticklabels([xmin,xmid,xmax]); ax.set_yticklabels([ymin,ymid,ymax])
    
    plt.subplots_adjust(top=0.97, bottom=0.21, left=0.205, right=0.935, hspace=0.185, wspace=0.205)
    plt.savefig('Fig3a_'+str(i+1)+'.svg')


#%%  Figure 3b - Van der Pol 1g system

def flow_model(t,z):
    return mod.vanDerPol_1g_constForce(t,z,para)

m_idx=1

# parameters
for i in range(3):
    A = ampVect[i]
    
    para = [eps,a_bif[0]-eps_bif,A,0] 
    
    plt.figure(figsize=(8.6*2/3*inCm*0.7,5.5*inCm*0.7))
    ax = plt.gca()

    # Flow
    U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
    ax.streamplot(Xg,Yg,U,V,density=0.7,color=[0.75,0.75,0.75,1],arrowsize=1,linewidth=0.85)
        
    #x-NC
    ax.plot(x_range,mod.xNC(x_range),'-k',lw=4,alpha=0.3)
    
    #y-NC default
    ax.plot(mod.yNC_1g(y_range,para),y_range,'--',color=colors[1],lw=2)
    
    #intersection points
    xNC_arr = np.column_stack((x_range,mod.xNC(x_range)))
    yNC_arr = np.column_stack((mod.yNC_1g(y_range,para),y_range))
    intersecPts = fun.intersections(xNC_arr,yNC_arr)
    
    for ii in range(intersecPts.shape[1]):
        x,y=intersecPts[:,ii]
        
        eigenvalues, eigenvectors = np.linalg.eig(jacobians[m_idx](x,y,para[0],para[1]))

        if any(eigenvalues<0):
            if any(eigenvalues>0): #saddle
                ax.plot(x,y,'o', color='grey', mec='k', ms=7)
            else: #stable FP
                ax.plot(x,y,'o', color='black', mec='k',ms=7)
        else: #stable FP
            ax.plot(x,y,'o', color='white', mec='k',ms=7)
    
    # Trajectories
    t_plot = int(100/dt); t_end = 500; npts = int(t_end/dt)
    time = np.linspace(0,t_end,npts+1)  
    
    solution = solve_ivp(mod.vanDerPol_1g_constForce, (0,t_end), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para]), method='LSODA')  

    ax.plot(solution.y[0,t_plot:],solution.y[1,t_plot:],'-',color=colors[1],lw=0.9,alpha=0.5)
    
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    ax.set_xticks([xmin,xmid,xmax]); ax.set_yticks([ymin,ymid,ymax])
    ax.set_xticklabels([xmin,xmid,xmax]); ax.set_yticklabels([ymin,ymid,ymax])
    
    plt.subplots_adjust(top=0.97, bottom=0.21, left=0.205, right=0.935, hspace=0.185, wspace=0.205)
    plt.savefig('Fig3b_'+str(i+1)+'.svg')

#%%  Figure 3c - Van der Pol 2g system

def flow_model(t,z):
    return mod.vanDerPol_2g_constForce(t,z,para)

m_idx=2

# parameters
for i in range(3):
    A = ampVect[i]

    para = [eps,a_bif[1]-eps_bif,A,0] 
    
    plt.figure(figsize=(8.6*2/3*inCm*0.7,5.5*inCm*0.7))
    ax = plt.gca()

    # Flow
    U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
    ax.streamplot(Xg,Yg,U,V,density=0.7,color=[0.75,0.75,0.75,1],arrowsize=1,linewidth=0.85)
        
    #x-NC
    ax.plot(x_range,mod.xNC(x_range),'-k',lw=4,alpha=0.3)
    
    #y-NC default
    ax.plot(mod.yNC_2g(y_range,para),y_range,'--',color=colors[2],lw=2)
    
    #intersection points
    xNC_arr = np.column_stack((x_range,mod.xNC(x_range)))
    yNC_arr = np.column_stack((mod.yNC_2g(y_range,para),y_range))
    intersecPts = fun.intersections(xNC_arr,yNC_arr)
    
    for ii in range(intersecPts.shape[1]):
        x,y=intersecPts[:,ii]
        
        eigenvalues, eigenvectors = np.linalg.eig(jacobians[m_idx](x,y,para[0],para[1]))

        if any(eigenvalues<0):
            if any(eigenvalues>0): #saddle
                ax.plot(x,y,'o', color='grey', mec='k', ms=7)
            else: #stable FP
                ax.plot(x,y,'o', color='black', mec='k',ms=7)
        else: #stable FP
            ax.plot(x,y,'o', color='white', mec='k',ms=7)
    
    # Trajectories
    t_plot = int(100/0.05); t_end = 500; npts = int(t_end/dt)
    time = np.linspace(0,t_end,npts+1)  
    
    solution = solve_ivp(mod.vanDerPol_2g_constForce, (0,t_end), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para]), method='LSODA')  

    ax.plot(solution.y[0,t_plot:],solution.y[1,t_plot:],'-',color=colors[2],lw=0.9,alpha=0.5)
    
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    ax.set_xticks([xmin,xmid,xmax]); ax.set_yticks([ymin,ymid,ymax])
    ax.set_xticklabels([xmin,xmid,xmax]); ax.set_yticklabels([ymin,ymid,ymax])
    
    plt.subplots_adjust(top=0.97, bottom=0.21, left=0.205, right=0.935, hspace=0.185, wspace=0.205)
    plt.savefig('Fig3c_'+str(i+1)+'.svg')



