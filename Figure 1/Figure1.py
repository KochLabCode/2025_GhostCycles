# -*- coding: utf-8 -*-
"""
This code reproduces Figure 1 from the paper:

    Daniel Koch, Ulrike Feudel, Aneta Koseska (2025):
    Criticality governs response dynamics and entrainment of periodically forced ghost cycles.
    Physical Review E, XX: XXXX-XXXX.
    
Copyright: Daniel Koch
"""

# Import packages and modules

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.signal import find_peaks
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

#%%  Figure 1a  - phase space

# window limits
xmin=-2.25;xmax=2.25
ymin=-1.2;ymax=1.2

xmid=0; ymid=0

Ng=151
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss
  

plt.figure(figsize=(8.6*2/3*inCm,5.5*inCm))
ax = plt.gca()

# Van der Pol

# parameters
a = 1; eps = 0.02; tau = 1; para = [eps,tau]

# Flow
def flow_model(t,z):
    return mod.vanDerPol(t,z,para)
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1,linewidth=0.75)

#x-NC
ax.plot(x_range,mod.xNC(x_range),'-k',lw=4)

#y-NC default
ax.plot(mod.yNC_vdp(y_range,0),y_range,'--',color=colors[0],lw=2)

# Trajectories

dt = 0.05
t_plot = 50
t_end = 100
npts = int(t_end/dt)
time = np.linspace(0,t_plot,npts+1)  

solution = solve_ivp(mod.vanDerPol, (0,t_end), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                      t_eval=time, args=([para]), method='LSODA')  #RK45, RK23, BDF, LSODA, Radau, DOP853

ax.plot(solution.y[0,t_plot:],solution.y[1,t_plot:],'-',color=colors[0],lw=2)

ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax]); ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax]); ax.set_yticklabels([ymin,ymid,ymax])

plt.subplots_adjust(top=0.97, bottom=0.21, left=0.205, right=0.935, hspace=0.185, wspace=0.205)

plt.savefig('Fig1a_vdp.svg', bbox_inches = "tight")

# Van der Pol 1g and 2g

plt.figure(figsize=(8.6*2/3*inCm,5.5*inCm))
ax = plt.gca()

# Flow
U,V=fun.vector_field(flow_model,grid_ss,dim='2D')    
ax.streamplot(Xg,Yg,U,V,density=0.8,color=[0.75,0.75,0.75,1],arrowsize=1,linewidth=0.75)

#x-NC
ax.plot(x_range,mod.xNC(x_range),'-k',lw=4)

#y-NC at SNIC - 2 ghosts
a = 3; para = [eps,a,0,0 ]# a set further from bifurcation for plotting 
ax.plot(mod.yNC_2g(y_range,para),y_range,'--',color=colors[2],lw=2.5)

#y-NC at SNIC - 1 ghost
a = 7; para = [eps,a,0,0 ]# a set further from bifurcation for plotting 
ax.plot(mod.yNC_1g(y_range,para),y_range,'--',color=colors[1],lw=2.5)

# # Trajectories
t_plot = int(50/0.05)
# 
a_bif = [7.131, 3.145]; eps_bif = 0.01

# VdP_2g
para = [eps,a_bif[1]-eps_bif]


solution = solve_ivp(mod.vanDerPol_2g, (0,t_end), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                      t_eval=time, args=([para]), method='LSODA')  #RK45, RK23, BDF, LSODA, Radau, DOP853

ax.plot(solution.y[0,t_plot:],solution.y[1,t_plot:],'-',color=colors[2],lw=2)

# VdP_1g
para = [eps,a_bif[0]-eps_bif]
solution = solve_ivp(mod.vanDerPol_1g, (0,t_end), np.array([0.1,0.1]), rtol=1.e-8, atol=1.e-8,
                      t_eval=time, args=([para]), method='LSODA')  #RK45, RK23, BDF, LSODA, Radau, DOP853

ax.plot(solution.y[0,t_plot:],solution.y[1,t_plot:],'-',color=colors[1],lw=2)

ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
ax.set_xticks([xmin,xmid,xmax]); ax.set_yticks([ymin,ymid,ymax])
ax.set_xticklabels([xmin,xmid,xmax]); ax.set_yticklabels([ymin,ymid,ymax])

plt.subplots_adjust(top=0.97, bottom=0.21, left=0.205, right=0.935, hspace=0.185, wspace=0.205)

plt.savefig('Fig1a_vdp_xg.svg', bbox_inches = "tight")



#%% Figure 1b - bifurcation diagrams

# Import XPPAUT data

with open("vdp_1g_SN.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat_vdp_1g_SN = np.asarray(data)

with open("vdp_1g_us.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat_vdp_1g_us = np.asarray(data)

with open("vdp_2g_SN.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat_vdp_2g_SN = np.asarray(data)

with open("vdp_2g_us.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat_vdp_2g_us = np.asarray(data)


# Plot
plt.figure(figsize=(8.62*2/3*inCm,5.5*inCm))

# Bifurcation diagram VdP_1g model
plt.subplot(2,1,1)

#plot unstable spiral from XPPAUT
plt.plot(dat_vdp_1g_us[:,3],dat_vdp_1g_us[:,6],'--k')

# plot SNs from XPPAUT
id_SN = 0
id_SN_end = 1292
plt.plot(dat_vdp_1g_SN[id_SN:id_SN_end,3],dat_vdp_1g_SN[id_SN:id_SN_end,6],'-',color='#37ABC8')
id_us_end = 2587
plt.plot(dat_vdp_1g_SN[id_SN_end:id_us_end,3],dat_vdp_1g_SN[id_SN_end:id_us_end,6],':k')

# plot min/max values of limit cycles
max_x = []; min_x = []
a_range = np.linspace(0,7,11)
for a in a_range:
    para = [eps,a_bif[0]-eps_bif]
    solution = solve_ivp(mod.vanDerPol_1g, (0,100), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                          t_eval=np.linspace(0,100,int(100/dt)+1), args=([para]), method='LSODA')  #RK45, RK23, BDF, LSODA, Radau, DOP853

    max_x.append(np.max(solution.y[0,:]))
    min_x.append(np.min(solution.y[0,:]))

plt.plot(a_range,max_x,'-',color='#AC9393')
plt.plot(a_range,min_x,'-',color='#AC9393')

plt.ylim(-2.4,2.4)
plt.yticks([-2,0,2], )
plt.xlim(0,10)
plt.xticks([])
plt.ylabel('x')


# # Bifurcation diagram VdP_2g model
plt.subplot(2,1,2)

#plot unstable spiral from XPPAUT
plt.plot(dat_vdp_2g_us[:,3],dat_vdp_2g_us[:,6],'--k')

# plot SNs from XPPAUT
id_SN = 0
id_SN_end = 1705
plt.plot(dat_vdp_2g_SN[id_SN:id_SN_end,3],dat_vdp_2g_SN[id_SN:id_SN_end,6],'-',color='#37ABC8')
id_us_end = 2500
plt.plot(dat_vdp_2g_SN[id_SN_end:id_us_end,3],dat_vdp_2g_SN[id_SN_end:id_us_end,6],':k')

id_SN = 2500
id_SN_end = 4205
plt.plot(dat_vdp_2g_SN[id_SN:id_SN_end,3],dat_vdp_2g_SN[id_SN:id_SN_end,6],'-',color='#37ABC8')
id_us_end = 4999
plt.plot(dat_vdp_2g_SN[id_SN_end:id_us_end,3],dat_vdp_2g_SN[id_SN_end:id_us_end,6],':k')

# plot min/max values of limit cycles (limit cycle branches were not captured by XPPAUT)
max_x = []; min_x = []
a_range = np.linspace(0,3,11)
for a in a_range:
    para = [eps,a_bif[1]-eps_bif]
    solution = solve_ivp(mod.vanDerPol_2g, (0,200), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                          t_eval=np.linspace(0,200,int(200/dt)+1)  , args=([para]), method='LSODA')  #RK45, RK23, BDF, LSODA, Radau, DOP853
    max_x.append(np.max(solution.y[0,:]))
    min_x.append(np.min(solution.y[0,:]))

plt.plot(a_range,max_x,'-',color='#AC9393')
plt.plot(a_range,min_x,'-',color='#AC9393')

plt.ylim(-2.4,2.4)
plt.yticks([-2,0,2], )
plt.xlim(0,10)
plt.xticks([0,5,10], )
plt.xticks()
plt.xlabel('$\\alpha$')
plt.ylabel('x')
plt.subplots_adjust(top=0.979, bottom=0.194, left=0.201, right=0.976, hspace=0.13, wspace=0.2)

plt.savefig('Fig1b.svg', bbox_inches = "tight")


#%% Figure 1c - timecourses

plt.figure(figsize=(2*8.6*inCm,4*inCm))
plt.subplot(1,3,1)

# VdP
t_end = 12
npts = int(t_end/dt)
time = np.linspace(0,t_end,npts+1)  

para = [eps,tau]
solution = solve_ivp(mod.vanDerPol, (0,t_end), np.array([1.5,-0.37]), rtol=1.e-6, atol=1.e-6,
                      t_eval=time, args=([para]), method='LSODA')  #RK45, RK23, BDF, LSODA, Radau, DOP853

plt.plot(time, solution.y[0,:],'-',color=colors[0],lw=1.5)
plt.plot(time, solution.y[1,:],'-.',color=colors[0],lw=1.5,alpha=0.5)
plt.ylabel('y')
plt.yticks([-2,0,2]); plt.ylim(-2.5,2.5)
plt.xlabel('time')
plt.xticks([0,6,12])
plt.gca().invert_yaxis()

# VdP_1g

t_end = 120
npts = int(t_end/dt)
time = np.linspace(0,t_end,npts+1)  

plt.subplot(1,3,2)
para = [eps,a_bif[0]-eps_bif]

solution = solve_ivp(mod.vanDerPol_1g, (0,t_end), np.array([1.9,0.6]), rtol=1.e-6, atol=1.e-6,
                      t_eval=time, args=([para]), method='LSODA')  #RK45, RK23, BDF, LSODA, Radau, DOP853

plt.plot(time, solution.y[0,:],'-',color=colors[1],lw=1.5)
plt.plot(time, solution.y[1,:],'-.',color=colors[1],lw=1.5,alpha=0.5)
plt.yticks([]); plt.ylim(-2.5,2.5)
plt.xlabel('time')
plt.xticks([0,60,120])
plt.gca().invert_yaxis()

# VdP_2g
plt.subplot(1,3,3)
para = [eps,a_bif[1]-eps_bif]

solution = solve_ivp(mod.vanDerPol_2g, (0,t_end), np.array([2,0.1]), rtol=1.e-6, atol=1.e-6,
                      t_eval=time, args=([para]), method='LSODA')  #RK45, RK23, BDF, LSODA, Radau, DOP853

plt.plot(time, solution.y[0,:],'-',color=colors[2],lw=1.5)
plt.plot(time, solution.y[1,:],'-.',color=colors[2],lw=1.5,alpha=0.5)
plt.yticks([]); plt.ylim(-2.5,2.5)
plt.xlabel('time')
plt.xticks([0,60,120])
plt.gca().invert_yaxis()

plt.subplots_adjust(top=0.88, bottom=0.41, left=0.065, right=0.99, hspace=0.2, wspace=0.09)

plt.savefig('Fig1c.svg', bbox_inches = "tight")


#%% Figure 1d - time scales

models = [mod.vanDerPol, mod.vanDerPol_1g, mod.vanDerPol_2g]

eps = 0.02; a_bif = [7.131, 3.145]; eps_bif = 0.01

IC = np.array([0.1,0.1])
para = [[eps,tau],[eps,a_bif[0]-eps_bif],[eps,a_bif[1]-eps_bif]]

t_tr = 100; t_end=5000; dt=0.05;
idx_tr = int(t_tr/dt)
npts = int(t_end/dt)
time = np.linspace(0,t_end,npts+1)  


# run simulations
velocities = []
for m_idx in range(3):
    solution = solve_ivp(models[m_idx], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para[m_idx]]), method='LSODA') 
    velocities.append(fun.euklideanVelocity(solution.y[:,idx_tr:].T, 1)/dt)

# plot
plt.figure(0, figsize=(2*8.6*inCm,4*inCm)) # velocity histograms
plt.figure(1, figsize=(10*inCm,2.2*inCm)) # inset

h = np.asarray([])
lbls = ['VdP','VdP_{1g}','VdP_{2g}']
timescales = []
peakDistFilt = [100, 50, 150]

for i in range(3):
    
    # velocity histograms
    plt.figure(0)
    plt.subplot(1,3,i+1)
    ax  = plt.gca()
    
    if i==0: plt.ylabel('relative frequency')
    
    v = np.asarray(velocities[i]).flatten()
    histo, bin_edges = ax.hist(np.log(v),200, weights=np.zeros_like(v) + 1. / v.size,histtype='step',color=colors[i],alpha=1,linewidth=0.5)[:2]
    plt.hist(np.log(v),200, weights=np.zeros_like(v) + 1. / v.size,alpha=0.3, label = 'a='+str(lbls[i]),color=colors[i])
    h = np.concatenate((h, v),axis=0) 
    
    binCentres = np.array([np.mean([bin_edges[i-1],bin_edges[i]]) for i in range(1,len(bin_edges))])
    binDistance = abs(binCentres[1]-binCentres[0])

    histo = np.concatenate((np.array([0]), histo, np.array([0])))
    binCentres = np.concatenate((np.array([binCentres[0]-binDistance]), binCentres, np.array([binCentres[len(binCentres)-1]+binDistance])))
    
    peaks, _ = find_peaks(histo,distance=peakDistFilt[i],prominence=0.0005)
    
    plt.plot(binCentres[peaks],histo[peaks],'x',color=colors[i],ms=5,lw=15)

    tsc = 1/np.exp(binCentres[peaks])
    timescales.append(tsc)
    
    plt.xlabel('log(v) (a.u.)')
    
    plt.xticks(range(-6,5,1))
    if i == 0: 
        plt.yticks([0,0.125,0.25])
    else:
        plt.yticks([])
    
    plt.subplots_adjust(top=0.88, bottom=0.41, left=0.065, right=0.99, hspace=0.2, wspace=0.09)
    
    
    # insets
    plt.figure(1) 
    plt.subplot(1,3,i+1)
    ax  = plt.gca()
    
    x_min = 3.5
    x_max = 4
    y_max = histo[peaks][len(peaks)-1]
    
    ax.hist(np.log(v),15, range=(x_min,x_max), weights=np.zeros_like(v) + 1. / v.size,histtype='step',color=colors[i],alpha=1,linewidth=0.5)[:2]
    plt.hist(np.log(v),15, range=(x_min,x_max), weights=np.zeros_like(v) + 1. / v.size,alpha=0.3, label = 'a='+str(lbls[i]),color=colors[i])
    
    plt.xlim(x_min,x_max); plt.ylim(0,y_max)
    plt.yticks([0,np.round(y_max,4)])
    plt.tight_layout()
   
plt.figure(0)
plt.savefig('Fig1d.svg', bbox_inches = "tight")

plt.figure(1)
plt.savefig('Fig1d_inset.svg', bbox_inches = "tight")