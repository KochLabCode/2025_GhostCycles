# -*- coding: utf-8 -*-
"""
This code reproduces Figure 5 from the paper:

    Daniel Koch, Ulrike Feudel, Aneta Koseska (2025):
    Criticality governs response dynamics and entrainment of periodically forced ghost cycles.
    Physical Review E, XX: XXXX-XXXX.
    
Copyright: Daniel Koch
"""

# Import packages and modules

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as mpcol
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from tqdm import tqdm

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

Ng=100
x_range=np.linspace(xmin,xmax,Ng)
y_range=np.linspace(ymin,ymax,Ng)
grid_ss = np.meshgrid(x_range, y_range)
Xg,Yg=grid_ss

# models and parameters
jacobians = [mod.jac_vdp, mod.jac_vdp1g, mod.jac_vdp2g]

# non-autonomous models
models = [mod.vanDerPol_na, mod.vanDerPol_1g_na, mod.vanDerPol_2g_na]
models_aut_trans = [mod.vanDerPol_na_aut, mod.vanDerPol_1g_na_aut, mod.vanDerPol_2g_na_aut]
jacobians_na = [mod.vdp_na_aut_jac, mod.vdp1g_na_aut_jac, mod.vdp2g_na_aut_jac]

eps = 0.02; tau = 16.5
a_bif = [7.131, 3.145]; eps_bif = 0.01
ampVect = [1.5,0,-1.5]


#%% Determine periods in absence of forcing - T0

dt = 0.05; t_end = 200; t_tr = 100; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  

A = 0; omega = 0

para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]

T0 = []

for m_idx in range(3):
    
    #transient phase
    solution = solve_ivp(models[m_idx], (0,t_tr), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                          args=([para[m_idx]]), method='LSODA') 
    
    #post transient
    IC = solution.y[:,solution.y.shape[1]-1]
    solution = solve_ivp(models[m_idx], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para[m_idx]]), method='LSODA') 
        
    xGrad = np.gradient(solution.y[0,:])
    
    if m_idx==0:
        peaks_out, _ = find_peaks(xGrad,height=0.15)
    else:
        peaks_out, _ = find_peaks(xGrad,height=0.33)
        
    t_peaks_out = time[peaks_out]   

    T_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
   
    T0.append(np.round(np.mean(T_out),0))

#%%  Figure 5a  - phase space plots

# Van der Pol
def flow_model(t,z):
    return mod.vanDerPol_constForce(t,z,para)

m_idx = 0

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
    
    # Trajectories
    
    # VdP
    dt = 0.05
    t_plot = int(150/dt); t_end = 200; npts = int(t_end/dt)
    time = np.linspace(0,t_end,npts+1)  

    solution = solve_ivp(mod.vanDerPol_constForce, (0,t_end), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para]), method='LSODA')  #RK45, RK23, BDF, LSODA, Radau, DOP853
    ax.plot(solution.y[0,t_plot:],solution.y[1,t_plot:],'-',color=colors[0],lw=0.9,alpha=0.5)
    
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    ax.set_xticks([xmin,xmid,xmax]); ax.set_yticks([ymin,ymid,ymax])
    ax.set_xticklabels([xmin,xmid,xmax]); ax.set_yticklabels([ymin,ymid,ymax])
    
    plt.subplots_adjust(top=0.97, bottom=0.21, left=0.205, right=0.935, hspace=0.185, wspace=0.205)
    plt.savefig('Fig5a_'+str(i)+'.svg')



#%%  Figure 5b  - bifurcation diagrams

# Import XPPAUT data
with open("vdp_HB.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat_vdp_HB = np.asarray(data)


# Plot
plt.figure(figsize=(8.6*2/3*inCm*0.7,5.5*inCm*0.7))
plt.plot(dat_vdp_HB[:46,3],dat_vdp_HB[:46,6],color='#37ABC8')
plt.plot(dat_vdp_HB[46:110,3],dat_vdp_HB[46:110,6],'--k')
plt.plot(dat_vdp_HB[110:152,3],dat_vdp_HB[110:152,6],color='#37ABC8')
plt.plot(dat_vdp_HB[152:,3],dat_vdp_HB[152:,6],'#AC9393')
plt.plot(dat_vdp_HB[152:,3],dat_vdp_HB[152:,8],'#AC9393')

plt.yticks([-2,0,2], )
plt.xlim(-2,2)
plt.ylabel('x')
plt.xlabel('A')
plt.savefig('Fig5b.svg')

    
#%% Figure 5c - Arnold tongues

# Determine T_in and T_out as a function forcing amplitude and frequency

loadData = True
dt = 0.05
A_range = np.linspace(0.0,2,40)
A_range[A_range==0.0] = A_range[1]/2
fold_unforced = np.logspace(-1,1,200)


if loadData == False:
    for m_idx in [0,1,2]:
    
        t_end = 60*T0[m_idx]; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  
        
        omega_range = fold_unforced*2*np.pi/T0[m_idx]
        
        T_inp = np.zeros((A_range.shape[0],omega_range.shape[0]))
        T_out = np.zeros((A_range.shape[0],omega_range.shape[0]))
        
        for i in tqdm(range(A_range.shape[0]),desc=f"Simulations Arnold tongues (Figure 5c), model {m_idx}"):
            print(i)
            for j in range(omega_range.shape[0]):
                
                A = A_range[i]
                omega = omega_range[j]
                para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]
                
                #transient phase
                solution = solve_ivp(models[m_idx], (0,t_tr), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                                      args=([para[m_idx]]), method='LSODA') 
                
                #post transient
                IC = solution.y[:,solution.y.shape[1]-1]
                solution = solve_ivp(models[m_idx], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                                      t_eval=time, args=([para[m_idx]]), method='LSODA')           
                xGrad = np.gradient(solution.y[0,:])
                if m_idx==0:
                    peaks_out, _ = find_peaks(xGrad,height=0.15)
                else:
                    peaks_out, _ = find_peaks(xGrad,height=0.33)
                t_peaks_out = time[peaks_out]
                
                t_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
                
                T_inp[i,j] = 2*np.pi/omega 
                T_out[i,j] = np.mean(t_out)
        
        np.save('T_inp_'+str(m_idx)+'.npy', T_inp)
        np.save('T_out_'+str(m_idx)+'.npy', T_out)

# calculate and plot Arnold tongues

sections = [[0.9,1.05],[1.3,4.6],[0.1,5.6]] # red lines indicating phase difference plots 
aTs = []

for m_idx in [0,1,2]:#]range(3):
    T_out = np.load('T_out_'+str(m_idx)+'.npy')
    T_inp = np.load('T_inp_'+str(m_idx)+'.npy')
    
    omega_range = fold_unforced*2*np.pi/T0[m_idx]
    
    fig = plt.figure(figsize=(8.6*2/3*inCm,4*inCm))
    
    ratios = []
    pairs = []
    ratio_lbls = []
    
    
    maxNr = 4 # max integer for both p and q in all p/q-ratios considered
    for p in range(1,maxNr+1):
        for q in range(1,maxNr+1):
            if p/q not in ratios:
                ratios.append(p/q)
                pairs.append((p,q))
    
    ratios_sorted = np.sort(ratios)
    sorting_idcs = np.asarray([np.where(ratios==r)[0][0] for r in ratios_sorted])
    pairs = [pairs[i] for i in sorting_idcs]
    
    
    # calculate arnold tongues for all p/q
    m_arnold = np.zeros((len(ratios),A_range.shape[0],omega_range.shape[0]))
    area_percent_thr = 1 # threshold: only ATs that make up more than 1% in the A/omega space are considered
    
    
    for i in range(len(ratios)):
        p = pairs[i][0]
        q = pairs[i][1]

        for j in range(A_range.shape[0]):
            for k in range(omega_range.shape[0]):
                if abs(T_out[j,k]/T_inp[j,k] - p/q) < 0.015:
                    m_arnold[i,j,k] = 1

        ratio_lbls.append(str(p)+':'+str(q))
    
    aT = np.sum(m_arnold,axis=0)
    
    # winding numbers
    rho = T_out/T_inp
    
    # plot
    cw = plt.gca().imshow(np.flipud(rho),cmap='magma',aspect=3,norm=mpcol.LogNorm(vmin=5e-2, vmax=1e1))
    cbar = fig.colorbar(cw)
    cbar.set_label('$\\rho$',rotation=0)
    
    lbls = ['none']
    area = 0
    for i in range(len(ratios)):
        area_percent = 100*np.sum((m_arnold[i]))/m_arnold[i].size
        if area_percent > area_percent_thr:
            plt.gca().contour(np.flipud(m_arnold[i]),0,colors='k', linestyles='-',linewidths=1)
            mud = np.flipud(m_arnold[i])
            alphas=mud*0.30
            plt.gca().imshow(np.flipud(m_arnold[i]),aspect=3,cmap='binary',alpha=alphas)
            lbls.append(ratio_lbls[i])
            area += area_percent
    
    if m_idx == 0: plt.hlines(20,0,199,color='skyblue',linestyles='dashed',lw=1)
        
    # largest Lyapunov exponents

    if loadData == False:

        nth = 3
        
        #reduced ranges
        A_range[A_range==0.0] = A_range[1]/2
        A_range_r  = A_range[::nth]
        
        fold_unforced_r = fold_unforced[::nth]
        omega_range_r = fold_unforced_r*2*np.pi/T0[m_idx]
        
        aT_reduced = np.copy(aT[::nth,::nth])
        
        idcs_LLE = np.asarray(np.where(aT_reduced==0)) # calculate LLEs only outside the Arnold tongues
        LLEs = []
        
        for i in range(idcs_LLE.shape[1]):
            print(i,idcs_LLE[0,i],idcs_LLE[1,i])
            A = A_range_r[idcs_LLE[0,i]]
            omega = omega_range_r[idcs_LLE[1,i]]
            para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]
            
            LLE = fun.maxLyap('LSODA', models_aut_trans[m_idx], para[m_idx], np.array([0.1,0.1,0]), jacobians_na[m_idx], max(20*T0[m_idx], 200), dt, max(2*T0[m_idx], 50), plotFit=False)
            x = np.where(fold_unforced == fold_unforced_r[idcs_LLE[1,i]])[0][0]
            y = np.where(A_range == A_range_r[idcs_LLE[0,i]])[0][0]
            LLEs.append([LLE,x,y])
        
        LLEs_AT = np.asarray(LLEs)
        
        np.save('LLEs_AT_mod'+str(m_idx)+'.npy',LLEs_AT)
      
    try:
        LLEs_AT = np.load('LLEs_AT_mod'+str(m_idx)+'.npy')
        for i in range(LLEs_AT.shape[0]):
            x,y = LLEs_AT[i,1:]
            
            if LLEs_AT[i,0]>0.02:
                plt.plot(x,39-y,'xc', markeredgecolor='c',ms=1.25) #x shifted by -2 for plotting purposes only
        
    except:
        print('LLEs_AT_mod'+str(m_idx)+'.npy not found!')
            
    plt.title('   '.join(l for l in lbls[1:]),fontsize=8)
    
    
    # ticks, labels etc
    plt.yticks(np.arange(A_range.shape[0])[::10],labels=np.round(np.flipud(A_range)[::10], decimals=2),fontsize=6)
    plt.xticks(np.arange(fold_unforced.shape[0])[::20],labels=np.round(fold_unforced[::20], decimals=1),fontsize=6)
    plt.xlabel('$\omega$ / $\omega_{0}$')
    plt.ylabel('A')              
    plt.subplots_adjust(top=1.0,
    bottom=0.155, left=0.195, right=1.0, hspace=0.145, wspace=0.18)
    
    plt.savefig('Fig5c_'+str(m_idx)+'.svg')
    
    # plot individual tongues
    # plt.figure()
    # cols = int(len(ratios)/3)+1
    # idx=1
    # for i in range(len(ratios)):
    #     area_percent = 100*np.sum((m_arnold[i]))/m_arnold[i].size
    #     if area_percent > area_percent_thr:
    #         plt.subplot(2,cols,idx)
    #         plt.gca().imshow(np.flipud(m_arnold[i]),aspect=3,cmap='binary')
    #         plt.title(ratio_lbls[i])
    #         idx+=1
