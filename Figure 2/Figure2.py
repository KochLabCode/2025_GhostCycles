# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:34:53 2024

@author: Daniel Koch
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as mpcol
import matplotlib.ticker as ticker
from scipy.signal import find_peaks, argrelextrema
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

# models and parameters

models = [mod.vanDerPol_na, mod.vanDerPol_1g_na, mod.vanDerPol_2g_na]
models_aut_trans = [mod.vanDerPol_na_aut, mod.vanDerPol_1g_na_aut, mod.vanDerPol_2g_na_aut]
jacobians = [mod.vdp_na_aut_jac, mod.vdp1g_na_aut_jac, mod.vdp2g_na_aut_jac]
    
a_bif = [7.131, 3.145]; eps_bif = 0.01
eps = 0.02; tau = 16.5 # 10 for matching VdP1G, 16.5 for matching VdP2G

intMethod = 'LSODA'

#%% Determine periods in absence of forcing - T0

dt = 0.05; t_end = 200; t_tr = 100; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  

A = 0; omega = 0

para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]

T0 = []

for m in range(3):
    
    #transient phase
    solution = solve_ivp(models[m], (0,t_tr), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                          args=([para[m]]), method='LSODA') 
    
    #post transient
    IC = solution.y[:,solution.y.shape[1]-1]
    solution = solve_ivp(models[m], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para[m]]), method='LSODA') 
        
    xGrad = np.gradient(solution.y[0,:])
    
    if m==0:
        peaks_out, _ = find_peaks(xGrad,height=0.15)
    else:
        peaks_out, _ = find_peaks(xGrad,height=0.33)
        
    t_peaks_out = time[peaks_out]

    T_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
   
    T0.append(np.round(np.mean(T_out),0))
    

#%% Figure 2a - Arnold tongues

# Determine T_in and T_out as a function forcing amplitude and frequency

loadData = True

dt = 0.05
A_range = np.linspace(0.0,0.25,40)
A_range[A_range==0.0] = A_range[1]/2
fold_unforced = np.logspace(-1,1,200)


if loadData == False:
    for m in [0,1,2]:
    
        t_end = 60*T0[m]; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  
        
        omega_range = fold_unforced*2*np.pi/T0[m]
        
        T_inp = np.zeros((A_range.shape[0],omega_range.shape[0]))
        T_out = np.zeros((A_range.shape[0],omega_range.shape[0]))
        
        for i in range(A_range.shape[0]):
            print(i)
            for j in range(omega_range.shape[0]):
                
                A = A_range[i]
                omega = omega_range[j]
                para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]
                                
                solution = solve_ivp(models[m], (0,t_tr), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                                      args=([para[m]]), method='LSODA') 
                
                #post transient
                IC = solution.y[:,solution.y.shape[1]-1]
                solution = solve_ivp(models[m], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                                      t_eval=time, args=([para[m]]), method='LSODA')           
                xGrad = np.gradient(solution.y[0,:])
                
                
                if m==0:
                    peaks_out, _ = find_peaks(xGrad,height=0.15)
                else:
                    peaks_out, _ = find_peaks(xGrad,height=0.33)
                    
                    
                t_peaks_out = time[peaks_out]
                
                t_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
                
                T_inp[i,j] = 2*np.pi/omega 
                T_out[i,j] = np.mean(t_out)
        if m==0:
            np.save('T_inp_'+str(m)+'_tau_'+"{:.0f}".format(tau)+'.npy', T_inp)
            np.save('T_out_'+str(m)+'_tau_'+"{:.0f}".format(tau)+'.npy', T_out)
        else:
            np.save('T_inp_'+str(m)+'.npy', T_inp)
            np.save('T_out_'+str(m)+'.npy', T_out)

# calculate and plot Arnold tongues

sections = [[0.9,1.1],[1.5,4.6],[0.1,5.6]] # red lines indicating phase difference plots 
aTs = []

for m in [0,1,2]:
    if m==0:
        T_out = np.load('T_out_'+str(m)+'_tau_'+"{:.0f}".format(tau)+'.npy')
        T_inp = np.load('T_inp_'+str(m)+'_tau_'+"{:.0f}".format(tau)+'.npy')
    else:
        T_out = np.load('T_out_'+str(m)+'.npy')
        T_inp = np.load('T_inp_'+str(m)+'.npy')
    
    omega_range = fold_unforced*2*np.pi/T0[m]
    
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
                if abs(T_out[j,k]/T_inp[j,k] - p/q) < 0.01:
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
    
    xmin = np.argmin(np.abs(fold_unforced-sections[m][0]))
    xmax = np.argmin(np.abs(fold_unforced-sections[m][1]))
    
    plt.hlines(20,xmin,xmax,color='r',linestyles='dashed',lw=1)
        
    # largest Lyapunov exponents
    if loadData == False:
        if m == 0:
            nth = 5
        else:
            nth = 3
        
        #reduced ranges
        A_range = np.linspace(0.0,0.25,40)
        A_range[A_range==0.0] = A_range[1]/2
        A_range_r  = A_range[::nth]
        
        fold_unforced = np.logspace(-1,1,200)
        fold_unforced_r = fold_unforced[::nth]
        omega_range_r = fold_unforced_r*2*np.pi/T0[m]
        
        aT_reduced = np.copy(aT[::nth,::nth])
        
        idcs_LLE = np.asarray(np.where(aT_reduced==0)) # calculate LLEs only outside the Arnold tongues
        LLEs = []
        
        for i in range(idcs_LLE.shape[1]):
            print(i,idcs_LLE[0,i],idcs_LLE[1,i])
            A = A_range_r[idcs_LLE[0,i]]
            omega = omega_range_r[idcs_LLE[1,i]]
            para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]
            
            LLE = fun.maxLyap('LSODA', models_aut_trans[m], para[m], np.array([0.1,0.1,0]), jacobians[m], max(20*T0[m], 200), dt, max(2*T0[m], 50), plotFit=False)

            x = np.where(fold_unforced == fold_unforced_r[idcs_LLE[1,i]])[0][0]
            y = np.where(A_range == A_range_r[idcs_LLE[0,i]])[0][0]
            LLEs.append([LLE,x,y])
        
        LLEs_AT = np.asarray(LLEs)
        
        np.save('LLEs_AT_mod'+str(m)+'.npy',LLEs_AT)
      
    try:
        LLEs_AT = np.load('LLEs_AT_mod'+str(m)+'.npy')
        for i in range(LLEs_AT.shape[0]):
            x,y = LLEs_AT[i,1:]
            if LLEs_AT[i,0]>0.02:
                plt.plot(x,39-y,'xc', markeredgecolor='c',ms=1.25) #x shifted by -2 for plotting purposes only
        
    except:
        print('LLEs_AT_mod'+str(m)+'.npy not found!')
            
    plt.title('   '.join(l for l in lbls[1:]),fontsize=8)
    
    x_ = [ np.argmin(np.abs(fold_unforced-x__)) for x__ in [0.3,1,2,2.6,5,7,10] ]
    
    # ticks, labels etc
    plt.yticks(np.arange(A_range.shape[0])[::10],labels=np.round(np.flipud(A_range)[::10], decimals=2),fontsize=6)
    plt.xticks(np.arange(fold_unforced.shape[0])[::20],labels=np.round(fold_unforced[::20], decimals=1),fontsize=6)
    plt.xlabel('$\omega$ / $\omega_{0}$')
    plt.ylabel('A')              
    plt.subplots_adjust(top=1.0,
    bottom=0.155, left=0.195, right=1.0, hspace=0.145, wspace=0.18)
    
    plt.savefig('Fig2a_'+str(m)+'.svg')
    
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
            


#%% Figure 2b - 1:1 entrainment at different noise levels

np.random.seed(0)
sigma_range = [0,1e-5,2e-2,4e-1]
nVec = np.array([0,1])
A=0.125

loadData = True

# simulations
if loadData == False:
    for m in range(3):
        t_end = 60*T0[m]; dt = 0.01
        rmin = np.log10(sections[m][0])
        rmax = np.log10(sections[m][1])
        fold_unforced = np.logspace(rmin,rmax,30)
        omega_range = fold_unforced*2*np.pi/T0[m]
        
        om_inp_sig = []
        om_out_sig = []
        om_out_sd_sig = []
        
        def wrapper(z,t,p):
            return models[m](t,z,p)
        
        for sig in sigma_range:
            print(sig)
        
            om_inp = np.zeros(omega_range.shape[0])
            om_out = np.zeros(omega_range.shape[0])
            om_out_sd = np.zeros(omega_range.shape[0])
            for j in range(omega_range.shape[0]):
                
                omega = omega_range[j]
                para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]
                
                #transient phase
                time, simDat = fun.RK4_na_noisy(wrapper,para[m],np.array([0.1,0.1]),-T0[m],dt,0,nVec, sig, naFun = None,naFunParams = None)
                #post transient
                IC = simDat[:,simDat.shape[1]-1]
                time, simDat = fun.RK4_na_noisy(wrapper,para[m],IC,0,dt, t_end,nVec, sig, naFun = None,naFunParams = None)
                
                
                xGrad = np.gradient(simDat[0,:])
                
                if m==0:
                    peaks_out, _ = find_peaks(xGrad,height=0.025)
                else:
                    peaks_out, _ = find_peaks(xGrad,height=0.33)
                    
                t_peaks_out = time[peaks_out]
                
        
                t_inp = 2*np.pi/omega  
                t_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
                
                om_inp[j] = np.mean(1/t_inp)
                om_out[j] = np.mean(1/t_out)
                om_out_sd[j] = np.std(1/t_out)
                
            om_inp_sig.append(om_inp)
            om_out_sig.append(om_out)
            om_out_sd_sig.append(om_out_sd)
            
        FreqDiffPlot = np.asarray([om_inp_sig,om_out_sig,om_out_sd_sig])
        np.save('FreqDiffPlot_'+str(m)+'.npy', FreqDiffPlot)
    

# phase difference plots

for m in range(3):

    rmin = np.log10(sections[m][0])
    rmax = np.log10(sections[m][1])
    fold_unforced = np.logspace(rmin,rmax,30) 

    FreqDiffPlot = np.load('FreqDiffPlot_'+str(m)+'.npy')
    
    om_inp_sig,om_out_sig,om_out_sd_sig =FreqDiffPlot
    
    
    fig = plt.figure(figsize=(8.6*2/3*inCm,4*inCm))
    
    colors = ['k','r','m','b']
    
    for i in [2,1,0]:
    
        plt.fill_between(fold_unforced,om_out_sig[i]-om_inp_sig[i] + om_out_sd_sig[i],om_out_sig[i]-om_inp_sig[i]-om_out_sd_sig[i],color=colors[i],alpha=0.15)
        if i == 0:
            plt.plot(fold_unforced,om_out_sig[i]-om_inp_sig[i],color=colors[i],linestyle='dashed',label='$\sigma=$'+ str(sigma_range[i]))
        else:
            plt.plot(fold_unforced,om_out_sig[i]-om_inp_sig[i],color=colors[i],label='$\sigma=$'+ str(sigma_range[i]))
    
    plt.xscale('log')
    plt.xticks(fold_unforced[[0,14,29]],labels=np.round(fold_unforced[[0,14,29]], decimals=1+int(m==0)),fontsize=6)
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    
    plt.xlabel('$\omega$ / $\omega_{0}$')
    plt.ylabel('$\Omega - \omega$')              
    plt.subplots_adjust(top=0.825, bottom=0.345, left=0.2, right=0.84, hspace=0.145, wspace=0.18)
    
    plt.ylim([[-0.005,0.005],[-0.05,0.1],[-0.015,0.015]][m])
    
    plt.savefig('Fig2b_'+str(m)+'.svg')
    

#%% Figure 2c - ISI histograms

lbls = ['$VdP$','$VdP_{1g}$','$VdP_{2g}$']
A = 0.125; fold_unforced = np.logspace(-1,1,200)

for m in range(3):
    t_end = 60*T0[m];  npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  
    omega_range = fold_unforced*2*np.pi/T0[m]
        
    # determine ISIs
    
    ISIs= []
    
    for i in range(len(omega_range)):
        print('Figure 2c: simulation '  +str(33.3*m+100*i/len(omega_range)/3) + '% complete')
        omega = omega_range[i]
        para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]
        
        solution = solve_ivp(models[m], (0,t_tr), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                              args=([para[m]]), method='LSODA') 
        
        #post transient
        IC = solution.y[:,solution.y.shape[1]-1]
        solution = solve_ivp(models[m], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                              t_eval=time, args=([para[m]]), method='LSODA')       
        
        xGrad = np.gradient(solution.y[0,:])
        if m==0:
            peaks_out, _ = find_peaks(xGrad,height=0.15)
        else:
            peaks_out, _ = find_peaks(xGrad,height=0.33)
        t_peaks_out = time[peaks_out]
        
        T_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
        
        ISIs.append(T_out)

    # plot histograms
    
    n = len(omega_range)
    
    isi_max = 0
    isi_min = T0[m]
    for i in range(n):
        if max(ISIs[i])>isi_max:
            isi_max = max(ISIs[i])
        if min(ISIs[i])<isi_min:
            isi_min = min(ISIs[i])
        
    if m != 0:
        isi_min = 0
    
    nbins = 60
        
    histograms = []
    for i in range(n):

        h = np.histogram(ISIs[i],bins=nbins,range=(isi_min,isi_max))
        histograms.append(h)
    
    yAx = h[1]
        
    matrix = np.zeros((n,nbins))
    
    for i in range(n):
        matrix[i,:] = histograms[i][0]/np.sum(histograms[i][0])
    
    
    fig = plt.figure(figsize=(8.6*2/3*inCm,4*inCm))
    
    cw = plt.gca().imshow(matrix.T,cmap='binary',aspect=1.85,vmin=0, vmax=0.5)
    cbar = fig.colorbar(cw)
    cbar.set_label('$p$',rotation=0)
    
    if m==0:
        plt.yticks(np.arange(yAx.shape[0])[1:yAx.shape[0]:10],labels=np.round(yAx[1:yAx.shape[0]:10], decimals=1),fontsize=6)
    else:
        plt.yticks(np.arange(yAx.shape[0])[1:yAx.shape[0]:6],labels=np.round(yAx[1:yAx.shape[0]:6], decimals=0),fontsize=6)
    plt.xticks(np.arange(fold_unforced.shape[0])[::20],labels=np.round(fold_unforced[::20], decimals=1),fontsize=6)
    
    
    plt.xlabel('$\omega$ / $\omega_{0}$')
    plt.ylabel('ISI (a.u.)')
    plt.subplots_adjust(top=1.0, bottom=0.11, left=0.185, right=0.99, hspace=0.2, wspace=0.2)

    plt.savefig('Fig2c_'+str(m)+'.png', dpi=400)
    plt.savefig('Fig2c_'+str(m)+'.svg')
    

#%% Figure 2d - chaotic time course VdP2G system

m = 2; dt = 0.05

A = 0.125
fold_unforced = 8.5
omega = fold_unforced*2*np.pi/T0[m]
    
plt.figure(figsize=(8.6*inCm,4*inCm))

t_end = 200; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  
para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]

#transient phase
solution = solve_ivp(models[m], (0,t_tr), np.array([0.12,0.1]), rtol=1.e-6, atol=1.e-6,
                      args=([para[m]]), method=intMethod) 

#post transient
IC = solution.y[:,solution.y.shape[1]-1]
solution = solve_ivp(models[m], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                      t_eval=time, args=([para[m]]), method=intMethod) 

plt.plot(time, solution.y[0,:], color=colors[m],lw=1.5)

#post transient - perturbed
LLE = fun.maxLyap(intMethod, models_aut_trans[m], para[m], np.array([0.1,0.1,0]), jacobians[m], max(20*T0[m], 500), dt, max(5*T0[m], 100))

if LLE > 0.01:
    solution = solve_ivp(models[m], (0,t_end), IC+np.array([-0.01,0.01]), rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para[m]]), method=intMethod) 
    plt.plot(time, solution.y[0,:],'-',color=colors[m],lw=2, alpha=0.4)
    
plt.plot(time, A*np.sin(omega*time),'g',alpha=0.7,lw=0.75)
plt.xlabel('time (a.u.)'); plt.ylabel('x')
plt.ylim(-2.5,2.5); plt.xlim([0,175]); plt.xticks([0,50,100,150])
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Fig2d.svg')

    
#%% Figure 2e

loadData = True

# simulation 
m = 2
A = 0.125

n = 100
interval1 = np.logspace(-1,1,n)
interval1 = interval1[np.where(interval1 < 4.7)[0]]
interval2 = np.logspace(0,1,n)
interval2 = interval2[np.where(interval2 >= 4.7)[0]]

fold_unforced = np.unique(np.sort(np.concatenate((interval1, interval2))))


# Largest Lyapunov exponents
dt = 0.05
LLE_allMods = []

# calculate LLEs for all three models
if loadData == False:
    for m in range(3):
        omega_range = fold_unforced*2*np.pi/T0[m]
        LLEs = []
        for i in range(len(fold_unforced)):
            print('Figure 2c: simulation '  +str(33.3*m+100*i/len(fold_unforced)/3) + '% complete')
            omega = omega_range[i]
            para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]
            LLE = fun.maxLyap(intMethod, models_aut_trans[m], para[m], np.array([0.1,0.1,0]), jacobians[m], max(20*T0[m], 300), dt, max(5*T0[m], 80))
            LLEs.append(LLE)
        LLE_allMods.append(LLEs)
    
    LLEs = np.asarray(LLE_allMods)  
    np.save('fig2e_LLEs.npy', np.vstack((fold_unforced, LLEs)))


#  Orbit diagram

if loadData == False:
    # simulation
    dt = 0.005
    
    t_end = 50*T0[m]; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  
    omega_range = fold_unforced*2*np.pi/T0[m]
    
    trajectories = []
    for i in range(len(omega_range)):
        
        print(i/len(omega_range))
        
        omega=omega_range[i]
    
        para = [[eps,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]
        
        #transient phase
        solution = solve_ivp(models[m], (0,t_tr), np.array([0.12,0.1]), rtol=1.e-6, atol=1.e-6,
                              args=([para[m]]), method=intMethod) 
        
        #post transient
        IC = solution.y[:,solution.y.shape[1]-1]
        solution = solve_ivp(models[m], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                              t_eval=time, args=([para[m]]), method=intMethod)
        
        trajectories.append(solution.y)
        
    dataTrj = np.asarray(trajectories)


    # define section
    pS_xCoords = np.array([1.65,1.65])
    pS_yCoords = np.array([0.8,1.2])
    pS = np.column_stack((pS_xCoords,pS_yCoords))
    
    
    plotSections = False
    
    idx_transient = 2000
    
    sections = []
    for i in range(len(omega_range)): 
        
        print(i/len(omega_range))
           
        idx_xrange = np.where(np.logical_and(dataTrj[i,0,idx_transient:]>0.8, dataTrj[i,0,idx_transient:]<1.9))[0]
        idx_yrange = np.where(np.logical_and(dataTrj[i,1,idx_transient:]>0.8, dataTrj[i,1,idx_transient:]<1.1))[0]
        idx_xyrange = np.intersect1d(idx_xrange,idx_yrange)
        
        dat = dataTrj[i,:,idx_transient:][:,idx_xyrange]
        segStart = argrelextrema(dat[0],np.less)[0]
        
          
        sec = []
        if plotSections == True:
            plt.figure()
            plt.plot(dataTrj[i,0,idx_transient:],dataTrj[i,1,idx_transient:] ,'-o',color='#FF5555',lw=0.25)
            plt.xlabel('x'); plt.ylabel('y')
        for ii in range(len(segStart)-1):
            segment_x = dat[0,segStart[ii]:segStart[ii+1]]
            segment_y = dat[1,segStart[ii]:segStart[ii+1]]
            
            segment = np.column_stack((segment_x,segment_y))
            intersection = fun.intersections(pS,segment)
            sec.append(intersection)
            
            if plotSections == True:
                plt.plot(segment_x,segment_y ,'-',lw=0.75)
                plt.plot(pS_xCoords,pS_yCoords ,'-',color='k',alpha=0.3,lw=1.5)
                plt.plot(intersection[0,:], intersection[1,:],'xk' )
        sections.append(np.reshape(np.asarray(sec),np.asarray(sec).shape[:2]))
    
    # save as matrix
    m_sections = np.zeros((len(sections), max([s.shape[0] for s in sections])))
    
    for i in range(len(sections)):
        m_sections[i,:sections[i].shape[0]] = sections[i][:,1]
        
    np.save('fig2e_orbit.npy', m_sections)



# plot
m_sections = np.load('fig2e_orbit.npy')
LLE_load = np.load('fig2e_LLEs.npy')
fold_unforced = LLE_load[0,:]
LLEs = LLE_load[1:,:]

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6*inCm*1.08,6.5*inCm), gridspec_kw={'height_ratios': [1.5, 3]})

for i in range(3):
   ax1.plot(fold_unforced,LLEs[i], color=colors[i],lw=1, label = lbls[i])
    
ax1.set_ylabel('LLE')
ax1.set_xscale('log')
ax1.set_xlabel('')
ax1.xaxis.set_major_locator(ticker.NullLocator())
ax1.xaxis.set_minor_locator(ticker.NullLocator())
ax1.set_xlim(0.1,10)

for i in range(len(fold_unforced)):
    ax2.scatter(fold_unforced[i]*np.ones(m_sections.shape[1]), m_sections[i,:], facecolors=colors[2], s=0.3)    

sections_min = np.min(m_sections[np.where(m_sections > 0)])
sections_max = np.max(m_sections[np.where(m_sections > 0)])
delta = sections_max-sections_min
ax2.set_ylim(sections_min-0.05*delta,sections_max+0.05*delta)

ax2.set_ylabel('$y^{*}$')
ax2.set_xscale('log')
ax2.set_xlabel('$\omega$ / $\omega_{0}$')

plt.xlim(0.1,10)
plt.tight_layout()

plt.savefig('Fig2e.svg')