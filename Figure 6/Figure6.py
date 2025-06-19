# -*- coding: utf-8 -*-
"""
This code reproduces Figure 6 from the paper:

    Daniel Koch, Ulrike Feudel, Aneta Koseska (2025):
    Criticality governs response dynamics and entrainment of periodically forced ghost cycles.
    Physical Review E, XX: XXXX-XXXX.
    
Copyright: Daniel Koch
"""

# Import packages and modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcol
import matplotlib.pylab as pylab
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
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
lbls = ['slow/fast regime','ghost cycle']

#model parameters  

I_snic = 39.69345
eps = 0.5

I_vals = [55, I_snic+eps]
IC = np.array([-40,0.1])

# common parameters
v1=-1.2;v2=18;cm=20;gk=8;gl=2;vca=120;vk=-80;vl=-60

#%% periods in absence of forcing

I_vals = [55, I_snic+eps]

dt = 0.05; t_end = 3000; t_tr = 100;  npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  

sig = 0
A = 0; omega = 0

T0 = []

for I in I_vals:
    
    if I >90:
        # type II excitability
        gca=4.4;phi=0.04;v3=2;v4=30 
    else:
        # type I excitability
        gca=4;phi=0.06667;v3=12;v4=17.4
    
    params = [I,v1,v2,v3,v4,gk,gl,gca,vca,vk,vl,phi,cm]
    
    #transient phase
    solution = solve_ivp(mod.MorrisLecar, (-t_tr,0), IC, rtol=1.e-6, atol=1.e-6,
                          args=([params]), method='LSODA') 
    
    #post transient
    IC = solution.y[:,solution.y.shape[1]-1]
    solution = solve_ivp(mod.MorrisLecar, (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([params]), method='LSODA') 
    
    peaks_out, _ = find_peaks(solution.y[0,:],distance=20,height=10, prominence=(35, None))
    
    t_peaks_out = time[peaks_out]

    T_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
    T0.append(np.round(np.mean(T_out),0))


# type I excitability
gca=4;phi=0.06667;v3=12;v4=17.4


#%% Supplementary Figure 6a - bifurcation diagram

with open("ML_bifurcation.dat") as f:
    lines = f.readlines()
    text = "".join(lines)

data = []

for l in lines:
    row = []
    for n in l.split(' ')[:len(l.split(' '))-1]: 
        row.append(float(n))
    data.append(np.asarray(row))

dat_ML = np.asarray(data)

plt.figure(figsize=(17.2/3*inCm,5.5*inCm))
plt.subplot(1,1,1)

#plot spiral branch
plt.plot(dat_ML[1:707,3],dat_ML[1:707,6],'-',color='#37ABC8')
plt.plot(dat_ML[707:2413,3],dat_ML[707:2413,6],'--k')

#plot SN branch
plt.plot(dat_ML[2415:2980,3],dat_ML[2415:2980,6],'-',color='#37ABC8')
plt.plot(dat_ML[2981:3899,3],dat_ML[2981:3899,6],':k')

#plot HB branch
plt.plot(dat_ML[3907:4448,3],dat_ML[3907:4448,6],':',color='#AC9393')
plt.plot(dat_ML[3907:4448,3],dat_ML[3907:4448,8],':',color='#AC9393')
plt.plot(dat_ML[4449:,3],dat_ML[4449:,6],'-',color='#AC9393') #6
plt.plot(dat_ML[4449:,3],dat_ML[4449:,8],'-',color='#AC9393') #6

plt.ylim(-50,40)
plt.xlim(10,125)

plt.ylabel('V (mV)')
plt.xlabel('I$_{ext}$ ($\mu$A/cm$^2$)')

plt.subplots_adjust(top=0.914, bottom=0.203, left=0.224, right=0.947, hspace=0.13, wspace=0.2)

plt.savefig('FigS6a.svg')


#%% Supplementary Figure 6b,c - timescales

# calculate velocity

t_tr = 100; t_end=10000; dt=0.05; npts = int((t_end+t_tr)/dt); time = np.linspace(0,t_end,npts+1) 

idx_tr = int(t_tr/dt)

velocities_hist = []

for I in tqdm(I_vals, "Simulations for Figure 6b,c"):
    params = [I,v1,v2,v3,v4,gk,gl,gca,vca,vk,vl,phi,cm]
    solution = solve_ivp(mod.MorrisLecar, (0,t_end), IC, rtol=1.e-6, atol=1.e-6, t_eval=time,
                          args=([params]), method='LSODA') 
    velocities_hist.append(fun.euklideanVelocity(solution.y[:,idx_tr:].T, 1)/dt)



# timescales and colorscale limits
plt.figure(0, figsize=(4/3*8.6*inCm,8*inCm))

h = np.asarray([])
timescales = []
peakDistFilt = [10, 10, 150]

for i in range(2):
    
    plt.figure(0)
    plt.subplot(2,2,i+1)
    ax  = plt.gca()
    
    if i==0: plt.ylabel('relative frequency')
    
    v = np.asarray(velocities_hist[i]).flatten()
    histo, bin_edges = ax.hist(np.log(v),200, weights=np.zeros_like(v) + 1. / v.size,histtype='step',color=colors[i],alpha=1,linewidth=0.5)[:2]
    plt.hist(np.log(v),200, weights=np.zeros_like(v) + 1. / v.size,alpha=0.3, label = 'a='+str(lbls[i]),color=colors[i])
    h = np.concatenate((h, v),axis=0) 
    
    binCentres = np.array([np.mean([bin_edges[i-1],bin_edges[i]]) for i in range(1,len(bin_edges))])
    binDistance = abs(binCentres[1]-binCentres[0])


    histo = np.concatenate((np.array([0]), histo, np.array([0])))
    binCentres = np.concatenate((np.array([binCentres[0]-binDistance]), binCentres, np.array([binCentres[len(binCentres)-1]+binDistance])))
    
    
    peaks, _ = find_peaks(histo,distance=peakDistFilt[i],prominence=(0.001, None))
    
    plt.plot(binCentres[peaks],histo[peaks],'x',color=colors[i],ms=5,lw=15)
    
    for i in range(len(peaks)):
        plt.text(binCentres[peaks[i]],histo[peaks[i]],'$\\tau_'+str(i+1)+'$')

    tsc = 1/np.exp(binCentres[peaks])
    timescales.append(tsc)
    
    plt.xlabel('log(velocity) (a.u.)')
    
    plt.xticks(range(-5,3,1))   
    
# highlighted timescales

t_transient = [0,230]; 


for I in [0,1]:
    if I == 0:
        dt=0.02
        t_tr = T0[I]*1.1
        t_end = T0[I]*2.1
    else:
        dt=0.01
        t_tr = T0[I]*1.05
        t_end = T0[I]*2.05
    
    idx_tr = int(t_tr/dt); npts = int((t_end)/dt); time = np.linspace(0,t_end,npts+1) 
    params = [I_vals[I],v1,v2,v3,v4,gk,gl,gca,vca,vk,vl,phi,cm]
    solution = solve_ivp(mod.MorrisLecar, (0,t_end+t_tr), [-30,0], rtol=1.e-6, atol=1.e-6, t_eval=time,
                          args=([params]), method='LSODA') 
    
    velocities = fun.euklideanVelocity(solution.y[:,idx_tr:].T, 1)/dt
    
    v_timescales = 1/timescales[I]
    
    idcs_v_ts = []
    
    for i in range(len(v_timescales)):
        idcs_v_ts.append( (np.abs(velocities - v_timescales[i])).argmin() )
        
    
    plt.subplot(2,2,I+3)
    time = np.linspace(-t_tr,t_end-t_tr,npts+1) 
    plt.plot(time[idx_tr:],solution.y[0,idx_tr:],'-', color=colors[I])
    for i in range(len(idcs_v_ts)):
        
        j = idcs_v_ts[i]
        plt.plot(time[idx_tr+j],solution.y[0,idx_tr+j],'o',color=colors[I],ms=3.5,lw=15)
        plt.text(time[idx_tr+j]+5,solution.y[0,idx_tr+j]+2,'$\\tau_'+str(i+1)+'$')
        
    plt.ylabel('V (mV)')
    plt.xlabel('time (ms)')
   
plt.subplots_adjust(top=0.942,
bottom=0.242,
left=0.115,
right=0.987,
hspace=0.824,
wspace=0.288)

plt.savefig('FigS6b,c.svg')

#%% Supplementary Figure 6d - Arnold tongues

load = True

I_range = [0,1]

dt = 0.05

A_range = np.linspace(0,10,40)
A_range[A_range==0.0] = A_range[1]/2
    

# simulations and save data

if load == False:
    for I in I_range:
        t_end = 60*T0[I]; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1) 
        fold_unforced = np.logspace(-1,1,200)
        omega_range = fold_unforced*2*np.pi/T0[I]
        
        if I == 2:
            # type II excitability
            gca=4.4;phi=0.04;v3=2;v4=30 
        else:
            # type I excitability
            gca=4;phi=0.06667;v3=12;v4=17.4
        
        
        T_inp = np.zeros((A_range.shape[0],omega_range.shape[0]))
        T_out = np.zeros((A_range.shape[0],omega_range.shape[0]))
        
        for i in tqdm(range(A_range.shape[0]),"Simulations for Figure 6d ("+str(I)+")"):
            for j in range(omega_range.shape[0]):
                A = A_range[i]
                omega = omega_range[j]
                
                params = [I_vals[I],v1,v2,v3,v4,gk,gl,gca,vca,vk,vl,phi,cm,A,omega]
        
                #transient phase
                solution = solve_ivp(mod.MorrisLecar_na, (-T0[I],0), np.array([30,0.1]), rtol=1.e-6, atol=1.e-6,
                                      args=([params]), method='LSODA') 
                
                #post transient
                IC = solution.y[:,solution.y.shape[1]-1]
                solution = solve_ivp(mod.MorrisLecar_na, (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                                      t_eval=time, args=([params]), method='LSODA') 
         
                
                peaks_out, _ = find_peaks(solution.y[0,:],distance=20,height=10, prominence=(35, None))
                
                t_peaks_out = time[peaks_out]

                t_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
                
                T_inp[i,j] = 2*np.pi/omega 
                T_out[i,j] = np.mean(t_out)

        np.save('T_inp_ML_'+str(I)+'.npy', T_inp)
        np.save('T_out_ML_'+str(I)+'.npy', T_out)

# load data and plot

area_entr = []
sections = [[0.86,1.15],[0.7,2.7]] # for frequency difference plots 
aTs = []

for I in range(2):
    T_out = np.load('T_out_ML_'+str(I)+'.npy')
    T_inp = np.load('T_inp_ML_'+str(I)+'.npy')
    
    A_range = np.linspace(0,10,40)
    A_range[A_range==0.0] = A_range[1]/2
 
 
    fold_unforced = np.logspace(-1,1,200)
    omega_range = fold_unforced*2*np.pi/T0[I]
    
    fig = plt.figure(figsize=(8.6*2/3*inCm,4*inCm))
    
    ratios = []
    pairs = []
    ratio_lbls = []
    
    maxNr = 4
    area_percent_thr = 1
    
    for n in range(1,maxNr+1):
        for m in range(1,maxNr+1):
            if n/m not in ratios:
                ratios.append(n/m)
                pairs.append((n,m))
    
    ratios_sorted = np.sort(ratios)
    sorting_idcs = np.asarray([np.where(ratios==r)[0][0] for r in ratios_sorted])
    pairs = [pairs[i] for i in sorting_idcs]
    
    
    m_arnold = np.zeros((len(ratios),A_range.shape[0],omega_range.shape[0]))
    
    for i in range(len(ratios)):
        m = pairs[i][0]
        n = pairs[i][1]
        
        for j in range(A_range.shape[0]):
            for k in range(omega_range.shape[0]):
                if abs(T_out[j,k]/T_inp[j,k] - m/n) < 0.01:
                    m_arnold[i,j,k] = 1

        ratio_lbls.append(str(m)+':'+str(n))
    
    aTs.append(np.sum(m_arnold,axis=0))
    
    rho = T_out/T_inp
    

    cw = plt.gca().imshow(np.flipud(rho),cmap='magma',aspect=3,norm=mpcol.LogNorm(vmin=5e-2, vmax=1e1))
    cbar = fig.colorbar(cw)
    cbar.set_label('$\\rho$',rotation=0)
    
    lbls = ['none']
    area = 0
    for i in range(len(ratios)):
        # print(i)
        area_percent = 100*np.sum((m_arnold[i]))/m_arnold[i].size
        if area_percent > area_percent_thr:
            plt.gca().contour(np.flipud(m_arnold[i]),0,colors='k', linestyles='-',linewidths=1)
            m = np.flipud(m_arnold[i])
            alphas=m*0.30
            plt.gca().imshow(np.flipud(m_arnold[i]),aspect=3,cmap='binary',alpha=alphas)
            lbls.append(ratio_lbls[i])
            area += area_percent
    area_entr.append(area)
    
    xmin = np.argmin(np.abs(fold_unforced-sections[I][0]))
    xmax = np.argmin(np.abs(fold_unforced-sections[I][1]))
    
    plt.hlines(20,xmin,xmax,color='r',linestyles='dashed',lw=1)
    
    
    ## LLEs
    
    try:
        LLEs_AT = np.load('LLEs_AT_mod'+str(mod)+'.npy')
        maxLLE = np.max(LLEs_AT[:,0])
        for i in range(LLEs_AT.shape[0]):
            x = np.where(fold_unforced == LLEs_AT[i,3])[0][0]
            y = np.where(A_range == LLEs_AT[i,1])[0][0]
            
            if LLEs_AT[i,0]>0.005:
                plt.plot(x-2,40-y,'xc', markeredgecolor='c',ms=5) #x shifted by -2 for plotting purposes only
            
    except:
        pass
    plt.title('   '.join(l for l in lbls[1:]),fontsize=8)
    
    plt.yticks(np.arange(A_range.shape[0])[::10],labels=np.round(np.flipud(A_range)[::10], decimals=1),fontsize=6)
    plt.xticks(np.arange(fold_unforced.shape[0])[::20],labels=np.round(fold_unforced[::20], decimals=1),fontsize=6)
    plt.xlabel('$\omega$ / $\omega_{0}$')
    plt.ylabel('A (mV)')              
    plt.subplots_adjust(top=1.0,
    bottom=0.155,
    left=0.195,
    right=1.0,
    hspace=0.145,
    wspace=0.18)
    
    # plot individual tongues for easier annotation
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

    plt.savefig('FigS6d_'+str(I)+'.svg')

#%% Supplementary Figure 6e - response to noise: frequency differences  

load = True

np.random.seed(0)
I_range = [0,1]
sigma_range = [0,1e-4,0.2]
A=5
nVec = np.array([1,0])

# type I excitability
gca=4;phi=0.06667;v3=12;v4=17.4

# simulation

def wrapper(z,t,p):
    return mod.MorrisLecar_na(t,z,p)

if load == False:
    for I in I_range:
        t_end = 60*T0[I]; dt = 0.05
        rmin = np.log10(sections[I][0])
        rmax = np.log10(sections[I][1])
        fold_unforced = np.logspace(rmin,rmax,30)
        omega_range = fold_unforced*2*np.pi/T0[I]
        
        om_inp_sig = []
        om_out_sig = []
        om_out_sd_sig = []
        
        for sig in sigma_range:
            print(sig)
        
            om_inp = np.zeros(omega_range.shape[0])
            om_out = np.zeros(omega_range.shape[0])
            om_out_sd = np.zeros(omega_range.shape[0])
            for j in range(omega_range.shape[0]):
                print(j)
                omega = omega_range[j]
                params = [I_vals[I],v1,v2,v3,v4,gk,gl,gca,vca,vk,vl,phi,cm,A,omega]
            
                #transient phase
                time, simDat = fun.RK4_na_noisy(wrapper,params,np.array([30,0.1]),-T0[I],dt,0,nVec, sig, naFun = None,naFunParams = None)
                #post transient
                IC = simDat[:,simDat.shape[1]-1]
                time, simDat = fun.RK4_na_noisy(wrapper,params,IC,0,dt,t_end,nVec, sig, naFun = None,naFunParams = None)
                 
                inp_dat = A*np.sin(omega*time)
                peaks_inp, _ = find_peaks(inp_dat)
                t_peaks_inp = time[peaks_inp]

                peaks_out, _ = find_peaks(simDat[0,:],distance=20,height=10, prominence=(35, None))
                t_peaks_out = time[peaks_out]     
                    
                t_inp = np.array([ np.abs(t_peaks_inp[i+1] - t_peaks_inp[i]) for i in range(len(t_peaks_inp)-1) ])
                t_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
                
    
                om_inp[j] = np.mean(1/t_inp)
                om_out[j] = np.mean(1/t_out)
                om_out_sd[j] = np.std(1/t_out)
        
                # # control plots
                # plt.figure()
                # plt.subplot(1,2,1)
                # plt.plot(time, simDat[0,:],'r')
                # plt.plot(time, inp_dat+40,'g')
                # plt.xlim(0,10*T0[I])
        
                # # plt.vlines(t_peaks_inp, -2.5,-2.3, colors = 'g', alpha=0.3)
                # plt.vlines(t_peaks_out, -50,-45, colors = 'r', alpha=0.3)
                    
                # plt.subplot(1,2,2)
                # plt.scatter(np.zeros(len(t_out)),1/t_out)
                # plt.scatter(0,np.mean(1/t_out))
               
                # plt.suptitle('$\sigma=$'+"{:.5f}".format(sig)+' $x\omega_{unforced}$ ='+"{:.3f}".format(fold_unforced[j]))
        
            om_inp_sig.append(om_inp)
            om_out_sig.append(om_out)
            om_out_sd_sig.append(om_out_sd)
            
        FreqDiffPlot = np.asarray([om_inp_sig,om_out_sig,om_out_sd_sig])
        np.save('FreqDiffPlot_ML_'+str(I)+'.npy', FreqDiffPlot)
    

# plot
for I in I_range:

    rmin = np.log10(sections[I][0])
    rmax = np.log10(sections[I][1])
    fold_unforced = np.logspace(rmin,rmax,30) 

    FreqDiffPlot = np.load('FreqDiffPlot_ML_'+str(I)+'.npy')
    
    om_inp_sig,om_out_sig,om_out_sd_sig = FreqDiffPlot
    
    fig = plt.figure(figsize=(8.6*2/3*inCm,4.3*inCm))
    
    colors = ['k','r','m','b']
    
    for i in [2,1,0]:
        plt.fill_between(fold_unforced,om_out_sig[i]-om_inp_sig[i] + om_out_sd_sig[i],om_out_sig[i]-om_inp_sig[i]-om_out_sd_sig[i],color=colors[i],alpha=0.15)
        if i == 0:
            plt.plot(fold_unforced,om_out_sig[i]-om_inp_sig[i],color=colors[i],linestyle='dashed',label='$\sigma=$'+ str(sigma_range[i]))
        else:
            plt.plot(fold_unforced,om_out_sig[i]-om_inp_sig[i],color=colors[i],label='$\sigma=$'+ str(sigma_range[i]))
    
    plt.xscale('log')
    
    plt.yticks(fontsize=6)
    plt.xticks(fold_unforced[[0,14,29]],labels=np.round(fold_unforced[[0,14,29]], decimals=1),fontsize=6)
    plt.gca().xaxis.set_minor_locator(plt.NullLocator())
    
    plt.xlabel('$\omega$ / $\omega_{0}$')
    plt.ylabel('$\Omega - \omega$')              
    plt.subplots_adjust(top=0.825,
    bottom=0.345,
    left=0.2,
    right=0.84,
    hspace=0.145,
    wspace=0.18)
        
    # plt.legend()
    
    plt.savefig('FigS6e_'+str(I)+'.svg')
    
#%% Supplementary Figure 6f - ISI histograms

for I in range (2):
    t_end = 60*T0[I] 
    dt = 0.05
    npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  
    A = 5
    
    fold_unforced = np.logspace(-1,1,200)
    
    omega_range = fold_unforced*2*np.pi/T0[I]
    
    ISIs= []
    
    
    for i in tqdm(range(len(omega_range)), "Simulations for Figure 6f "+['(slow-fast regime)','(1-ghost cycle'][I]):

        omega = omega_range[i]
        params = [I_vals[I],v1,v2,v3,v4,gk,gl,gca,vca,vk,vl,phi,cm,A,omega]
    
        #transient phase
        solution = solve_ivp(mod.MorrisLecar_na, (-t_tr,0), IC, rtol=1.e-6, atol=1.e-6,
                              args=([params]), method='LSODA') 
        
        #post transient
        IC = solution.y[:,solution.y.shape[1]-1]
        solution = solve_ivp(mod.MorrisLecar_na, (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                              t_eval=time, args=([params]), method='LSODA') 
        
        peaks_out, _ = find_peaks(solution.y[0,:],distance=20,height=10, prominence=(35, None))
        t_peaks_out = time[peaks_out]
    
        T_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
            
        ISIs.append(T_out)      
     
    # plot histograms
    
    n = len(omega_range)
    
    isi_max = 0
    isi_min = T0[I]
    for i in range(n):
        if max(ISIs[i])>isi_max:
            isi_max = max(ISIs[i])
        if min(ISIs[i])<isi_min:
            isi_min = min(ISIs[i])
            
    if I == 1:
        isi_min = 0    
        
    nbins = 60
        
    histograms = []
    for i in range(n):
        h = np.histogram(ISIs[i],bins=nbins,range=(isi_min,isi_max))
        histograms.append(h)
    
    yAx = h[1]
        
    m = np.zeros((n,nbins))
    
    for i in range(n):
        m[i,:] = histograms[i][0]/np.sum(histograms[i][0])
    
    
    fig = plt.figure(figsize=(8.6*2/3*inCm,4*inCm))
    
    cw = plt.gca().imshow(m.T,cmap='binary',aspect=1.85,vmin=0, vmax=0.5)
    cbar = fig.colorbar(cw)
    cbar.set_label('$p$',rotation=0)
    
    plt.yticks(np.arange(yAx.shape[0])[1:yAx.shape[0]:6],labels=np.round(yAx[1:yAx.shape[0]:6], decimals=0),fontsize=6)
    plt.xticks(np.arange(fold_unforced.shape[0])[::20],labels=np.round(fold_unforced[::20], decimals=1),fontsize=6)
    
    plt.xlabel('$\omega$ / $\omega_{0}$')
    plt.ylabel('ISI (ms)')
    plt.subplots_adjust(top=1.0,
    bottom=0.11,
    left=0.185,
    right=0.99,
    hspace=0.2,
    wspace=0.2)
    
    plt.savefig('FigS6f_'+str(I)+'.png', dpi=400)
    plt.savefig('FigS6f_'+str(I)+'.svg')
