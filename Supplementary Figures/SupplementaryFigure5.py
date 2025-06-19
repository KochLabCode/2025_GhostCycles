# -*- coding: utf-8 -*-
"""
This code reproduces Supplementary Figure 5 from the paper:

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


#%% Supplementary Figure 5 - Timecourses periodically forced ML system

colors = ['#0000FF','m','#FF5555']
I=1
A = 5
t_end = 10*T0[I]
dt = 0.05
npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  

m_labels = ['slow-fast regime', '1-ghost cycle regime']

fold_unforced = np.logspace(-1,1,20)
omega_range = fold_unforced*2*np.pi/T0[I]

plt.figure(figsize=(17.2*inCm,18*inCm))
plt.suptitle(m_labels[I] + ', A = ' + str(A) + ' mV', fontsize=10)

for i in tqdm(range(len(omega_range)), "Simulations for Figure S5"):

    
    omega = omega_range[i]
    params = [I_vals[I],v1,v2,v3,v4,gk,gl,gca,vca,vk,vl,phi,cm,A,omega]
    
    plt.subplot(5,4,1+i)
    plt.title('$\omega =$'+"{:.2f}".format(fold_unforced[i])+'$\\cdot \omega_0$',fontsize=8)
     
    #transient phase
    solution = solve_ivp(mod.MorrisLecar_na, (-T0[I],0), IC, rtol=1.e-6, atol=1.e-6,
                          args=([params]), method='LSODA') 
    
    #post transient
    IC = solution.y[:,solution.y.shape[1]-1]
    solution = solve_ivp(mod.MorrisLecar_na, (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([params]), method='LSODA') 
     
    plt.plot(time, solution.y[0,:], color=colors[I],lw=0.75)
        
    plt.ylim(-50,50)
    plt.plot(time, A*np.sin(omega*time),'g',alpha=0.7,lw=0.75)
    
    plt.xlabel('time (ms)')
    plt.ylabel('V (mV)')
    
    
    plt.xlim([0,t_end])
    if I==2 and i >= 10:
        plt.xlim([0,500])

plt.tight_layout()

plt.savefig('FigS5.svg')
