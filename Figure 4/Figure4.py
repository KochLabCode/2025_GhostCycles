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
from scipy.signal import find_peaks, argrelextrema
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from scipy.optimize import root_scalar

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
    
eps = 0.02; tau = 16.5
a_bif = [7.131, 3.145]; eps_bif = 0.01


# Potential for VdP_2G system

# Define the critical manifold function h(y) by solving y = x^3/3 - x for x numerically
def h(y):
    # Define the equation f(x) = x^3/3 - x - y
    def f(x):
        return x**3 / 3 - x - y
    
    # Use root finding to solve f(x) = 0 near an initial guess
    # Here we assume x ~ y as an initial guess to start the solver
    sol = root_scalar(f, bracket=[-3, 3], method='brentq')  # bracket chosen to cover expected range
    return sol.root

# Define the function for the derivative of V(y, t) with respect to y
def dV_dy(y, t, p):
    eps,alpha,A,omega = p
    return h(y) - alpha * (y - y**3 / 3) - A * np.sin(omega * t)

# Define the potential function V(y, t) by integrating dV_dy from y=0 to y
def Vy(y, t, p):
    V_y, _ = quad(dV_dy, 0, y, args=(t,p))
    return V_y


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
    
#%% Figure 4

model_lbls = ['VdP','VdP$_{1g}$','VdP$_{2g}$']

m_idx = 2
newFig=plt.figure(figsize=(17.2*inCm,12.5*inCm))    

AmpFreqPairs = [(0.125,1),(0.125,4.75),(0.125,5.25)]

frames = [[177,199,79,109],[46,54,56,67], [57,65,69,78]]

for k in range(3):

    A, fold_unforced = AmpFreqPairs[k]
    model_lbl = model_lbls[m_idx]
    
    omega=fold_unforced*2*np.pi/T0[m_idx]
    para = [eps,a_bif[1]-eps_bif,A,omega]
    yNCfunc = mod.yNC_2g_na
    nth =  1
    t_end = 1.5*T0[m_idx]
    dt = 0.025
    
    def current_model(t,z):
        return models[m_idx](t,z,para)
    
    t_tr = 100; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1) 
    
    # Simulation 
     
    #transient phase
    solution = solve_ivp(models[m_idx], (0,t_tr), np.array([0.12,0.1]), rtol=1.e-6, atol=1.e-6,
                          args=([para]), method='LSODA') 
    
    #post transient
    IC = solution.y[:,solution.y.shape[1]-1]
    solution = solve_ivp(models[m_idx], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para]), method='LSODA') 
        
    # plot trajectories
    
    simDat_red = solution.y[:,::nth]
    time_red = time[::nth]
    
    xmin=-2.3;xmax=2.3
    ymin=-2;ymax=2
    
    xmid=0
    ymid=0
    
    Ng=151
    x_range=np.linspace(xmin,xmax,Ng)
    y_range=np.linspace(ymin,ymax,Ng)
    grid_ss = np.meshgrid(x_range, y_range)
    
    Xg,Yg=grid_ss
      
    movWin = 8
    
    n_frames = int(time_red.size/movWin)-1
    
    fadingFactor = np.linspace(0.1,1,movWin)
    
        
    k_col = 0
 
    for i in frames[k]:
        
        sp_idx = 1 + k*4 + k_col
        
        plt.subplot(3,4,sp_idx)
        
     
        t = time_red[i*movWin]
        
        ax = plt.gca()
        ax.set_box_aspect(1/1.2)
        ridx = i*movWin-1
       
        
        yPot_range = np.linspace(0, 2, 150)
        V_values = np.array([Vy(y, t, para) for y in yPot_range])
            
        v_imin,v_imax = argrelextrema(V_values, np.less), argrelextrema(V_values, np.greater)
        
       
        plt.plot(yPot_range,V_values,'-k')
        plt.plot(yPot_range[v_imin],V_values[v_imin],'o', color='black', mec='k')
        plt.plot(yPot_range[v_imax],V_values[v_imax],'o', color='grey', mec='k')
        
        
        if simDat_red[0,ridx] > 1.7 and simDat_red[1,ridx] > 0:
            plt.plot(simDat_red[1,ridx], Vy(simDat_red[1,ridx], t, para),'o', color=colors[m_idx],ms=4.5)
        
        plt.ylim(0,1.1)
        
        if k == 2:
            plt.xlabel('y')
        else:
            plt.xticks([])
        if k_col == 0:
            plt.ylabel('V(y,t)')
        else:
            plt.yticks([])

        #inset
        axin1 = ax.inset_axes([0.675, 0.17, 0.3, 0.2])
        axin1.plot(np.linspace(0,2*np.pi), np.sin(np.linspace(0,2*np.pi)),'-g',lw=1.5)
        axin1.plot([omega*t%(2*np.pi), omega*t%(2*np.pi)],[-1,1],'--r',lw=1.5)

        axin1.set_xticks([0,np.pi,2*np.pi],['0','$\\pi$', '2$\\pi$'],fontsize=7)
        axin1.set_yticks([-1,0,1],['-1','0','1'],fontsize=7)
        k_col += 1
        
        
plt.tight_layout()
       
    
plt.savefig('Fig4.svg', bbox_inches = "tight")


   