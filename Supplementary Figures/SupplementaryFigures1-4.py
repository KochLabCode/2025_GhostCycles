# -*- coding: utf-8 -*-
"""
This code reproduces Supplementary Figures 1-4 from the paper:

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
from scipy.integrate import solve_ivp, quad
from scipy.optimize import root_scalar
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
lbls = ['$VdP$','$VdP_{1g}$','$VdP_{2g}$']

# models and parameters

models = [mod.vanDerPol_na, mod.vanDerPol_1g_na, mod.vanDerPol_2g_na]
models_aut_trans = [mod.vanDerPol_na_aut, mod.vanDerPol_1g_na_aut, mod.vanDerPol_2g_na_aut]
jacobians = [mod.vdp_na_aut_jac, mod.vdp1g_na_aut_jac, mod.vdp2g_na_aut_jac]
    
a_bif = [7.131, 3.145]; eps_bif = 0.01
eps = 0.02; tau=16.5

#%% Determine periods in absence of forcing - T0

dt = 0.05; t_end = 1000; t_tr = 100; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  

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
    # plt.figure()
    # plt.plot(time,xGrad)
    if m_idx==0:
        peaks_out, _ = find_peaks(xGrad,height=0.15)
    else:
        peaks_out, _ = find_peaks(xGrad,height=0.33)
    t_peaks_out = time[peaks_out]

    T_out = np.array([ np.abs(t_peaks_out[i+1] - t_peaks_out[i]) for i in range(len(t_peaks_out)-1) ])
   
    T0.append(np.round(np.mean(T_out),0))
    

#%% Supplementary Figure 1 - dynamics on critical manifold: VdP vs VdP2g system

eps = 0.02; tau = 16.5
a_bif = [7.131, 3.145]; eps_bif = 0.01

para = [[0],[a_bif[1]-eps_bif]]


dt = 0.01
t_end = 2.9; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)


plt.figure(figsize=(8.6*inCm,5.3*inCm))

solution = solve_ivp(mod.vanDerPol_2g_CritFast, (0,t_end), [3], rtol=1.e-6, atol=1.e-6,
                      t_eval=time,args=([para[0]]), method='LSODA') 

plt.plot(time, solution.y[0,:], color=colors[0],lw=1.5,label='VdP')

t_end = 22.115; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)

solution = solve_ivp(mod.vanDerPol_2g_CritFast, (0,t_end), [3], rtol=1.e-6, atol=1.e-6,
                      t_eval=time,args=([para[1]]), method='LSODA') 

plt.plot(time, solution.y[0,:], color=colors[2],lw=1.5,label='VdP$_{2G}$')
        
plt.legend(frameon=False)
plt.xlabel('time (a.u.)'); plt.ylabel('x')
plt.tight_layout()
plt.savefig('FigureS1.svg')

#%% Supplementary Figure 2 - adjusted periods

eps = 0.02; tau = 16.5
a_bif = [7.131, 3.145]; eps_bif = 0.01

para = [[eps,tau],[eps,a_bif[0]-eps_bif],[eps,a_bif[1]-eps_bif]]


dt = 0.05    
t_end = 100; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)


plt.figure(figsize=(8.6*inCm,4*inCm))

solution = solve_ivp(mod.vanDerPol, (0,t_end), [2,-.5], rtol=1.e-6, atol=1.e-6,
                      t_eval=time,args=([para[0]]), method='LSODA') 

plt.plot(time, solution.y[0,:], color=colors[0],lw=1.5)

solution = solve_ivp(mod.vanDerPol_2g, (0,t_end), [2,.8], rtol=1.e-6, atol=1.e-6,
                      t_eval=time,args=([para[2]]), method='LSODA') 

plt.plot(time, solution.y[0,:], color=colors[2],lw=1.5)

    
plt.xlabel('time (a.u.)'); plt.ylabel('x')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.title('$T$')
plt.savefig('FigureS2.svg')

    
#%% Supplementary Figures 3 - timecourses of periodically forced VdP_1g system

dt = 0.05

A = 0.125

for m_idx in [1]:

    t_end = 10*T0[m_idx]; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1)  
    
    fold_unforced = np.logspace(-1,1,20)
    omega_range = fold_unforced*2*np.pi/T0[m_idx]
    
    plt.figure(figsize=(17.2*inCm,18*inCm))
    plt.suptitle(lbls[m_idx] + ', A = ' + str(A), fontsize=10)
    
    for i in tqdm(range(len(omega_range)),desc=f"Simulation Supplementary Figure {m_idx+2}"):
         
        omega = omega_range[i]
        para = [[eps,tau,A,omega],[eps,a_bif[0]-eps_bif,A,omega],[eps,a_bif[1]-eps_bif,A,omega]]
        
        plt.subplot(5,4,1+i)
        plt.title('$\omega = $'+"{:.2f}".format(fold_unforced[i])+'$\cdot \omega_{0}$',fontsize=8)
        
        #transient phase
        solution = solve_ivp(models[m_idx], (0,t_tr), np.array([0.1,0.1]), rtol=1.e-6, atol=1.e-6,
                              args=([para[m_idx]]), method='LSODA') 
        
        #post transient
        IC = solution.y[:,solution.y.shape[1]-1]
        solution = solve_ivp(models[m_idx], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                              t_eval=time, args=([para[m_idx]]), method='LSODA') 
        
        plt.plot(time, solution.y[0,:], color=colors[m_idx],lw=0.75)
        
        #post transient - perturbed
        LLE = fun.maxLyap('LSODA', models_aut_trans[m_idx], para[m_idx], np.array([0.1,0.1,0]), jacobians[m_idx], max(20*T0[m_idx], 500), dt, max(5*T0[m_idx], 100))

        if LLE > 0.02:
            solution = solve_ivp(models[m_idx], (0,t_end), IC+np.array([-0.01,0.01]), rtol=1.e-6, atol=1.e-6,
                                  t_eval=time, args=([para[m_idx]]), method='LSODA') 
            plt.plot(time, solution.y[0,:],'-',color=colors[m_idx],lw=2, alpha=0.4)
            plt.plot(time, A*np.sin(omega*time),'g',alpha=0.7,lw=0.75)
            
        plt.plot(time, A*np.sin(omega*time),'g',alpha=0.7,lw=0.75)
        plt.xlabel('time (a.u.)')
        plt.ylabel('x')
        
        plt.xticks([0,100,200,300])
        plt.ylim(-2.25,2.25)
        plt.xlim([0,t_end])
        if m_idx==2 and i >= 10:
            plt.xlim([0,150])
            plt.xticks([0,50,100,150])
            
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('FigureS'+str(2+m_idx)+'.svg')
    

#%% Supplementary Figure 4 - potential VdP vs VdP2G

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

newFig=plt.figure(figsize=(17.2/2*inCm,7*inCm))    

for k in range(2):
    
    A = 0; omega=1; t = 10
    
    para = [eps,[0,a_bif[1]-eps_bif][k],A,omega]
    
    yPot_range = np.linspace(0, 2.45, 150)
    V_values = np.array([Vy(y, t, para) for y in yPot_range])       

    plt.plot(yPot_range,V_values,['--k','-k'][k],label=['$\\alpha=0$','$\\alpha=3.135$'][k])   
    plt.xlabel('y')
    plt.ylabel('V(y)')
    
plt.legend()    
plt.tight_layout()
plt.savefig('FigureS4.svg', bbox_inches = "tight")    
    
