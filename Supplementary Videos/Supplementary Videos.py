# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:34:53 2024

@author: Daniel Koch
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.signal import find_peaks, argrelextrema
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from scipy.optimize import root_scalar

from tqdm import tqdm


import glob
from PIL import Image
import ffmpy 

"""
Note: for generation of *.mp4 files you need to install ffmpeg 
for ffmpy to work (https://www.gyan.dev/ffmpeg/builds)
see e.g. https://www.wikihow.com/Install-FFmpeg-on-Windows
"""
onlyGIF = True # set to "True" if you haven't installed ffmpeg ir only want to generate *.gif files

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

def vector_field_na(t,current_model,grid,dim):
  
    if dim=='3D':
        Xg,Yg,Zg=grid_ss
        
        x_range=Xg[0]
        y_range=Yg[:,0]
        z_range=Zg[0]
        
        Lx,Ly,Lz=len(x_range),len(y_range),len(z_range)
        U=np.zeros((Lx,Ly,Lz));V=np.zeros((Lx,Ly,Lz));W=np.zeros((Lx,Ly,Lz))
        
        for i in range(Lx):
            for j in range(Ly): 
                for k in range(Lz): 
                    U[i,j,k],V[i,j,k],W[i,j,k]=current_model(t,[Xg[i,j,k],Yg[i,j,k],Zg[i,j,k]])
        
        return U,V,W
    elif dim=='2D':
        
        Xg,Yg=grid_ss
        
        x_range=Xg[0]
        y_range=Yg[:,0]
        
        Lx,Ly=len(x_range),len(y_range)
        
        U=np.empty((Lx,Ly),np.float64);V=np.empty((Lx,Ly),np.float64)
        
        for i in range(Lx):
            for j in range(Ly):  
                U[i,j],V[i,j]=current_model(t,[Xg[i,j],Yg[i,j]])
        return U,V

# Nullclines etc
def xNC(x):
    return x**3/3-x

def yNC_na(y,p,t):
    eps,tau,A,omega = p
    return 0*y + A*np.sin(omega*t)  

def yNC_1g_na(y,p,t):
    eps,a,A,omega = p
    return a*((y+0.7)-1/3*(y+0.7)**3)*((1+np.tanh(y+0.7))/2)**10 + A*np.sin(omega*t)   

def yNC_2g_na(y,p,t):
    eps,a,A,omega = p
    return a*(y-1/3*y**3) + A*np.sin(omega*t)   

# def yNC_2g(y,p):
#     eps,a,A,omega = p
#     return a*(y-1/3*y**3)+A

# def jac_vdp2g(x,y,eps,alpha):
#     return np.array([[(1-x**2)/eps,1/eps],[-1,alpha*(1-y**2)]])

jacobians = [mod.jac_vdp,mod.jac_vdp1g, mod.jac_vdp2g]

# potential VdP_2G system

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


model_lbls = ['VdP','VdP$_{1g}$','VdP$_{2g}$']


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
    
#%% Supplementary Videos 1-4


for k in range(4): #[3,5,6]:#range(3,5):

    if k == 0:
        m = 0
        model_lbl = model_lbls[m]
        A=0.125; omega=1*2*np.pi/T0[m] # 1:1
        para = [eps,tau,A,omega]
        yNCfunc = yNC_na
        nth = 2
        t_end = 6*T0[m]
        dt = 0.05
        
    elif k == 1:
        m = 1
        model_lbl = model_lbls[m]
        A=0.125; omega=2.6*2*np.pi/T0[m] # 1:1
        para = [eps,a_bif[0]-eps_bif,A,omega]
        yNCfunc = yNC_1g_na
        nth = 2
        t_end = 4*T0[m]
        dt = 0.02
    
    elif k == 2:
        m = 1
        model_lbl = model_lbls[m]
        A=0.125; omega=0.5*2*np.pi/T0[m] # bursting
        para = [eps,a_bif[0]-eps_bif,A,omega]
        yNCfunc = yNC_1g_na
        nth = 2
        t_end = 4.5*T0[m]
        dt = 0.02
            
    elif k == 3:
        m = 2
        model_lbl = model_lbls[m]
        A=0.125; omega=0.5*2*np.pi/T0[m] # 1:1
        para = [eps,a_bif[1]-eps_bif,A,omega]
        yNCfunc = yNC_2g_na
        nth = 2
        t_end = 6*T0[m]
        dt = 0.025
        
    
    def current_model(t,z):
        return models[m](t,z,para)
    
    
    
    t_tr = 100; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1) 
    
    # Simulation 
       
    #transient phase
    solution = solve_ivp(models[m], (-t_tr,0), np.array([0.12,0.1]), rtol=1.e-6, atol=1.e-6,
                          args=([para]), method='LSODA') 
    
    #post transient
    IC = solution.y[:,solution.y.shape[1]-1]
    solution = solve_ivp(models[m], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para]), method='LSODA') 
        
    # plot trajectories, nullclines and create frames
    
    fileDirectory = os.path.dirname(os.path.abspath(__file__))
    path_vid = os.path.join(fileDirectory, 'Supplementary Video '+str(k+1))   
    
    if not os.path.exists(path_vid):
        os.makedirs(path_vid)
    
    simDat_red = solution.y[:,::nth]
    time_red = time[::nth]
    
    #full
    xmin=-2.25;xmax=2.25
    ymin=-2;ymax=2
    
    xmid=0
    ymid=0
    
    Ng=151
    x_range=np.linspace(xmin,xmax,Ng)
    y_range=np.linspace(ymin,ymax,Ng)
    grid_ss = np.meshgrid(x_range, y_range)
    
    Xg,Yg=grid_ss
      
    movWin = 10
    
    n_frames = int(time_red.size/movWin)-1
    
    fadingFactor = np.linspace(0.1,1,movWin)
    
    for i in tqdm(range(1,n_frames), "Simulation for Supplementary Video "+str(k+1)):
        
        newFig=plt.figure(figsize=(16*inCm,7*inCm))
        
        # Time course
        
        plt.subplot(1,2,1)
        
        plt.plot(time, solution.y[0,:], color=colors[m],lw=0.75)
        
        plt.title(model_lbl+', A = '+"{:.3f}".format(A)+', $\omega = $'+"{:.2f}".format(omega/(2*np.pi)*T0[m])+'$\cdot \omega_0$')
    
        plt.plot(time, A*np.sin(omega*time),'g',alpha=0.7,lw=0.75)
        plt.xlabel('time (a.u.)')
        plt.ylabel('x')
        plt.vlines(time_red[i*movWin],-2.5,3.25,colors='r',linestyles='dashed')
        
        plt.ylim(-2.5,2.5)
        plt.xlim([0,t_end])
        plt.gca().set_box_aspect(2/3)
        plt.gca().invert_yaxis()
        
    
        # Phase space
    
        plt.subplot(1,2,2)
        
        ax = plt.gca()
        ax.set_box_aspect(1/1)
            
        U,V=vector_field_na(time_red[i*movWin],current_model,grid_ss,dim='2D')  
        ax.streamplot(Xg,Yg,U,V,density=1,color=[0.75,0.75,0.75,0.5],arrowsize=1,linewidth=0.75)
        
        #x-NC
        ax.plot(x_range,xNC(x_range),'-k',lw=2,alpha=0.3)
    
        #y-NC default
        ax.plot(yNCfunc(y_range,para,time_red[i*movWin]),y_range,'--',color=colors[m],lw=2,alpha=0.66)
        
        # trajectory
        for j in range(movWin):
            ridx = (i-1)*movWin+j
            ax.plot(simDat_red[0,ridx:ridx+2], simDat_red[1,ridx:ridx+2],'-', color=colors[m],lw=1.5,alpha=fadingFactor[j])
            
            if j == movWin-1:
                            
                #intersection points
                xNC_arr = np.column_stack((x_range,xNC(x_range)))
                # yNC_arr = np.column_stack((yNC_2g(y_range,para),y_range))
                yNC_arr = np.column_stack((yNCfunc(y_range,para,time_red[i*movWin]),y_range))
                
                
                intersecPts = fun.intersections(xNC_arr,yNC_arr)
                
                for ii in range(intersecPts.shape[1]):
                    x,y=intersecPts[:,ii]
                    
                    eigenvalues, eigenvectors = np.linalg.eig(jacobians[m](x,y,para[0],para[1]))
         
                    if any(eigenvalues<0):
                        if any(eigenvalues>0): #saddle
                            ax.plot(x,y,'o', color='grey', mec='k', ms=7)
                        else: #stable FP
                            ax.plot(x,y,'o', color='black', mec='k',ms=7)
                    else: #stable FP
                        ax.plot(x,y,'o', color='white', mec='k',ms=7)
                        
                ax.plot(simDat_red[0,ridx], simDat_red[1,ridx],'o',ms=5, color=colors[m])
                
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_xlim(xmin,xmax)
                ax.set_ylim(ymin,ymax)
                ax.set_xticks([xmin,xmid,xmax])
                ax.set_yticks([ymin,ymid,ymax])
                ax.set_xticklabels([xmin,xmid,xmax])
                ax.set_yticklabels([ymin,ymid,ymax])
                
                plt.tight_layout()
                    
        plt.savefig(os.path.join(path_vid,str(i)+'.png'),dpi=200, bbox_inches = "tight")
        plt.close(newFig)
        
    # make videos 
                
    # frameDur = 10
    frameDur = 9/0.3*omega/(2*np.pi)
    
    fp_in = os.path.join(path_vid,"*.png")
    fp_out = os.path.join(path_vid,"Supplementary Video "+str(k+1)+".gif")
    
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) ) ]
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=frameDur, compress_level=9)
    
    # convert to video format
    if onlyGIF == False:
        fp_out_video = os.path.join(path_vid,"Supplementary Video "+str(k+1)+".mp4")
        
        ff = ffmpy.FFmpeg(
            inputs={fp_out: None},
            outputs={fp_out_video: '-crf 20 -pix_fmt yuv420p -vf "scale=-2:min(1080\,trunc(ih/2)*2)" '})
        ff.run()
    
    
#%% Supplementary Videos 5-7

m = 2

for k in range(3):
    
    A, fold_unforced = [(0.125,1),(0.125,4.75),(0.125,5.25)][k]

    model_lbl = model_lbls[m]
    
    omega=fold_unforced*2*np.pi/T0[m]
    para = [eps,a_bif[1]-eps_bif,A,omega]
    # para = [eps,0,A,omega]
    yNCfunc = yNC_2g_na
    nth =  1
    t_end = 4*T0[m]
    dt = 0.025
    
    def current_model(t,z):
        return models[m](t,z,para)
    
        
    
    t_tr = 100; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1) 
    
    # Simulation 
    
    #transient phase
    solution = solve_ivp(models[m], (-t_tr,0), np.array([0.12,0.1]), rtol=1.e-6, atol=1.e-6,
                          args=([para]), method='LSODA') 
    
    #post transient
    IC = solution.y[:,solution.y.shape[1]-1]
    solution = solve_ivp(models[m], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para]), method='LSODA') 
        
    # plot trajectories, nullclines and create frames
    
    fileDirectory = os.path.dirname(os.path.abspath(__file__))
    path_vid = os.path.join(fileDirectory, 'Supplementary Video '+str(k+5))   
    
    if not os.path.exists(path_vid):
        os.makedirs(path_vid)
    
    simDat_red = solution.y[:,::nth]
    time_red = time[::nth]
    
    #full
    xmin=-2.3;xmax=2.3
    ymin=-2;ymax=2
    
    xmid=0
    ymid=0
    
    Ng=151
    x_range=np.linspace(xmin,xmax,Ng)
    y_range=np.linspace(ymin,ymax,Ng)
    grid_ss = np.meshgrid(x_range, y_range)
    
    Xg,Yg=grid_ss
      
    movWin = 10
    
    n_frames = int(time_red.size/movWin)-1
    
    fadingFactor = np.linspace(0.1,1,movWin)
    
    for i in tqdm(range(31,n_frames), "Simulation for Supplementary Video "+str(k+5)):
        
        newFig=plt.figure(figsize=(14*inCm,7.2*inCm))
             
        if m == 2: 
            plt.subplot(1,2,1)
        
        plt.suptitle(model_lbl+', A = '+"{:.3f}".format(A)+', $\omega = $'+"{:.2f}".format(omega/(2*np.pi)*T0[m])+'$\cdot \omega_0$')
        
        ax = plt.gca()
        ax.set_box_aspect(1/1)
            
        U,V=vector_field_na(time_red[i*movWin],current_model,grid_ss,dim='2D')  
        ax.streamplot(Xg,Yg,U,V,density=1.7,color=[0.75,0.75,0.75,1],arrowsize=1,linewidth=0.85)
        
        #x-NC
        ax.plot(x_range,xNC(x_range),'-k',lw=2,alpha=0.3)
    
        #y-NC default
        ax.plot(yNCfunc(y_range,para,time_red[i*movWin]),y_range,'--',color=colors[m],lw=2,alpha=0.66)
        
        
    
        # trajectory
        for j in range(movWin):
            ridx = (i-1)*movWin+j
            ax.plot(simDat_red[0,ridx:ridx+2], simDat_red[1,ridx:ridx+2],'-', color=colors[m],lw=1.5,alpha=fadingFactor[j])
            
            if j == movWin-1:
                            
                #intersection points
                xNC_arr = np.column_stack((x_range,xNC(x_range)))
                yNC_arr = np.column_stack((yNCfunc(y_range,para,time_red[i*movWin]),y_range))
                
                intersecPts = fun.intersections(xNC_arr,yNC_arr)
                
                for ii in range(intersecPts.shape[1]):
                    x,y=intersecPts[:,ii]
    
                    eigenvalues, eigenvectors = np.linalg.eig(jacobians[m](x,y,para[0],para[1]))
                    
                    if any(eigenvalues<0):
                        if any(eigenvalues>0): #saddle
                            for e in range(2):
                                if eigenvalues[e]< 0:
                                    col = 'black'
                                
                                    # ax.arrow(0, 0, 1, 1, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
                                    x_ = np.linspace(x,x+eigenvectors[0,e],7)
                                    y_ = np.linspace(y,y+eigenvectors[1,e],7)
                                    
                                    ax.plot(x_,y_,'->',color=col,lw=2,alpha=0.6)
                                    
                                    x_ = np.linspace(x,x-eigenvectors[0,e],7)
                                    y_ = np.linspace(y,y-eigenvectors[1,e],7)
                                    
                                    ax.plot(x_,y_,'-<',color=col,lw=2,alpha=0.6)
                                    
                                    
                                if eigenvalues[e]> 0:
                                    col = 'purple'
                                    x_ = np.linspace(x,x+eigenvectors[0,e],5)
                                    y_ = np.linspace(y,y+eigenvectors[1,e],5)
                                    
                                    ax.plot(x_,y_,'-v',color=col,lw=2,alpha=0.6)
                                    
                                    x_ = np.linspace(x,x-eigenvectors[0,e],5)
                                    y_ = np.linspace(y,y-eigenvectors[1,e],5)
                                    
                                    ax.plot(x_,y_,'-^',color=col,lw=2,alpha=0.6)
                            ax.plot(x,y,'o', color='grey', mec='k', ms=10)
                        else: #stable FP
                            ax.plot(x,y,'o', color='black', mec='k',ms=10)
                    else: #stable FP
                        ax.plot(x,y,'o', color='white', mec='k',ms=10)
                        
                ax.plot(simDat_red[0,ridx], simDat_red[1,ridx],'o',ms=5, color=colors[m])             
                
                #inset
                axin1 = ax.inset_axes([0.775, 0.025, 0.2, 0.2])
                axin1.plot(solution.y[0,:], solution.y[1,:],'-',lw=0.25, color=colors[m])
                axin1.plot(simDat_red[0,ridx], simDat_red[1,ridx],'o',ms=2, color=colors[m])
                axin1.set_yticks([])
                axin1.set_xticks([])
    
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(1.6,2.4); ax.set_ylim(0.4,1.6)
        ax.plot(10,10,'->',color='black',lw=2,alpha=0.6, label= 'eig.vector stable saddle direction')
        ax.plot(10,10,'-v',color='purple',lw=2,alpha=0.6, label= 'eig.vector unstable saddle direction')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),fontsize=6, frameon=False)
        
        t = time_red[i*movWin]
        if m == 2: 
            plt.subplot(1,2,2)
            ridx = i*movWin-1
           
            yPot_range = np.linspace(0, 2, 150)
            V_values = np.array([Vy(y, t, para) for y in yPot_range])
            
            v_imin,v_imax = argrelextrema(V_values, np.less), argrelextrema(V_values, np.greater)
            
           
            plt.plot(yPot_range,V_values,'-k')
            plt.plot(yPot_range[v_imin],V_values[v_imin],'o', color='black', mec='k')
            plt.plot(yPot_range[v_imax],V_values[v_imax],'o', color='grey', mec='k')
            
            
            if simDat_red[0,ridx] > 1.5 and simDat_red[1,ridx] > 0:
                plt.plot(simDat_red[1,ridx], Vy(simDat_red[1,ridx], t, para),'o', color=colors[m])
            
            plt.ylim(0,1.25)
            plt.xlabel('y')
            plt.ylabel('V(y)')
        
        plt.tight_layout()
           
        plt.savefig(os.path.join(path_vid,str(i)+'.png'),dpi=200, bbox_inches = "tight")
        plt.close(newFig)
   
        
    # make gif file  
    
    frameDur = 9/0.3*fold_unforced
        
    fp_in = os.path.join(path_vid,"*.png")
    fp_out = os.path.join(path_vid,'Supplementary Video '+str(k+5)+".gif")
        
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]) ) ]
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=frameDur, compress_level=9)
    
    # convert to video format
    
    if onlyGIF == False:
        fp_out_video = os.path.join(path_vid,'Supplementary Video '+str(k+5)+".mp4")
        
        ff = ffmpy.FFmpeg(
            inputs={fp_out: None},
            outputs={fp_out_video: '-crf 20 -pix_fmt yuv420p -vf "scale=-2:min(1080\,trunc(ih/2)*2)" '})
        ff.run()
    
#%% Supplementary Video 8

n = 0

for k in range(2):

    if k == 0:
        m = 0
        model_lbl = model_lbls[m]
        A=1.5; omega=0.29*2*np.pi/T0[m] # 1:1
        para = [eps,tau,A,omega]
        yNCfunc = yNC_na
        nth = 3
        t_end = 8*T0[m]
        dt = 0.05
    elif k == 1:
        m = 0
        model_lbl = model_lbls[m]
        A=1.5; omega=2*2.5*np.pi/T0[m] # 1:1
        para = [eps,tau,A,omega]
        yNCfunc = yNC_na
        nth = 1
        t_end = 3.5*T0[m]
        dt = 0.05
    
    
    def current_model(t,z):
        return models[m](t,z,para)
    
    
    
    t_tr = 150; npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1) 
    
    # Simulation 
    
    #transient phase
    solution = solve_ivp(models[m], (-t_tr,0), np.array([0,np.sqrt(3)]), rtol=1.e-6, atol=1.e-6,
                          args=([para]), method='LSODA') 
    
    #post transient
    IC = solution.y[:,solution.y.shape[1]-1]
    solution = solve_ivp(models[m], (0,t_end), IC, rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([para]), method='LSODA') 
        
    # plot trajectories, nullclines and create frames
    
    fileDirectory = os.path.dirname(os.path.abspath(__file__))
    path_vid = os.path.join(fileDirectory, 'Supplementary Video 8')   
    
    if not os.path.exists(path_vid):
        os.makedirs(path_vid)
    
    simDat_red = solution.y[:,::nth]
    time_red = time[::nth]
    
    #full
    xmin=-2.25;xmax=2.25
    ymin=-2;ymax=2
    
    xmid=0
    ymid=0
    
    Ng=151
    x_range=np.linspace(xmin,xmax,Ng)
    y_range=np.linspace(ymin,ymax,Ng)
    grid_ss = np.meshgrid(x_range, y_range)
    
    Xg,Yg=grid_ss
      
    movWin = 10
    
    n_frames = int(time_red.size/movWin)-1
    
    fadingFactor = np.linspace(0.1,1,movWin)
    
    for i in tqdm(range(1,n_frames), "Simulation for Supplementary Video 8 " + ["(part 1)","(part 2)"][k]):
    
        newFig=plt.figure(figsize=(16*inCm,7*inCm))
        
        # Time course
        
        plt.subplot(1,2,1)
        
        plt.plot(time, solution.y[0,:], color=colors[m],lw=0.75)
        
        plt.title(model_lbl+', A = '+"{:.3f}".format(A)+', $\omega = $'+"{:.2f}".format(omega/(2*np.pi)*T0[m])+'$\cdot \omega_0$')
    
        plt.plot(time, A*np.sin(omega*time),'g',alpha=0.7,lw=0.75)
        plt.xlabel('time (a.u.)')
        plt.ylabel('x')
        plt.vlines(time_red[i*movWin],-2.5,3.25,colors='r',linestyles='dashed')
        
        plt.ylim(-2.5,2.5)
        plt.xlim([0,t_end])
        plt.gca().set_box_aspect(2/3)
        plt.gca().invert_yaxis()
        
        # Phase space
    
        plt.subplot(1,2,2)
        
        ax = plt.gca()
        ax.set_box_aspect(1/1)
            
        U,V=vector_field_na(time_red[i*movWin],current_model,grid_ss,dim='2D')  
        ax.streamplot(Xg,Yg,U,V,density=1,color=[0.75,0.75,0.75,0.5],arrowsize=1,linewidth=0.75)
        
        #x-NC
        ax.plot(x_range,xNC(x_range),'-k',lw=2,alpha=0.3)
    
        #y-NC default
        ax.plot(yNCfunc(y_range,para,time_red[i*movWin]),y_range,'--',color=colors[m],lw=2,alpha=0.66)      
        
        # trajectory
        for j in range(movWin):
            ridx = (i-1)*movWin+j
            ax.plot(simDat_red[0,ridx:ridx+2], simDat_red[1,ridx:ridx+2],'-', color=colors[m],lw=1.5,alpha=fadingFactor[j])
            
            if j == movWin-1:
                            
                #intersection points
                xNC_arr = np.column_stack((x_range,xNC(x_range)))
                yNC_arr = np.column_stack((yNCfunc(y_range,para,time_red[i*movWin]),y_range))
                
                
                intersecPts = fun.intersections(xNC_arr,yNC_arr)
                
                for ii in range(intersecPts.shape[1]):
                    x,y=intersecPts[:,ii]
                    
                    eigenvalues, eigenvectors = np.linalg.eig(jacobians[m](x,y,para[0],para[1]))
                    
                    if any(eigenvalues<0):
                        if any(eigenvalues>0): #saddle
                            ax.plot(x,y,'o', color='grey', mec='k', ms=7)
                        else: #stable FP
                            ax.plot(x,y,'o', color='black', mec='k',ms=7)
                    else: #stable FP
                        ax.plot(x,y,'o', color='white', mec='k',ms=7)
                        
                ax.plot(simDat_red[0,ridx], simDat_red[1,ridx],'o',ms=5, color=colors[m])
                
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_xlim(xmin,xmax)
                ax.set_ylim(ymin,ymax)
                ax.set_xticks([xmin,xmid,xmax])
                ax.set_yticks([ymin,ymid,ymax])
                ax.set_xticklabels([xmin,xmid,xmax])
                ax.set_yticklabels([ymin,ymid,ymax])
                
                plt.tight_layout()
                            
        plt.savefig(os.path.join(path_vid,str(n)+'.png'),dpi=200, bbox_inches = "tight")
        plt.close(newFig)
        n+=1
        
    if k==0:
        for j in range(30):
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(16*inCm,7*inCm))
            
            # Set the background color to white
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # Add the text at the center
            ax.text(0.5, 0.5, "Higher forcing frequency $\omega$", fontsize=12, color='black', ha='center', va='center')
            
            # Remove the axes for a clean look
            ax.axis('off')
            
            plt.tight_layout()
            
            # Display the figure
            plt.savefig(os.path.join(path_vid,str(n)+'.png'),dpi=200, bbox_inches = "tight")
            plt.close(fig)       
            n+=1
    
    
frameDur = 60  # Duration in ms (adjust as needed, e.g., 40 ms = 25 fps)

# Paths
fp_in = os.path.join(path_vid, "*.png")
fp_out = os.path.join(path_vid, "Supplementary Video 8.gif")
fp_out_video = os.path.join(path_vid, "Supplementary Video 8.mp4")

# Load and sort images
images = sorted(glob.glob(fp_in), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
img, *imgs = [Image.open(f) for f in images]

# Save GIF
img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=frameDur, compress_level=9)

if onlyGIF == False:
    # Convert to video
    ff = ffmpy.FFmpeg(
        inputs={fp_out: None},
        outputs={fp_out_video: '-crf 20 -pix_fmt yuv420p -vf "scale=-2:min(1080\\,trunc(ih/2)*2)"'}
    )
    ff.run()