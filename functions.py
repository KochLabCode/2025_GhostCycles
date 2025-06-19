# -*- coding: utf-8 -*-
"""
This code features various functions used in other codes for the paper:

    Daniel Koch, Ulrike Feudel, Aneta Koseska (2025):
    Criticality governs response dynamics and entrainment of periodically forced ghost cycles.
    Physical Review E, XX: XXXX-XXXX.
    
Copyright: Daniel Koch
"""

import numpy as np
import shapely
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def get_rcparams():
    params = {'legend.fontsize': 10,
              'axes.labelsize': 10,
              'axes.labelpad' : 15,
              'axes.titlesize':12,
              'xtick.labelsize':7,
              'ytick.labelsize':7,
               'text.usetex': False
              }
    return params

def RK4_na_noisy(f,p,ICs,t0,dt,t_end, noiseVector, sigma=0, naFun = None,naFunParams = None,**kwargs):     # args: ODE system, parameters, initial conditions, starting time t0, dt, number of steps
        
        # using Euler-Maruyama method (https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
        
        if 'multiplicative_noise' in kwargs:
            multNoise = kwargs['multiplicative_noise']
            # print(multNoise)
        else:
            multNoise = False

        steps = int((t_end-t0)/dt)
        dims = tuple([steps]+list(ICs.shape))
        
        x = np.zeros(dims)
        t = np.zeros(steps,dtype=float)
        x[0] = ICs
        t[0] = t0
        
        if naFun != None and naFunParams != None:
            for i in range(1,steps):
                
                t[i] = t0 + i*dt
                # RK4 algorithm
                k1 = f(x[i-1],t[i-1],p,naFun,naFunParams)*dt
                k2 = f(x[i-1]+k1/2,t[i-1],p,naFun,naFunParams)*dt
                k3 = f(x[i-1]+k2/2,t[i-1],p,naFun,naFunParams)*dt
                k4 = f(x[i-1]+k3,t[i-1],p,naFun,naFunParams)*dt
                x_next = x[i-1] + (k1+2*k2+2*k3+k4)/6
                if multNoise == False:
                    dW=sigma*np.sqrt(dt)*np.random.normal(size=x_next.shape)
                    x[i,:] = x_next + dW*noiseVector 
                else:
                    dW=sigma*np.sqrt(dt*x[i-1])*np.random.normal(size=x_next.shape)
                    x[i,:] = x_next + dW*noiseVector 
                    
        else:
            for i in range(1,steps):
                t[i] = t0 + i*dt
                # RK4 algorithm
                k1 = f(x[i-1],t[i-1],p)*dt
                k2 = f(x[i-1]+k1/2,t[i-1],p)*dt
                k3 = f(x[i-1]+k2/2,t[i-1],p)*dt
                k4 = f(x[i-1]+k3,t[i-1],p)*dt
                x_next = x[i-1] + (k1+2*k2+2*k3+k4)/6
                if multNoise == False:
                    dW=sigma*np.sqrt(dt)*np.random.normal(size=x_next.shape)
                    x[i,:] = x_next + dW*noiseVector
                else:
                    dW=sigma*np.sqrt(dt*x[i-1])*np.random.normal(size=x_next.shape)
                    x[i,:] = x_next + dW*noiseVector 
            
        return t,x.T    
    
def vector_field(current_model,grid_ss,dim):
  
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
                    U[i,j,k],V[i,j,k],W[i,j,k]=current_model(0,[Xg[i,j,k],Yg[i,j,k],Zg[i,j,k]])
        
        return U,V,W
    elif dim=='2D':
        
        Xg,Yg=grid_ss
        
        x_range=Xg[0]
        y_range=Yg[:,0]
        
        Lx,Ly=len(x_range),len(y_range)
        
        U=np.empty((Lx,Ly),np.float64);V=np.empty((Lx,Ly),np.float64)
        
        for i in range(Lx):
            for j in range(Ly):  
                U[i,j],V[i,j]=current_model(0,[Xg[i,j],Yg[i,j]])
        return U,V
    

def vector_field_na(t,current_model,grid_ss,dim):
  
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
    
def maxLyap(method, sys, params, ICs, jac, t_end, dt = 0.01, t_transient = 100, eps = 1e-8, **kwargs):
    
    if 'plotFit' in kwargs:
        plotFit = kwargs['plotFit']
        if type(plotFit) != bool:
            print('plotFit not boolean. Set plotFit to \'False\'')
            plotFit = False
    else:
        plotFit = False
        
        
    dim = ICs.shape[0]

    if t_transient > 0:
        npts = int(t_transient/dt); time = np.linspace(0,t_transient,npts+1)  
        sol_transient = solve_ivp(sys, (0,t_transient), ICs, rtol=1.e-6, atol=1.e-6,
                              t_eval=time, args=([params]), method=method) 

        ICs_ = sol_transient.y[:,sol_transient.y.shape[1]-1]
    else:
        ICs_ = ICs
        
    def coupled_sys(t,x0,p):      
        x = sys(t,x0[:dim],p)
        x_ = np.matmul(jac(x0[:dim],params),x0[dim:])
        return np.hstack((x,x_))
    
     
    ICs_cpld = np.hstack((ICs_,ICs_+ np.random.normal(-1,1,size=dim)*eps))
    
    npts = int(t_end/dt); time = np.linspace(0,t_end,npts+1) 
    sol_tanSpace = solve_ivp(coupled_sys, (0,t_end), ICs_cpld, rtol=1.e-6, atol=1.e-6,
                          t_eval=time, args=([params]), method=method) 
    
    
    norm_dY = np.linalg.norm(sol_tanSpace.y,axis=0)
    log_norm_dY = np.log(norm_dY)
    
    
    lineFit = np.polyfit(time, log_norm_dY, 1)
    line_fn = np.poly1d(lineFit)
    
    if plotFit == True:       
        plt.figure()
        plt.plot(time,log_norm_dY)
        plt.plot(time, line_fn(time),'-r')
        plt.xlabel('time')
        plt.ylabel('||$\delta y$||')
        plt.show()
    
    return lineFit[0]


def euklDist(x,y): #calculates the euclidean distance between vectors x and y
    if x.shape != y.shape:
        print("Euklidean distance cannot be calculated as arrays have different dimensions")
    elif x.ndim == 1:
        EDsum = 0
        for i in range(0,x.size):
            EDsum = EDsum + np.square(x[i]-y[i])
        return np.sqrt(EDsum)
    else:
        print("Unsuitable arguments for Euklidean distance calculation.")

def euklideanVelocity(x,dt):
    v = np.array([])
    n = x.shape[0]
    for i in range(1,n):
        d = euklDist(x[i,:],x[i-1,:])
        v = np.append(v, d/dt)
    return v

def intersections(a,b):
    
    # determines intersections between two lines a and b
    
    a_ = shapely.LineString(a)
    b_ = shapely.LineString(b)
    intsecs = shapely.intersection(a_,b_).wkt

    start = intsecs.find('(')
    stop = intsecs.find(')')

    valStr = intsecs[start+1:stop]
    valStr = valStr.replace(',','')

    vals = np.fromstring(valStr, dtype=float, sep=' ')
    vals = np.reshape(vals, (int(len(vals)/2),2)).T
    
    return vals
