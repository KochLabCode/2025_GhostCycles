# -*- coding: utf-8 -*-
"""
This code features all models and model-related functions (jacobians, nullclines) used in the paper:

    Daniel Koch, Ulrike Feudel, Aneta Koseska (2025):
    Criticality governs response dynamics and entrainment of periodically forced ghost cycles.
    Physical Review E, XX: XXXX-XXXX.
    
Copyright: Daniel Koch
"""
import numpy as np

# Autonomous versions of Van der Pol based models

def vanDerPol(t,z,para):
    eps, tau = para    
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy=-z[0]
    return np.array([dx, dy])/tau

def vanDerPol_1g(t,z,para):
    eps,alpha=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy=-z[0] + alpha*((z[1]+0.7)-1/3*(z[1]+0.7)**3)*((1+np.tanh((z[1]+0.7)))/2)**10      
    return np.array([dx, dy])

def vanDerPol_2g(t,z,para):
    eps,alpha=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy = -z[0] + alpha*(z[1] - 1/3*z[1]**3)
    return np.array([dx, dy])

# # Periodically forced versions of Van der Pol based models

def vanDerPol_na(t,z,para):
    eps,tau,A,omega=para   
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy=-z[0] + A*np.sin(omega*t)
    return np.array([dx, dy])/tau

def vanDerPol_1g_na(t,z,para):
    eps,alpha,A,omega=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy=-z[0] + alpha*((z[1]+0.7)-1/3*(z[1]+0.7)**3)*((1+np.tanh((z[1]+0.7)))/2)**10 + A*np.sin(omega*t)    
    return np.array([dx, dy])

def vanDerPol_2g_na(t,z,para):
    eps,alpha,A,omega=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy = -z[0] + alpha*(z[1] - 1/3*z[1]**3) + A*np.sin(omega*t)
    return np.array([dx, dy])


def vanDerPol_2g_na_alt(t,z,para):
    eps,alpha,beta,A,omega=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy=-z[0] + alpha*((z[1]+0.7+beta)-1/3*(z[1]+0.7+beta)**3)*((1+np.tanh((z[1]+0.7+beta)))/2)**10 + A*np.sin(omega*t)
    return np.array([dx, dy])


# Non-autonomous systems transformed into autonomous formulation via additional variable z

def vanDerPol_na_aut(t,z,para):
    eps,tau,A,omega=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])/tau
    dy=-z[0]/tau + A*np.sin(z[2])/tau
    dz = omega
    return np.array([dx, dy, dz])

def vanDerPol_1g_na_aut(t,z,para):
    eps,alpha,A,omega=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy=-z[0] + alpha*((z[1]+0.7)-1/3*(z[1]+0.7)**3)*((1+np.tanh((z[1]+0.7)))/2)**10 + A*np.sin(z[2])
    dz = omega
    return np.array([dx, dy, dz])

def vanDerPol_2g_na_aut(t,z,para):
    eps,alpha,A,omega=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy = -z[0] + alpha*(z[1] - 1/3*z[1]**3) + A*np.sin(z[2])
    dz = omega
    return np.array([dx, dy, dz])

# Jacobians as functions

# autonomous versions (for linear stability analysis)

def jac_vdp(x,y,eps,tau):
    return np.array([[(1-x**2)/eps,1/eps],[-1/tau,0]])

def jac_vdp1g(x,y,eps,alpha):
    h = alpha*(1 - (y + 0.7)**2)*(np.tanh(y + 0.7)/2 + 1/2)**10 + alpha*(5 - 5*np.tanh(y + 0.7)**2)*(np.tanh(y + 0.7)/2 + 1/2)**9*(y - 1/3*(y + 0.7)**3 + 0.7)
    return np.array([[(1-x**2)/eps, 1/eps],[-1,h]])

def jac_vdp2g(x,y,eps,alpha):
    return np.array([[(1-x**2)/eps,1/eps],[-1,alpha*(1-y**2)]])


# non-autonomous versions turned into autonomous (for calculation of Lyapunov exponents)
def vdp_na_aut_jac(x, params):
    eps,tau,A,omega = params 
    J = np.array([
        [(1-x[0]**2)/(eps*tau), 1/(eps*tau), 0],
        [-1/tau, 0, A*np.cos(x[2])/tau],
        [0,0,0]
        ])
    return J

def vdp1g_na_aut_jac(x, params):
    eps,alpha,A,omega = params
    h = alpha*(1 - (x[1] + 0.7)**2)*(np.tanh(x[1] + 0.7)/2 + 1/2)**10 + alpha*(5 - 5*np.tanh(x[1] + 0.7)**2)*(np.tanh(x[1] + 0.7)/2 + 1/2)**9*(x[1] - 1/3*(x[1] + 0.7)**3 + 0.7)
    J = np.array([
        [(1-x[0]**2)/eps, 1/eps, 0],
        [-1, h, A*np.cos(x[2])],
        [0,0,0]
        ])
    return J

def vdp2g_na_aut_jac(x, params):
    eps,alpha,A,omega = params
    J = np.array([
        [(1-x[0]**2)/eps, 1/eps, 0],
        [-1, alpha*(1-x[1]**2), A*np.cos(x[2])],
        [0,0,0]
        ])
    return J

# Constantly forced versions of Van der Pol based models

def vanDerPol_constForce(t,z,para):
    eps,tau,A,omega=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy=-z[0] + A
    return np.array([dx, dy])/tau

def vanDerPol_1g_constForce(t,z,para):
    eps,alpha,A,omega=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy=-z[0] + alpha*((z[1]+0.7)-1/3*(z[1]+0.7)**3)*((1+np.tanh((z[1]+0.7)))/2)**10 + A
    return np.array([dx, dy])

def vanDerPol_2g_constForce(t,z,para):
    eps,alpha,A,omega=para
    dx=1/eps*(z[1]-1/3*z[0]**3+z[0])
    dy = -z[0] + alpha*(z[1] - 1/3*z[1]**3) + A
    return np.array([dx, dy])

# Nullclines of Van der Pol based models

def xNC(x):
    return x**3/3-x

def yNC_vdp(y,p):
    A = p
    return 0*y+A

def yNC_1g(y,p):
    eps,a,A,omega = p
    return a*((y+0.7)-1/3*(y+0.7)**3)*((1+np.tanh(y+0.7))/2)**10+A

def yNC_2g(y,p):
    eps,a,A,omega = p
    return a*(y-1/3*y**3)+A

def yNC_vdp_na(y,p,t):
    eps,tau,A,omega = p
    return 0*y + A*np.sin(omega*t)  

def yNC_1g_na(y,p,t):
    eps,a,A,omega = p
    return a*((y+0.7)-1/3*(y+0.7)**3)*((1+np.tanh(y+0.7))/2)**10 + A*np.sin(omega*t)  

def yNC_2g_na(y,p,t):
    eps,a,A,omega = p
    return a*(y-1/3*y**3) + A*np.sin(omega*t)   


# Dynamics on critical manifold of VdP2G

def vanDerPol_2g_CritFast(t,z,para):
    alpha=para
    z=(z-alpha*(z**3/3-z-(z**3/3-z)**3/3))/(1-z**2)
    return z
    

# Autonomous version of Morris-Lecar system as described in: https://doi.org/10.1016/j.neucom.2005.03.006

def MorrisLecar(t,z,para):
    
    Iext,v1,v2,v3,v4,gk,gl,gca,vca,vk,vl,phi,cm=para
    
    minf=0.5*(1+np.tanh((z[0]-v1)/v2))
    ninf=0.5*(1+np.tanh((z[0]-v3)/v4))
    taun=1/(phi*np.cosh((z[0]-v3)/(2*v4)))
    
    dV=(-gl*(z[0]-vl)-gca*minf*(z[0]-vca)-gk*z[1]*(z[0]-vk)+Iext)/cm
    dN=(ninf-z[1])/taun
    
    return np.array([dV, dN])

# Non-autonomous version of Morris-Lecar system

def MorrisLecar_na(t,z,para):
    
    Iext,v1,v2,v3,v4,gk,gl,gca,vca,vk,vl,phi,cm,A,omega=para
    
    minf=0.5*(1+np.tanh((z[0]-v1)/v2))
    ninf=0.5*(1+np.tanh((z[0]-v3)/v4))
    taun=1/(phi*np.cosh((z[0]-v3)/(2*v4)))
    
    dV=(-gl*(z[0]-vl)-gca*minf*(z[0]-vca)-gk*z[1]*(z[0]-vk)+(Iext+ A*np.sin(omega*t)))/cm
    dN=(ninf-z[1])/taun
    
    return np.array([dV, dN])

# Autonomous version of van Nes slow/fast model 2 in: https://link.springer.com/article/10.1007/s10021-006-0176-0
def vanNes2007_model2(t,z,para): 
    
    pE,rV,hE,gammaE,pwin,tau,g,pSOD,HSOD,lSOD,hV,kV=para
    
    z[0] = np.max([z[0],0])
    z[1] = np.max([z[1],0])
    
    E=gammaE*z[1]*hV/(hV+z[0])
    SOD=kV*z[0]/lSOD

    dv=rV*z[0]*(1-(z[0]*(hE**pE+E**pE)/(hE**pE)))
    dpw=(pwin-z[1])/tau+g*(SOD**pSOD/(SOD**pSOD+HSOD**pSOD))
    
    return np.array([dv, dpw])

# Autonomous version of van Nes slow/fast model 2 in: https://link.springer.com/article/10.1007/s10021-006-0176-0
def vanNes2007_model2_na(t,z,para): 
    
    pE,rV,hE,gammaE,pwin,tau,g,pSOD,HSOD,lSOD,hV,kV,A,omega=para
    
    z[0] = np.max([z[0],0])
    z[1] = np.max([z[1],0])
    
    E=gammaE*z[1]*hV/(hV+z[0])
    SOD=kV*z[0]/lSOD

    dv=rV*z[0]*(1-(z[0]*(hE**pE+E**pE)/(hE**pE)))
    dpw=(pwin-z[1])/tau+g*(SOD**pSOD/(SOD**pSOD+HSOD**pSOD))+A/tau*np.sin(omega*t)
    
    return np.array([dv, dpw])

def vanNes2007_na_aut(t,z,para):
    
    pE,rV,hE,gammaE,pwin,tau,g,pSOD,HSOD,lSOD,hV,kV,A,omega=para
    
    z[0] = np.max([z[0],0])
    z[1] = np.max([z[1],0])
    
    E=gammaE*z[1]*hV/(hV+z[0])
    SOD=kV*z[0]/lSOD

    dv=rV*z[0]*(1-(z[0]*(hE**pE+E**pE)/(hE**pE)))
    dpw=(pwin-z[1])/tau+g*(SOD**pSOD/(SOD**pSOD+HSOD**pSOD))+A/tau*np.sin(z[2])
    dz=omega
    
    return np.array([dv, dpw, dz])

def vanNes2007_na_aut_jac(x, params):
      
    pE,rV,hE,gammaE,pwin,tau,g,pSOD,HSOD,lSOD,hV,kV,A,omega=params
    
    dVdV = x[0]*rV*(x[0]*pE*(x[1]*gammaE*hV/(x[0] + hV))**pE/(hE**pE*(x[0] + hV)) - (hE**pE + (x[1]*gammaE*hV/(x[0] + hV))**pE)/hE**pE) + rV*(-x[0]*(hE**pE + (x[1]*gammaE*hV/(x[0] + hV))**pE)/hE**pE + 1)
    dVdPw = -x[0]**2*pE*rV*(x[1]*gammaE*hV/(x[0] + hV))**pE/(x[1]*hE**pE)
    dVdZ = 0
    
    dPwdV = -x[0]**(2*pSOD)*g*pSOD/(x[0]*(x[0]**pSOD + (HSOD*lSOD/kV)**pSOD)**2) + x[0]**pSOD*g*pSOD/(x[0]*(x[0]**pSOD + (HSOD*lSOD/kV)**pSOD))
    dPwdPw = -1/tau
    dPwdZ = A*np.cos(x[2])/tau
    
    dZdV = 0
    dZdPw = 0
    dZdZ = 0
    
    J = np.array([
        [dVdV,dVdPw,dVdZ],
        [dPwdV,dPwdPw,dPwdZ],
        [dZdV,dZdPw,dZdZ]
        ])
    return J
