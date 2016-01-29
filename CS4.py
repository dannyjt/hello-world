# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 23:19:44 2015

@author: dannyjt
"""

from numpy import *
from scipy.optimize import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
plt.close("all")

w = np.array((0, 1, 2, 3, 4))
for w in range(0, len(w)):
    ###############################################################################
    #chose w to pick which case
    #w goes from 0 to 4
    #w = 0
    if w == 0:
        Title = "C=0.1 s=0.25"
    elif w == 1:
        Title = "C=0.5 s=0.25"
    elif w == 2:
        Title = "C=2 s=0.25"
    elif w == 3:
        Title = "C=0.5 s=0.5"
    elif w == 4:
        Title = "C=0.5 s=1"
    
    ###############################################################################
    #Coefficients & Parameters
    u = 0.2 #(m/s) covection velocity
    D = 0.005 #(m2/s) diffision coefficient 
    L = 1
    k = 2*np.pi/L
    tau = 1/((k**2)*D)
    
    #Domain
    x0 = 0
    x_end = L
    t0 = 0
    t_end = tau
    
    # 5 Cases
    s = np.array((0.25, 0.25, 0.25, 0.5, 1))
    C = np.array((0.1, 0.5, 2, 0.5, 0.5))
    
    # delta t & delta x
    dt = (D/u**2)*(C**2/s)
    dx = (D/u)*(C/s)
    Nx = (x_end - x0)/dx + 1
    Nt = (t_end - t0)/dt + 1
    x = np.linspace(x0, x_end, Nx[w])
    t = np.arange(x0, t_end, dt[w])
    
    ##############################################################################
    # ANALYTICAL SOLUTION
    figure(2*w+1)
    plt.xlabel('Length (m)')
    plt.ylabel(r'$\phi$')
    plt.title(Title)
    Phi_analy = np.zeros((Nx[w],1))
    time = t0
    while time <= t_end:
        for i in range(0,len(x)):
            Phi_analy[i] = np.exp((-k**2)*D*time)*np.sin(k*(x[i]-u*time))
        plt.plot(x,Phi_analy)
        time = time + dt[w]
    plt.show()
    
    Phi_analy_tau = np.zeros((Nx[w],1))
    
    for i in range(0,len(x)):
        Phi_analy_tau[i] = np.exp((-k**2)*D*t_end)*np.sin(k*(x[i]-u*t_end))
    
    
    ##############################################################################
    #TRAPEZOIDAL METHOD (CRANK-NICOLSON)
        
    # Allocate zeros in the A (left-hand side) and B (right-hand side)
    A = np.zeros((Nx[w],Nx[w])) 
    B = np.zeros((Nx[w],Nx[w]))
    Aa = -u/(4*dx[w]) - D/(2*dx[w]**2)
    Ab = 1/dt[w] + D/dx[w]**2
    Ac = u/(4*dx[w]) - D/(2*dx[w]**2)
    Ba = D/(2*dx[w]**2) + u/(4*dx[w])
    Bb = 1/dt[w] - D/dx[w]**2
    Bc = -u/(4*dx[w]) + D/(2*dx[w]**2)
    
    # A and B matrix diagonals
    # left diagonal
    for i in range(1,len(x)):
        A[i,i-1] = Aa
        B[i,i-1] = Ba
        
    # middle diagonal
    for i in range(0,len(x)):
        A[i,i] = Ab
        B[i,i] = Bb
        
    # right diagonal
    for i in range(1,len(x)):
        A[i-1,i] = Ac
        B[i-1,i] = Bc
        
    #periodic boundary conditions
    B[0,Nx[w]-2] = Ba
    B[Nx[w]-1,1] = Bc
    A[0,Nx[w]-2] = Aa
    A[Nx[w]-1,1] = Ac
    Phi_trap_old = np.sin(k*x[:])
    
    figure(2*w+2)
    plt.xlabel('Length (m)')
    plt.ylabel(r'$\phi$')
    plt.title(Title)
    plt.plot(x,Phi_analy_tau, '-', label='Analytical')
    #plt.plot(x,Phi_analy)
    
    time = t0
    while time <= t_end:
        Phi_hold = np.dot(B,Phi_trap_old)
        Phi_trap_new = np.linalg.solve(A,Phi_hold)
        #plt.plot(x,Phi_trap_new, label="Trapezoidal")
        time = time + dt[w]
        Phi_trap_old = Phi_trap_new
    
    ###############################################################################
    # CENTRAL DIFFERENCING
    A_cen = np.zeros((Nx[w],Nx[w])) 
    Aa_cen = D/dx[w]**2 + u/(2*dx[w])
    Ab_cen = -2*D/dx[w]**2
    Ac_cen = D/dx[w]**2 - u/(2*dx[w])
    
    # A and B matrix diagonals
    # left diagonal
    for i in range(1,len(x)):
        A_cen[i,i-1] = Aa_cen
        
    # middle diagonal
    for i in range(0,len(x)):
        A_cen[i,i] = Ab_cen
        
    # right diagonal
    for i in range(1,len(x)):
        A_cen[i-1,i] = Ac_cen
        
    #periodic boundary conditions
    A_cen[0,Nx[w]-2] = Aa_cen
    A_cen[Nx[w]-1,1] = Ac_cen
    Phi_cen_old = np.sin(k*x[:])
    
    def myfunc(t,phi_cen):
        dphidt_cen = np.dot(A_cen,phi_cen)
        return [dphidt_cen]
    
    r = ode(myfunc).set_integrator('vode', method='bdf', order=4, atol=10e-6, rtol=10e-6)
    r.set_initial_value(Phi_cen_old,t0)
    Phi_cen = [] #solution vector
    t = [] #time vector
    while r.successful() and r.t <= t_end:
        r.integrate(r.t + dt[w])
        Phi_cen.append(r.y)
        t.append(r.t)
    Phi_cen = np.array(Phi_cen)
    
    ###############################################################################
    # UPWIND
    A_up = np.zeros((Nx[w],Nx[w])) 
    Aa_up = D/dx[w]**2 + u/dx[w]
    Ab_up = -2*D/dx[w]**2 - u/dx[w]
    Ac_up = D/dx[w]**2
    
    # A and B matrix diagonals
    # left diagonal
    for i in range(1,len(x)):
        A_up[i,i-1] = Aa_up
        
    # middle diagonal
    for i in range(0,len(x)):
        A_up[i,i] = Ab_up
        
    # right diagonal
    for i in range(1,len(x)):
        A_up[i-1,i] = Ac_up
        
    #periodic boundary conditions
    A_up[0,Nx[w]-2] = Aa_up
    A_up[Nx[w]-1,1] = Ac_up
    Phi_up_old = np.sin(k*x[:])
    
    def myfunc(t,phi_up):
        dphidt_up = np.dot(A_up,phi_up)
        return [dphidt_up]
    
    r = ode(myfunc).set_integrator('vode', method='bdf', order=4, atol=10e-6, rtol=10e-6)
    r.set_initial_value(Phi_up_old,t0)
    Phi_up = [] #solution vector
    t = [] #time vector
    while r.successful() and r.t <= t_end:
        r.integrate(r.t + dt[w])
        Phi_up.append(r.y)
        t.append(r.t)
    Phi_up = np.array(Phi_up)
    
    ###############################################################################
    # QUICK
    A_q = np.zeros((Nx[w],Nx[w])) 
    a1 = 3./8.
    a2 = 1./8.
    Aa_q = -u*(-1+a1-2*a2)/dx[w] + D/dx[w]**2
    Ab_q = -2*D/dx[w]**2 - u*(1-a1+a2-a1)/dx[w]
    Ac_q = D/dx[w]**2 - u*a1/dx[w]
    Ad_q = -u*a2/dx[w]
    
    # A and B matrix diagonals
    # left left diagonal
    for i in range(2,len(x)):
        A_q[i,i-2] = Ad_q
    # left diagonal
    for i in range(1,len(x)):
        A_q[i,i-1] = Aa_q
        
    # middle diagonal
    for i in range(0,len(x)):
        A_q[i,i] = Ab_q
        
    # right diagonal
    for i in range(1,len(x)):
        A_q[i-1,i] = Ac_q
        
    #periodic boundary conditions
    A_q[0,Nx[w]-3] = Ad_q
    A_q[0,Nx[w]-2] = Aa_q
    A_q[1,Nx[w]-2] = Ad_q
    A_q[Nx[w]-1,1] = Ac_q
    Phi_q_old = np.sin(k*x[:])
    
    def myfunc(t,phi_q):
        dphidt_q = np.dot(A_q,phi_q)
        return [dphidt_q]
    
    r = ode(myfunc).set_integrator('vode', method='bdf', order=4, atol=10e-6, rtol=10e-6)
    r.set_initial_value(Phi_q_old,t0)
    Phi_q = [] #solution vector
    t = [] #time vector
    while r.successful() and r.t <= t_end:
        r.integrate(r.t + dt[w])
        Phi_q.append(r.y)
        t.append(r.t)
    Phi_q = np.array(Phi_q)
    
    ###############################################################################
    # PLOTS
    
    plt.plot(x,Phi_trap_new, '--', label="Trapezoidal")
    plt.plot(x,Phi_cen[-1,:], 'g^', label="Central Dif")
    plt.plot(x,Phi_up[-1,:], 'bs', label="Upwind")
    plt.plot(x,Phi_q[-1,:], 'r*', label="QUICK")
    plt.legend(loc='upper right', numpoints = 1)
    
    ###############################################################################
    # RMS ERROR
    
    sum_rms_trap = 0
    sum_rms_cen = 0
    sum_rms_up = 0
    sum_rms_q = 0
    for i in range(0,len(x)):
        sum_rms_trap += (Phi_trap_new[i] - Phi_analy_tau[i])**2
        sum_rms_cen += (Phi_cen[-1,i] - Phi_analy_tau[i])**2
        sum_rms_up += (Phi_up[-1,i] - Phi_analy_tau[i])**2
        sum_rms_q += (Phi_q[-1,i] - Phi_analy_tau[i])**2
    RMS_trap = np.sqrt(sum_rms_trap/len(x))
    RMS_cen = np.sqrt(sum_rms_cen/len(x))
    RMS_up = np.sqrt(sum_rms_up/len(x))
    RMS_q = np.sqrt(sum_rms_q/len(x))
    print Title
    print RMS_trap
    print RMS_cen
    print RMS_up
    print RMS_q

