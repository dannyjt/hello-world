# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 13:46:51 2015

@author: dannyjt
"""

from numpy import *
from scipy.optimize import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
plt.close("all")

# this script runs with a fixed dt value for 2 dx's
dx = np.array((0.1, 0.05, 0.025))
dx = list(dx)

for w in range(0,len(dx)):
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
    
    # delta t & delta x
    ###############################################################################
    dt = 0.00005
    ###############################################################################
    Nx = (x_end - x0)/dx[w] + 1
    Nt = (t_end - t0)/dt + 1
    x = np.linspace(x0, x_end, Nx)
    t = np.arange(x0, t_end, dt)
    
    ##############################################################################
    # ANALYTICAL SOLUTION
    """
    figure(1)
    plt.xlabel('Length (m)')
    plt.ylabel(r'$\phi$')
    plt.title(Title)
    """
    Phi_analy = np.zeros((Nx,1))
    time = t0
    while time <= t_end:
        for i in range(0,len(x)):
            Phi_analy[i] = np.exp((-k**2)*D*time)*np.sin(k*(x[i]-u*time))
        #plt.plot(x,Phi_analy)
        time = time + dt
    #plt.show()
    
    Phi_analy_tau = np.zeros((Nx,1))
    
    for i in range(0,len(x)):
        Phi_analy_tau[i] = np.exp((-k**2)*D*t_end)*np.sin(k*(x[i]-u*t_end))
    
    
    ##############################################################################
    #TRAPEZOIDAL METHOD (CRANK-NICOLSON)
        
    # Allocate zeros in the A (left-hand side) and B (right-hand side)
    A = np.zeros((Nx,Nx)) 
    B = np.zeros((Nx,Nx))
    Aa = -u/(4*dx[w]) - D/(2*dx[w]**2)
    Ab = 1/dt + D/dx[w]**2
    Ac = u/(4*dx[w]) - D/(2*dx[w]**2)
    Ba = D/(2*dx[w]**2) + u/(4*dx[w])
    Bb = 1/dt - D/dx[w]**2
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
    B[0,Nx-2] = Ba
    B[Nx-1,1] = Bc
    A[0,Nx-2] = Aa
    A[Nx-1,1] = Ac
    Phi_trap_old = np.sin(k*x[:])
    
    """
    figure(2)
    plt.xlabel('Length (m)')
    plt.ylabel(r'$\phi$')
    plt.title(Title)
    plt.plot(x,Phi_analy_tau, '-', label='Analytical')
    #plt.plot(x,Phi_analy)
    """
    
    time = t0
    while time <= t_end:
        Phi_hold = np.dot(B,Phi_trap_old)
        Phi_trap_new = np.linalg.solve(A,Phi_hold)
        #plt.plot(x,Phi_trap_new, label="Trapezoidal")
        time = time + dt
        Phi_trap_old = Phi_trap_new
    
    ###############################################################################
    # CENTRAL DIFFERENCING
    A_cen = np.zeros((Nx,Nx)) 
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
    A_cen[0,Nx-2] = Aa_cen
    A_cen[Nx-1,1] = Ac_cen
    Phi_cen_old = np.sin(k*x[:])
    
    def myfunc(t,phi_cen):
        dphidt_cen = np.dot(A_cen,phi_cen)
        return [dphidt_cen]
    
    r = ode(myfunc).set_integrator('vode', method='bdf', order=4, atol=10e-6, rtol=10e-6)
    r.set_initial_value(Phi_cen_old,t0)
    Phi_cen = [] #solution vector
    t = [] #time vector
    while r.successful() and r.t <= t_end:
        r.integrate(r.t + dt)
        Phi_cen.append(r.y)
        t.append(r.t)
    Phi_cen = np.array(Phi_cen)
    
    ###############################################################################
    # UPWIND
    A_up = np.zeros((Nx,Nx)) 
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
    A_up[0,Nx-2] = Aa_up
    A_up[Nx-1,1] = Ac_up
    Phi_up_old = np.sin(k*x[:])
    
    def myfunc(t,phi_up):
        dphidt_up = np.dot(A_up,phi_up)
        return [dphidt_up]
    
    r = ode(myfunc).set_integrator('vode', method='bdf', order=4, atol=10e-6, rtol=10e-6)
    r.set_initial_value(Phi_up_old,t0)
    Phi_up = [] #solution vector
    t = [] #time vector
    while r.successful() and r.t <= t_end:
        r.integrate(r.t + dt)
        Phi_up.append(r.y)
        t.append(r.t)
    Phi_up = np.array(Phi_up)
    
    ###############################################################################
    # QUICK
    A_q = np.zeros((Nx,Nx)) 
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
    A_q[0,Nx-3] = Ad_q
    A_q[0,Nx-2] = Aa_q
    A_q[1,Nx-2] = Ad_q
    A_q[Nx-1,1] = Ac_q
    Phi_q_old = np.sin(k*x[:])
    
    def myfunc(t,phi_q):
        dphidt_q = np.dot(A_q,phi_q)
        return [dphidt_q]
    
    r = ode(myfunc).set_integrator('vode', method='bdf', order=4, atol=10e-6, rtol=10e-6)
    r.set_initial_value(Phi_q_old,t0)
    Phi_q = [] #solution vector
    t = [] #time vector
    while r.successful() and r.t <= t_end:
        r.integrate(r.t + dt)
        Phi_q.append(r.y)
        t.append(r.t)
    Phi_q = np.array(Phi_q)
    
    ###############################################################################
    # PLOTS
    """
    plt.plot(x,Phi_trap_new, '--', label="Trapezoidal")
    plt.plot(x,Phi_cen[-1,:], 'g^', label="Central Dif")
    plt.plot(x,Phi_up[-1,:], 'bs', label="Upwind")
    plt.plot(x,Phi_q[-1,:], 'r*', label="QUICK")
    plt.legend(loc='upper right', numpoints = 1)
    """
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
    print dx[w]
    print RMS_trap
    print RMS_cen
    print RMS_up
    print RMS_q
    
    if w == 0:
        RMS_trap0 = RMS_trap
        RMS_cen0 = RMS_cen
        RMS_up0 = RMS_up
        RMS_q0 = RMS_q
    elif w == 1:
        RMS_trap1= RMS_trap
        RMS_cen1 = RMS_cen
        RMS_up1 = RMS_up
        RMS_q1 = RMS_q
    elif w == 2:
        RMS_trap2 = RMS_trap
        RMS_cen2 = RMS_cen
        RMS_up2 = RMS_up
        RMS_q2 = RMS_q
        
RMS_trap = np.array((RMS_trap0, RMS_trap1, RMS_trap2))
RMS_cen = np.array((RMS_cen0, RMS_cen1, RMS_cen2))
RMS_up = np.array((RMS_up0, RMS_up1, RMS_up2))
RMS_q = np.array((RMS_q0, RMS_q1, RMS_q2))

figure
plt.loglog(dx, RMS_trap, '--' ,label='RMS Trap')
plt.loglog(dx, RMS_cen, 'g^-', label='RMS Central Diff')
plt.loglog(dx, RMS_up, 'bs-', label='RMS Upwind')
plt.loglog(dx, RMS_q, 'r*-', label='RMS QUICK')
plt.xlabel('log(dx)')
plt.ylabel('log(RMS)')
plt.title('log(RMS) vs dx')
plt.legend(loc='lower right', numpoints = 1)

# fit with np.polyfit
m_trap, b = np.polyfit(log10(dx), log10(RMS_trap), 1)
m_cen, b = np.polyfit(log10(dx), log10(RMS_cen), 1)
m_up, b = np.polyfit(log10(dx), log10(RMS_up), 1)
m_q, b = np.polyfit(log10(dx), log10(RMS_q), 1)
print 'slope'
print m_trap
print m_cen
print m_up
print m_q



    
