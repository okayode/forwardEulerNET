import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import random
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint,solve_ivp
from scipy import integrate,stats

# solving using actual parameters

#epsilon = par[0]; alpha = par[1]; beta = par[2];
def savefig(filename, crop = True):
    plt.savefig('{}.pdf'.format(filename))

def fitzHughNag(par,init_cond, t0,tf,incr):
    t=np.linspace(t0,tf,incr)
    def f(y,t):
        y1=y[0]
        y2=y[1]
        f0 = (1/par[0])*(y1-(y1**3)/3-y2)
        f1 = par[0]*(y1 - par[1]*y2 + par[2])
        df=[f0,f1]
        return df
    soln = odeint(f,init_cond,t)
    return (soln,t)

# Actual parameters for the FitzHugh Nagumo model
t0 = 0
tf = 40
incr = 2000
init_cond = [1,-1]
#par = [0.3,0.5,0.75]
par = [0.3,0.5,0.75]
F0,T = fitzHughNag(par,init_cond,t0,tf,incr)

fig, (ax0) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(15, 12))
#plt.figure(1)
ax0.plot(T,F0[:,0],'-r')
ax0.plot(T,F0[:,1],'-g')
ax0.legend(('V','W'),loc='best',fontsize = 20)
ax0.set_xlabel('Time',fontsize = 20)
ax0.set_ylabel('V, W',fontsize = 20)
ax0.set_title('membrane potential v vs linear recovery variable w  ',fontsize = 20)

savefig('./figures/potentialVSrecovery')

#tol=1e-8
tol=1e-4
for i in range(1,len(F0[:,1])):
    if abs(F0[i,1]-(F0[i,0] - (1/3)*F0[i,0]**3))<tol:
        u_equi=F0[i,0]
        v_equi=F0[i,1]

print(u_equi)
print(v_equi)
# Trajectory
#plt.figure(2)
fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(15, 12))
F_nullcline1=par[1]*F0[:,1]-par[2]
F_critical=F0[:,0] - (1/3)*F0[:,0]**3

# Nullclines
ax1.plot(F_nullcline1 , F0[:,1],'r-',label='Nullclines')
#plt.plot(F0[:,0],F_nullcline2 ,'r-',label='Nullclines')

# Critical manifold
ax1.plot(F0[:,0],F_critical ,'b-',label='Critical manifold')

ax1.legend(loc='best',fontsize = 25)
ax1.set_xlabel('u',fontsize = 25)
ax1.set_ylabel('v',fontsize = 25)
ax1.annotate('(u_e,v_e)',(u_equi,v_equi),fontsize = 30)
#plt.set_aspect('equal')
ax1.grid(True, which='both')
ax1.spines['left'].set_position('zero')
ax1.spines['right'].set_color('none')
ax1.yaxis.tick_left()
ax1.spines['bottom'].set_position('zero')
ax1.spines['top'].set_color('none')
ax1.xaxis.tick_bottom()
#ax1.set_title('Trajectory of FHN model ',fontsize = 20)

savefig('./figures/nullclines')
