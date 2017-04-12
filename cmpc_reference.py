import numpy as np
import mpctools as mpc
import matplotlib.pyplot as plt
import itertools
import casadi

import sys
import utils
# Some global options and parameters.
movingHorizon = True
terminalConstraint = False
terminalWeight = False
transientCost = False


def generate_weight_matrix(weight_array, nrobots):
    return  np.diag(np.tile(weight_array, nrobots))

def generate_matrix_B(robot_list, u_matrix, Nx, Nu, Nrobots):
    U = np.zeros((Nx*Nrobots, Nu))
    if not(isinstance(robot_list, (list, tuple))):
        if isinstance(robot_list, int):
            robot_list=[robot_list]
        else:
            raise ValueError("robot_list must be a list, a tuple or an integer.")
    for i in robot_list:
        U[i*Nx:i*Nx+Nx,:] = u_matrix
    return U

def generate_matrix_C(robot_list, p_matrix, Nx, Nu, Nrobots):
    from scipy.linalg import block_diag
    C = block_diag(*[p_matrix for i in range(Nrobots)])
    for i in range(Nrobots):
        if i not in robot_list:
            C[i*Nx:i*Nx+Nx,:] = 0
    return C
        
def generate_matrix_A(a_matrix, Nrobots):
    from scipy.linalg import block_diag
    return block_diag(*[a_matrix for i in range(Nrobots)])
    
        

Nrobots = 3
Nx = 4
Nu = 3
Np = Nu*Nrobots
Nt = 10
Nsim = 60
Delta = 0.5
r_dist = 1.0


if len(sys.argv)==3:
    M_XREF = np.tile(np.array([float(sys.argv[1]),float(sys.argv[2]),0,0]), Nrobots)
else:
    M_XREF = np.tile(np.array([10,10,0,0]), Nrobots)
    
# Some bounds.
umax = 2

# Four states: x1, x2, v1, v2
# Two controls: a1, a2
Acont = np.array([
    [0,0,1,0],
    [0,0,0,1],
    [0,0,0,0],
    [0,0,0,0],
])
Bcont = np.array([
    [0,0,0],
    [0,0,0],
    [1,0,0],
    [0,1,0],

])

# Build the matrix for the centralized MPC
Am_cont = generate_matrix_A(Acont, Nrobots)
Bm_cont = generate_matrix_A(Bcont, Nrobots) #C(range[Nrobots], Bcont, Nx, Nu, Nrobots)
# Discretize.    
(A,B) = mpc.util.c2d(Am_cont, Bm_cont, Delta)

f_casadi = mpc.getCasadiFunc(lambda x,u: mpc.mtimes(A,x) + mpc.mtimes(B,u),
                             [Nx*Nrobots,Nu*Nrobots],["x","u"], "f")
                      
Al = A
Bl = B

r = .25
# m = 3                      
# centers = np.linspace(0,xmax,m+1)
# centers = list(.5*(centers[1:] + centers[:-1]))
centers = [1]
holes = [(p,r) for p in itertools.product(centers,centers)]

Ne=3
def nlcon(x,u):
    # [x1, x2] = x[0:2] # Doesn't work in Casadi 3.0
    return np.array([r_dist**2 -(x[0]-x[4])**2 - (x[1]-x[5])**2 + u[2],
                     r_dist**2 -(x[0]-x[8])**2 - (x[1]-x[9])**2 + u[5],
                     r_dist**2 -(x[8]-x[4])**2 - (x[9]-x[5])**2 + u[8],
                     #0.26**2 - (x[0]-1.0)**2 - (x[1]-1.0)**2,
                     #0.26**2 - (x[4]-1.0)**2 - (x[5]-1.0)**2,
                     #0.26**2 - (x[8]-1.0)**2 - (x[9]-1.0)**2,
                     ])
e = mpc.getCasadiFunc(nlcon, [Nx*Nrobots,Nu*Nrobots], ["x","u"], "e")

M_X0 = [np.array([1,1,0]), np.array([-1,-1,0]), np.array([5,5,0])]
x0 = np.array([1,1,0,0, -1,-1,0,0, 5,5,0,0])
lb = {
    "u": np.tile(np.tile([-umax,-umax,-r_dist],Nrobots),(Nt,1)),
}
ub = {
    "u" : np.tile(np.tile([umax,umax,r_dist],Nrobots),(Nt,1)),
}

Q  = generate_weight_matrix(np.array([1.0, 1.0, 0.0, 0.0]), Nrobots)
Qn = generate_weight_matrix(np.array([5.0, 5.0, 0.0, 0.0]), Nrobots)
R  = generate_weight_matrix(np.array([5,5,10000]), Nrobots)

def lfunc(x,u):
    return mpc.mtimes((x-M_XREF).T,Q,(x-M_XREF)) + mpc.mtimes(u.T,R,u) # - casadi.log( - 0.2**2 +(x[0]-x[4])**2 + (x[1]-x[5])**2)
l = mpc.getCasadiFunc(lfunc, [Nx*Nrobots,Nu*Nrobots], ["x","u"], "l")

Pf = mpc.getCasadiFunc(lambda x: mpc.mtimes((x-M_XREF).T,Qn,(x-M_XREF)), [Nx*Nrobots], ["x"], "Pf")

def terminalconstraint(x):
    return np.array([r_dist**2 -(x[0]-x[4])**2 - (x[1]-x[5])**2,
                     r_dist**2 -(x[0]-x[8])**2 - (x[1]-x[9])**2,
                     r_dist**2 -(x[8]-x[4])**2 - (x[9]-x[5])**2,
                     ])
ef = mpc.getCasadiFunc(terminalconstraint, [Nx*Nrobots], ["x"], "ef")
Nef = 3

funcargs = {"f" : ["x","u"], "e" : ["x","u"], "l" : ["x","u"], "ef" : ["x"]}

# Build controller and adjust some ipopt options.

N = {"x":Nx*Nrobots, "u":Nu*Nrobots, "e":Ne, "t":Nt, "ef":Nef}

controller = mpc.nmpc(f_casadi, l, N, x0, e=e, ef=ef, lb=lb, ub=ub, funcargs=funcargs, Pf=Pf,
                      verbosity=0,casaditype="SX")
controller.initialize(solveroptions=dict(max_iter=5000))
huskies = []
for h_idx in range(Nrobots):
    r_idx = h_idx*Nx
    x0_husky = M_X0[h_idx]
    huskies.append( utils.create_robot_controller(h_idx,Delta,x0_husky))

# Now ready for simulation.
X = np.zeros((Nsim+1,Nx*Nrobots))
X[0,:] = x0
U = np.zeros((Nsim,Nu*Nrobots))


# Define model.
def ode(x,u):
    return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])
husky_rk4 = mpc.getCasadiFunc(ode, [3,2], ["x","u"], rk4=True, Delta=Delta, M=1)

_xk_husky = []
_uk_husky = []
for i in range(Nrobots):
    _xk_husky.append( M_X0[i] )
    _uk_husky.append( np.zeros((huskies[i]['Nu'],)) )

_XK = np.zeros((Nrobots*Nx))

for t in range(Nsim):    
    for _j in range(Nrobots):
        _XK[_j*Nx] = _xk_husky[_j][0]
        _XK[_j*Nx+1] = _xk_husky[_j][1]
        _XK[_j*Nx+2] = _uk_husky[_j][0]*np.cos(_xk_husky[_j][2])
        _XK[_j*Nx+3] = _uk_husky[_j][0]*np.sin(_xk_husky[_j][2])
    controller.fixvar("x",0,_XK)
    controller.solve()
    print "%5d: %20s" % (t,controller.stats["status"])
    
    for _j in range(Nrobots):
        x_ref_k = np.zeros((huskies[_j]['Nx'],huskies[_j]['Nt']+1))
        for _k in range(huskies[_j]['Nt']+1):
            x_ref_k[0,_k] = controller.var['x',_k,_j*Nx+0]
            x_ref_k[1,_k] = controller.var['x',_k,_j*Nx+1]
            huskies[_j]['solver'].par['x_sp',_k] = x_ref_k[:,_k]
        huskies[_j]['solver'].fixvar("x",0,_xk_husky[_j])
        
        huskies[_j]['solver'].solve()

        # Print status and make sure solver didn't fail.
        print "%d: %s" % (t,huskies[_j]['solver'].stats["status"])
        # print "%s" % (solver.stats["status"])
        if huskies[_j]['solver'].stats["status"] != "Solve_Succeeded":
            break
        else:
            huskies[_j]['solver'].saveguess()
        _uk_husky[_j] = np.squeeze(huskies[_j]['solver'].var["u",0])
        print _uk_husky[_j]
        _xk_husky[_j] = husky_rk4(_xk_husky[_j],_uk_husky[_j])
        X[t+1,_j*Nx] = _xk_husky[_j][0]
        X[t+1,_j*Nx+1] = _xk_husky[_j][1]
#    
#    X[t+1,:] = np.squeeze(controller.var["x",1])
#    U[t,:] = np.squeeze(controller.var["u",0])
        

f = plt.figure()
ax = f.add_subplot(1,1,1)
# for (p,r) in holes:
#     circ = plt.Circle(p,r,edgecolor="red",facecolor=(1,0,0,.5))
#     ax.add_artist(circ)
for i in range(Nrobots):
    ax.plot(X[:,i*Nx+0],X[:,i*Nx+1],'-o',label="robot"+str(i))   
    # ax.plot(X[i][:,i*Nx+4],X[i][:,i*Nx+5],'-o',label="robot"+str(i+1))   
plt.legend(loc="lower right")
plt.grid()

for i in range(Nrobots):
    plt.figure()
    plt.title("Controles robot "+str(i))
    plt.step(U[:,i*Nu+0],'-o',label="u"+str(0),where='post')
    plt.step(U[:,i*Nu+1],'-o',label="u"+str(1),where='post')   
    plt.legend(loc="lower right")
    plt.grid()

f = plt.figure()
# ax = f.add_subplot(1,Nrobots,1)
for i in range(Nrobots):
    ax = f.add_subplot(Nrobots,1,i+1)
    ax.plot(X[:,i*Nx+0],'-o',label="robot"+str(i)+" x")  
    ax.plot(X[:,i*Nx+1],'-o',label="robot"+str(i)+" y")
    # ax.plot(X[i][:,i*Nx+4],X[i][:,i*Nx+5],'-o',label="robot"+str(i+1))   
    plt.legend(loc="lower right")
    plt.grid()

d = np.zeros((Nsim,3))
for t in range(Nsim):
    d[t,0] = np.linalg.norm(X[t,0:2]-X[t,4:6])
    d[t,1] = np.linalg.norm(X[t,0:2]-X[t,8:10])
    d[t,2] = np.linalg.norm(X[t,4:6]-X[t,8:10])

f = plt.figure()
# plt.title("Distancias")

ax = f.add_subplot(3,1,1)
ax.plot(d[:,0],label="d(r0,r1)")
plt.legend(loc="lower right")
plt.grid()

ax = f.add_subplot(3,1,2)
ax.plot(d[:,1],label="d(r0,r2)")
plt.legend(loc="lower right")
plt.grid()

ax = f.add_subplot(3,1,3)
ax.plot(d[:,2],label="d(r3,r1)")
plt.legend(loc="lower right")
plt.grid()


CMPC_X = X
CMPC_U = U