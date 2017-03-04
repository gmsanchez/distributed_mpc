import numpy as np
import mpctools as mpc
import matplotlib.pyplot as plt
import itertools
import casadi
# Rolling ball game example. Linear model but nonlinear constraints.

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
Nt = 15
Nsim = 60
Delta = 0.2
r_dist = 0.4

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

x0 = np.array([2,2,0,0, 1.8,2,0,0, 2,2.1,0,0])
lb = {
    "u": np.tile(np.tile([-umax,-umax,-0.4],Nrobots),(Nt,1)),
}
ub = {
    "u" : np.tile(np.tile([umax,umax,0.4],Nrobots),(Nt,1)),
}

Q  = generate_weight_matrix(np.array([5.0, 5.0, 0.0, 0.0]), Nrobots)
Qn = generate_weight_matrix(np.array([50.0, 50.0, 0.0, 0.0]), Nrobots)
R  = generate_weight_matrix(np.array([50,50,10000]), Nrobots)

def lfunc(x,u):
    return mpc.mtimes(x.T,Q,x) + mpc.mtimes(u.T,R,u) # - casadi.log( - 0.2**2 +(x[0]-x[4])**2 + (x[1]-x[5])**2)
l = mpc.getCasadiFunc(lfunc, [Nx*Nrobots,Nu*Nrobots], ["x","u"], "l")

Pf = mpc.getCasadiFunc(lambda x: mpc.mtimes(x.T,Qn,x), [Nx*Nrobots], ["x"], "Pf")

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

# Now ready for simulation.
X = np.zeros((Nsim+1,Nx*Nrobots))
X[0,:] = x0
U = np.zeros((Nsim,Nu*Nrobots))

for t in range(Nsim):
    controller.fixvar("x",0,X[t,:])
    controller.solve()
    print "%5d: %20s" % (t,controller.stats["status"])
    
    X[t+1,:] = np.squeeze(controller.var["x",1])
    U[t,:] = np.squeeze(controller.var["u",0])
        

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