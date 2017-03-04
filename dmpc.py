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
import math

def generate_weight_matrix(weight_array, nrobots):
    return  np.diag(np.tile(weight_array, nrobots))
    
def generate_weight_matrix_selfish(weight_array, nrobots, this_robot):
    Q_ = np.zeros((weight_array.shape[0]*Nrobots))
    Q_[this_robot*Nx:this_robot*Nx+Nx] = weight_array
    return np.diag(Q_)

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
Rlist = range(Nrobots)
Nx = 4
Nslack = Nrobots-1
Nu = 2
Np = (Nu+Nslack)*Nrobots
Nt = 15
Nsim = 60
Delta = 0.2

# Some bounds.
umax = 2
r_dist = 0.4

# Four states: x1, x2, v1, v2
# Two controls: a1, a2
Acont = np.array([
    [0,0,1,0],
    [0,0,0,1],
    [0,0,0,0],
    [0,0,0,0],
])
Bcont = np.array([
    [0,0],
    [0,0],
    [1,0],
    [0,1],
])
Bslack = np.zeros((Nx,Nslack))
Bcont = np.concatenate((Bcont, Bslack), axis=1)

# Build the matrix for each of the distributed MPC
Al = []
Bl = []
Cl = []
f_casadi = []
e = []
ef = []
l = []
Pf = []
for i in range(Nrobots):
    other_robots_list = range(Nrobots)
    other_robots_list.remove(i)
    Am_cont = generate_matrix_A(Acont, Nrobots)
    Bm_cont = generate_matrix_B([i], Bcont, Nx, Nu+Nslack, Nrobots)
    Cm_cont = generate_matrix_C(other_robots_list, Bcont, Nx, Np, Nrobots)
    # Discretize.    
    (A,B,C,_) = mpc.util.c2d(Am_cont, Bm_cont, Delta, Bp=Cm_cont)
    
    f_casadi.append(mpc.getCasadiFunc(lambda x,u,p: mpc.mtimes(A,x) + mpc.mtimes(B,u) + mpc.mtimes(C,p),
                    [Nx*Nrobots,Nu+Nslack,Np],["x","u","p"], "f"))
                      
    Al.append(A)
    Bl.append(B)
    Cl.append(C)


    def nlcon(x,u):
        x_ = x[i*Nx+0]
        y_ = x[i*Nx+1]
        dist_con = []
        for j in Rlist:
            if j!=i:
                dist_con.append(r_dist**2-(x_-x[j*Nx+0])**2-(y_-x[j*Nx+1])**2+u[(Nu)+len(dist_con)])
        print dist_con
        return np.array(dist_con)
    e.append(mpc.getCasadiFunc(nlcon, [Nx*Nrobots, Nu+Nslack], ["x","u"], "e"))
    def terminalconstraint(x):
        x_ = x[i*Nx+0]
        y_ = x[i*Nx+1]
        dist_con = []
        for j in Rlist:
            if j!=i:
                dist_con.append(r_dist**2-(x_-x[j*Nx+0])**2-(y_-x[j*Nx+1])**2)
        return np.array(dist_con)
    ef.append(mpc.getCasadiFunc(terminalconstraint, [Nx*Nrobots], ["x"], "ef"))
    
    Q  = generate_weight_matrix(np.array([5, 5, 0, 0]), Nrobots)
    Qn = generate_weight_matrix(np.array([50, 50, 0, 0]), Nrobots)
    #Q  = generate_weight_matrix_selfish(np.array([10, 10, 0, 0]), Nrobots, i)
    #Qn = generate_weight_matrix_selfish(np.array([50, 50, 0, 0]), Nrobots, i)
    R  = generate_weight_matrix(np.array([50, 50, 10000, 10000]), 1)

    def lfunc(x,u):
        return mpc.mtimes(x.T,Q,x) + mpc.mtimes(u.T,R,u)
    l.append( mpc.getCasadiFunc(lfunc, [Nx*Nrobots,Nu+Nslack], ["x","u"], "l") )

    Pf.append( mpc.getCasadiFunc(lambda x: mpc.mtimes(x.T,Qn,x), [Nx*Nrobots], ["x"], "Pf") )
    
Ne  = Nrobots-1
Nef = Nrobots-1
x0 = np.array([2,2,0,0, 1.8,2,0,0, 2,2.1,0,0])

lb = {
    "u": np.tile(np.tile([-umax,-umax,-0.4, -0.4], 1),(Nt,1)),
}
ub = {
    "u": np.tile(np.tile([umax,umax,0.4,0.4], 1),(Nt,1)),
}


funcargs = {"f" : ["x","u","p"], "e" : ["x","u"], "l" : ["x","u"], "ef":["x"]}

# Build controller and adjust some ipopt options.

controller=[]
N = {"x":Nx*Nrobots, "u":Nu+Nslack, "e":Ne, "ef":Nef, "t":Nt, "p":Np}

Xk_DATA = np.zeros((Nx*Nrobots,))
U_DATA = np.zeros((Nt,Np))

for i in range(Nrobots):
    controller.append(mpc.nmpc(f_casadi[i], l[i], N, x0, e=e[i], ef=ef[i],lb=lb, ub=ub, funcargs=funcargs, Pf=Pf[i],
                      verbosity=0,casaditype="SX", p=U_DATA))
    controller[i].initialize(solveroptions=dict(max_iter=5000))

# Now ready for simulation.
X = []
U = []

for i in range(Nrobots):
    X.append(np.zeros((Nsim+1,Nx*Nrobots)))
    X[i][0,:] = x0
    U.append(np.zeros((Nsim,Nu+Nslack)))
    Xk_DATA = x0.copy()

r_i_x = [range(i*Nx,i*Nx+Nx) for i in range(Nrobots)]
r_i_p = [range(i*(Nu+Nslack),i*(Nu+Nslack)+(Nu+Nslack)) for i in range(Nrobots)]
for t in range(Nsim):
    for i in range(Nrobots):
        controller[i].fixvar("x",0,Xk_DATA)
        #controller[i].fixvar("x",0,X[i][t,:])
        controller[i].par["p"]=0
        for j in range(1):
            controller[i].par["p",j] = U_DATA[j,:]
        
        controller[i].solve()
        print "%5d: %20s" % (t,controller[i].stats["status"])
    
        X[i][t+1,:] = np.squeeze(controller[i].var["x",1])
        U[i][t,:] = np.squeeze(controller[i].var["u",0])
        
        # for j in range(Nt):
        #     U_DATA[j,i*(Nu+Nslack):i*(Nu+Nslack)+(Nu+Nslack)] = np.squeeze(controller[i].var["u",j])
        # U_DATA[i*(Nu+Nslack):i*(Nu+Nslack)+(Nu+Nslack)] = U[i][t,:]
        # Xk_DATA[i*Nx:i*Nx+Nx] = X[i][t+1,i*Nx:i*Nx+Nx]
    for i in range(Nrobots):
        for j in range(1):
            # U_DATA[j,i*(Nu+Nslack):i*(Nu+Nslack)+(Nu+Nslack)] = np.squeeze(controller[i].var["u",j])
            U_DATA[j,r_i_p[i]] = np.squeeze(controller[i].var["u",j])
        # Xk_DATA[i*Nx:i*Nx+Nx] = X[i][t+1,i*Nx:i*Nx+Nx]
        Xk_DATA[r_i_x[i]] = X[i][t+1,r_i_x[i]]

f = plt.figure()
ax = f.add_subplot(1,1,1)
#for (p,r) in holes:
#    circ = plt.Circle(p,r,edgecolor="red",facecolor=(1,0,0,.5))
#    ax.add_artist(circ)
for i in range(Nrobots):
    ax.plot(X[i][:,i*Nx+0],X[i][:,i*Nx+1],'-o',label="robot"+str(i))   
plt.legend(loc="lower right")
plt.grid()

for i in range(Nrobots):
    plt.figure()
    plt.title("Controles robot "+str(i))
    plt.step(U[i][:,0],'-o',label="u"+str(0), where="post")
    plt.step(U[i][:,1],'-o',label="u"+str(1), where="post")
    plt.legend(loc="lower right")
    plt.grid()

Ndist = math.factorial(Nrobots)/(2*math.factorial(Nrobots-2))
dist = np.zeros((Nsim, Ndist))
dist_lbl = []
k=0
for i in Rlist:
    for j in Rlist:
        if j>i:
            dist_lbl.append("d("+str(i)+","+str(j)+")")
            for t in range(Nsim):
                dist[t,k] = np.linalg.norm(X[i][t,i*Nx:i*Nx+2]-X[j][t,j*Nx:j*Nx+2])
            k+=1

f = plt.figure()
for i in range(Ndist):
    ax = f.add_subplot(Ndist,1,i+1)
    ax.plot(dist[:,i], label=dist_lbl[i])
    plt.legend()
    plt.grid()
    
f = plt.figure()
for i in Rlist:
    ax = f.add_subplot(Nrobots,1,i+1)
    ax.plot(X[i][:,i*Nx+0],label="rx"+str(i))
    ax.plot(X[i][:,i*Nx+1],label="ry"+str(i))
    plt.legend()
    plt.grid()
    


DMPC_X = X
DMPC_U = U
#if not movingHorizon:
#    sol = mpc.util.casadiStruct2numpyDict(controller.var)
#    x = sol["x"]
#    u = sol["u"]
#
#def plotsol(x,holes,xmax,cushion=1):
#    f = plt.figure()
#    ax = f.add_subplot(1,1,1)
#    for (p,r) in holes:
#        circ = plt.Circle(p,r,edgecolor="red",facecolor=(1,0,0,.5))
#        ax.add_artist(circ)
#    ax.plot(x[:,0],x[:,1],'-ok')
#    ax.set_xlabel("$x_1$")
#    ax.set_ylabel("$x_2$")
#    ax.set_xlim((-cushion,xmax+cushion))
#    ax.set_ylim((-cushion,xmax+cushion))
#    return f
#    
#fig = plotsol(x,holes,xmax,cushion)
#mpc.plots.showandsave(fig, "ballmaze.pdf")
