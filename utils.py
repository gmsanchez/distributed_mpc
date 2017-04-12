# Twist is the message type for sending movement commands.
import rospy
import math
import mpctools as mpc
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry




def toEulerAngle(odom_msg):
    qx = odom_msg.pose.pose.orientation.x
    qy = odom_msg.pose.pose.orientation.y
    qz = odom_msg.pose.pose.orientation.z
    qw = odom_msg.pose.pose.orientation.w

    ysqr = qy * qy
    t0 = -2.0 * (ysqr + qz * qz) + 1.0
    t1 = +2.0 * (qx * qy - qw * qz)
    t2 = -2.0 * (qx * qz + qw * qy)
    t3 = +2.0 * (qy * qz - qw * qx)
    t4 = -2.0 * (qx * qx + ysqr) + 1.0

    t2 = 1.0 if t2>1.0 else t2
    t2 = -1.0 if t2<-1.0 else t2

    pitch = -math.asin(t2)
    roll = -math.atan2(t3, t4)
    yaw = -math.atan2(t1, t0)
    return (roll, pitch, yaw)
    
def create_robot_controller(idx,Delta,x0):
    print idx, x0
    Nx = 3
    Nu = 2
    # Define model.
    def ode(x,u):
        return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])
    # Then get nonlinear casadi functions and the linearization.
    ode_casadi = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"], funcname="f")

    Q = np.array([[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 0.01]])
    Qn = np.array([[150.0, 0.0, 0.0], [0.0, 150.0, 0.0], [0.0, 0.0, 0.01]])
    R = np.array([[5.0, 0.0], [0.0, 5.0]])
    
    def lfunc(x,u,x_sp):
        return mpc.mtimes((x-x_sp).T,Q,(x-x_sp)) + mpc.mtimes(u.T,R,u)
    l = mpc.getCasadiFunc(lfunc, [Nx,Nu,Nx], ["x","u","x_sp"], funcname="l")

    def Pffunc(x,x_sp):
        return mpc.mtimes((x-x_sp).T,Qn,(x-x_sp))
    Pf = mpc.getCasadiFunc(Pffunc, [Nx,Nx], ["x","x_sp"], funcname="Pf")

    # Make optimizers. Note that the linear and nonlinear solvers have some common
    # arguments, so we collect those below.
    Nt = 5
    x_sp = np.zeros((Nt+1,Nx))
    sp = dict(x=x_sp)
    u_ub = np.array([1000, 1000])
    u_lb = np.array([-1000, -1000])

    commonargs = dict(
        verbosity=0,
        l=l,
        x0=x0,
        Pf=Pf,
        lb={"u" : np.tile(u_lb,(Nt,1))},
        ub={"u" : np.tile(u_ub,(Nt,1))},
        funcargs = {"l":["x","u","x_sp"], "Pf":["x","x_sp"]}
        )
    Nlin = {"t":Nt, "x":Nx, "u":Nu, "x_sp":Nx}
    Nnonlin = Nlin.copy()
    Nnonlin["c"] = 2 # Use collocation to discretize.

    solver = mpc.nmpc(f=ode_casadi,N=Nnonlin,sp=sp,Delta=Delta,**commonargs)
    solver.fixvar('x',0,x0)
    return dict(x0=x0,solver=solver,Nt=Nt,Nx=Nx,Nu=Nu)