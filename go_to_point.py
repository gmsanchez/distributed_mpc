#!/usr/bin/env python
"""
Simple example of subscribing to sensor messages and publishing
twist messages to the turtlebot.

Author: Nathan Sprague
Version: 1/12/2015

"""
import rospy
import math
import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
import sys

# Twist is the message type for sending movement commands.
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# globals
ODOM = None

# Some constants.
Delta = .02
Nx = 3
Nu = 2


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
    
# Define model.
def ode(x,u):
    # return np.array([-x[0] - (1+u[0])*x[1], (1+u[0])*x[0] + x[1] + u[1]])
    return np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])

# Then get nonlinear casadi functions and the linearization.
ode_casadi = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"], funcname="f")

Q = np.array([[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 0.01]])
Qn = np.array([[150.0, 0.0, 0.0], [0.0, 150.0, 0.0], [0.0, 0.0, 0.01]])
R = np.array([[25.0, 0.0], [0.0, 25.0]])

if len(sys.argv)==3:
    x_ref = np.array([float(sys.argv[1]),float(sys.argv[2]),0])
else:
    x_ref = np.array([10,10,0])

# Define stage cost and terminal weight.
def lfunc(x,u):
    return mpc.mtimes((x-x_ref).T,Q,(x-x_ref)) + mpc.mtimes(u.T,R,u)
l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")

def Pffunc(x):
    return mpc.mtimes((x-x_ref).T,Qn,(x-x_ref))
Pf = mpc.getCasadiFunc(Pffunc, [Nx], ["x"], funcname="Pf")

# Make optimizers. Note that the linear and nonlinear solvers have some common
# arguments, so we collect those below.
x0 = np.array([0.0,0.0,0.0*np.pi*0.5])
Nt = 20
u_ub = np.array([4., 1.0])
u_lb = np.array([-4., -1.0])

commonargs = dict(
    verbosity=0,
    l=l,
    x0=x0,
    Pf=Pf,
    lb={"u" : np.tile(u_lb,(Nt,1))},
    ub={"u" : np.tile(u_ub,(Nt,1))},
)
Nlin = {"t":Nt, "x":Nx, "u":Nu}
Nnonlin = Nlin.copy()
Nnonlin["c"] = 2 # Use collocation to discretize.

solver = mpc.nmpc(f=ode_casadi,N=Nnonlin,Delta=Delta,**commonargs)

# This function will be called every time a new scan message is
# published.
def odom_callback(odom_msg):
    # Save a global reference to the most recent sensor state so that
    # it can be accessed in the main control loop.
    # (The global keyword prevents the creation of a local variable here.)
    global ODOM
    ODOM = odom_msg

# This is the 'main'
def start():
    
    # Turn this into an official ROS node named approach
    rospy.init_node('approach', anonymous=True)

    # Subscribe to the /scan topic.  From now on
    # scan_callback will be called every time a new scan message is
    # published.
    rospy.Subscriber("odometry/filtered", Odometry, odom_callback)
    # Create a publisher object for sending Twist messages to the
    # turtlebot_node's velocity topic. 
    vel_pub = rospy.Publisher('husky_velocity_controller/cmd_vel', Twist, queue_size=10)
    # Create a twist object. 
    # twist.linear.x represents linear velocity in meters/s.
    # twist.angular.z represents rotational velocity in radians/s.
    twist = Twist()

    # Wait until the first scan is available.
    while ODOM is None and not rospy.is_shutdown():
        rospy.sleep(.1)

    # Rate object used to make the main loop execute at 10hz.
    rate = rospy.Rate(50)
    t = 0

    while not rospy.is_shutdown() and t<1500:

        # Back up if the scan is bad, or if we are too close. 
        # if math.isnan(SCAN.ranges[320]) or SCAN.ranges[320] < target:
        #     twist.linear.x = -0.1  # forward velocity in meters/second
        #     twist.angular.z = 0   # rotation in radians/second
        # else:
        #     twist.linear.x = 0.1 
        #     twist.angular.z = 0  

        # print "Range: {:.3f}".format(SCAN.ranges[320])

####
        _xk = np.zeros((Nx,))
        _xk[0] = ODOM.pose.pose.position.x
        _xk[1] = ODOM.pose.pose.position.y

        euler = toEulerAngle(ODOM)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        _xk[2] = yaw

        # We can stop early if we are already close to the origin.
        # if np.sum(_xk[:2]**2) < 1e-4:
        if np.sum((_xk[:2]-x_ref[:2])**2) < 1e-1:
            print "At origin."# after %d iterations." % (t)
            break

        print "xk",_xk

        solver.fixvar("x",0,_xk)
        solver.solve()

        # Print status and make sure solver didn't fail.
        print "%d: %s" % (t,solver.stats["status"])
        # print "%s" % (solver.stats["status"])
        if solver.stats["status"] != "Solve_Succeeded":
            break
        else:
            solver.saveguess()
        _uk = np.squeeze(solver.var["u",0])
####
        twist.linear.x = _uk[0]
        twist.angular.z = _uk[1]
#        if ODOM.pose.pose.position.x<10:
#            twist.linear.x = 0.5
#        else:
#            twist.linear.x = 0.0
#            
#        print ODOM.pose.pose.position.x

        vel_pub.publish(twist) # These velocities will be applied for .6 seconds
                               # unless another command is sent before that. 
        rate.sleep()           # Pause long enough to maintain correct rate.
        t+=1
        

# This is how we usually call the main method in Python. 
if __name__ == "__main__":
    start()