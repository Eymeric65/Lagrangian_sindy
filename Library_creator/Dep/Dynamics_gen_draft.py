import sympy as sp
import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import RK45
import numpy as np

def Euler_lagranged(expr,symbol_matrix,t,qi): #Euler Lagrange en symbolique
    dL_dq = sp.diff(expr, symbol_matrix[1,qi])
    dL_dq_d = sp.diff(expr,symbol_matrix[2,qi])
    d_dt = (sp.diff(dL_dq_d,t))

    for j in range(symbol_matrix.shape[1]):

        d_dt = d_dt.replace(sp.Derivative(symbol_matrix[1,j],t),symbol_matrix[2,j])
        d_dt = d_dt.replace(sp.Derivative(symbol_matrix[2, j], t), symbol_matrix[3, j])

    return dL_dq - d_dt

def Dynamics_f(Acc,Fext):

    def func(t,State):

        State = np.transpose(np.reshape(State,(-1,2)))

        Input = np.zeros((State.shape[0] + 2, State.shape[1]))

        Input[1:3,:] = State

        Input[0,:] = Fext(t)

        ret = np.zeros(State.shape)
        ret[:,0]=State[1,:]
        ret[:,1]=Acc(Input)[:,0]
        return np.reshape(ret,(-1,))

    return func


# Experience Single Pendulum

# On a k coordonnées généralisé

F_ext1 = sp.symbols("F_ext1")

theta1 = sp.Function("theta1")
theta1_d = sp.Function("theta1_d")
theta1_dd = sp.Function("theta1_dd")

F_ext2 = sp.symbols("F_ext2")

theta2 = sp.Function("theta2")
theta2_d = sp.Function("theta2_d")
theta2_dd = sp.Function("theta2_dd")

t = sp.symbols("t")

Symbol_matrix = np.array([[F_ext1,F_ext2],[theta1(t),theta2(t)],[theta1_d(t),theta2_d(t)],[theta1_dd(t),theta2_dd(t)]])

m1,l1,m2,l2,g = sp.symbols("m1 l1 m2 l2 g")

L1t = 0.2
L2t = 0.2
Lt= L1t+L2t
Substitution = {"g":9.81,"l1":L1t,"m1":0.1,"l2":L2t,"m2":0.1}

Time_end = 14

F_ext_time = np.array([0,2,4,6,8,Time_end])
F_ext_Value = np.array([[0,1,-1,1,1,-1],[0,1,-1,1,1,-1]])*0.0 # De la forme (k,...)

F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value,axis=1)

print(F_ext_func([0,2]))

Y0 = np.array([[2,0],[0.1,0]]) # De la forme (k,2)
#Y0 = np.reshape(Y0,(-1,)) # de la forme(2*k,)

L = 1/2*(m1+m2)*l1**2*theta1_d(t)**2+1/2*m2*l2**2*theta2_d(t)**2+m2*l1*l2*theta1_d(t)*theta2_d(t)*sp.cos(theta1(t)-theta2(t))+(m1+m2)*g*l1*sp.cos(theta1(t))+m2*g*l2*sp.cos(theta2(t))

def Lagrangian_to_model(L,Symbol_matrix,t,Substitution):

    Qk = Symbol_matrix.shape[1]

    Acc = np.zeros((Qk,1),dtype="object")

    for i in range(Qk):

        Dyn = sp.simplify(Euler_lagranged(L, Symbol_matrix,t,i)) + Symbol_matrix[0,i]
        Acc_s = sp.solve(Dyn, Symbol_matrix[3,i])[0]
        Acc[i,0] =Acc_s.subs(Substitution)

    Acc_lambda = sp.lambdify([Symbol_matrix], Acc)

    return Dynamics_f(Acc_lambda,F_ext_func)


# Dynamics1 = sp.simplify(Euler_lagranged(L, Symbol_matrix,t,0)) + F_ext1 #- 0.1*theta_d(t)
# Dynamics2 = sp.simplify(Euler_lagranged(L, Symbol_matrix,t,1)) + F_ext2 #- 0.1*theta_d(t)
#
# Acc1 = sp.solve(Dynamics1, theta1_dd(t))[0]
# Acc2 = sp.solve(Dynamics2, theta2_dd(t))[0]
#
# Acc1 = Acc1.subs(Substitution)
# Acc2 = Acc2.subs(Substitution)
#
# Acc = np.array([[Acc1],[Acc2]])
#
# print(Acc.dtype)
#
# print(Acc)
#
# Acc_lambda = sp.lambdify([Symbol_matrix],Acc)
#
# Dynamics_system = Dynamics_f(Acc_lambda,F_ext_func)

Dynamics_system = Lagrangian_to_model(L,Symbol_matrix,t,Substitution)

print(Dynamics_system(1,[1,0.2,1.2,-0.2]))


def Run_RK45(model, Y0, Time_end):
    Qk = Y0.shape[0]

    Y0_f = np.reshape(Y0, (-1,))  # de la forme(2*k,)
    Model = RK45(model, 0, Y0_f, Time_end, 0.05, 0.001, np.e ** -6)

    # collect data
    t_values = []
    theta_values = []
    for i in range(1000):
        # get solution step state
        Model.step()
        t_values.append(Model.t)
        theta_values.append(Model.y)
        # break loop after modeling is finished
        if Model.status == 'finished':
            print("End step : ", i)
            break

    theta_values = np.array(theta_values)

    return t_values, theta_values

t_values,theta_values = Run_RK45(Dynamics_system,Y0,Time_end)

print(theta_values.shape)

# plt.figure(0)
# plt.plot(t_values, theta_values[:, 0], label="True Model")
# plt.plot(t_values, theta_values[:, 1], label="True Model")
#
# plt.show()

def Animate_double_pendulum(L1t,L2t,theta_values,t_values):

    Lt = L1t+L2t

    x1 = L1t*np.sin(theta_values[:, 0])
    y1 = -L1t*np.cos(theta_values[:, 0])

    x2 = L2t*np.sin(theta_values[:, 2]) + x1
    y2 = -L2t*np.cos(theta_values[:, 2]) + y1

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-Lt, Lt), ylim=(-Lt, Lt))
    ax.set_aspect('equal')
    ax.grid()


    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        history_x = x2[:i]
        history_y = y2[:i]

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (t_values[i]))
        return line, trace, time_text


    ani = animation.FuncAnimation(
        fig, animate, len(theta_values), interval=40, blit=True)
    plt.show()


Animate_double_pendulum(L1t,L2t,theta_values,t_values)

