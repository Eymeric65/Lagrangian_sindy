import sympy as sp
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import RK45
import numpy as np
import time

def Euler_lagranged(expr,q,q_d,q_dd):

    dL_dq = sp.diff(expr, q(t))

    dL_dq_d = sp.diff(expr,q_d(t))

    d_dt = sp.diff(dL_dq_d,t).replace(sp.Derivative(q(t),t),q_d(t)).replace(sp.Derivative(q_d(t),t),q_dd(t))

    return dL_dq - d_dt

def Dynamics_f(Acc,Fext):

    def func(t,State):

        State = np.reshape(State,(-1,2))

        ret = np.zeros(State.shape)
        ret[:,0]=State[:,1]
        ret[:,1]=Acc(State[:,0],State[:,1],Fext(t))
        return np.reshape(ret,(-1,))

    return func

t = sp.symbols("t")

F_ext = sp.symbols("F_ext")

theta = sp.Function("theta")
theta_d = sp.Function("theta_d")
theta_dd = sp.Function("theta_dd")

m,l,g = sp.symbols("m l g")

Substitution = {"g":9.81,"l":0.2,"m":0.1}

Time_end = 14

F_ext_time = np.array([0,2,4,6,8,Time_end])
F_ext_Value = np.array([1,1,-1,0,0,0])*0.12

Y0 = np.array([[0,0]])
Y0 = np.reshape(Y0,(-1,))

F_ext_func = interpolate.interp1d(F_ext_time, F_ext_Value)

L = m*l**2/2*theta_d(t)**2+sp.cos(theta(t))*l*m*g

Dynamics = sp.simplify(Euler_lagranged(L,theta,theta_d,theta_dd))+F_ext

Acc = sp.solve(Dynamics,theta_dd(t))[0]

Acc_sub = Acc.subs(Substitution)
Acc_lambda = sp.lambdify([theta(t), theta_d(t), F_ext], Acc_sub)

System_func = Dynamics_f(Acc_lambda, F_ext_func)

Model = RK45(System_func,0,Y0,Time_end ,0.05, 0.001, np.e**-6)

# collect data
t_values = []
y_values = []
for i in range(1000):
    # get solution step state
    Model.step()
    t_values.append(Model.t)
    y_values.append(Model.y[0])
    # break loop after modeling is finished
    if Model.status == 'finished':
        print("End step : ",i)
        break

plt.plot(t_values,y_values)

plt.show()



