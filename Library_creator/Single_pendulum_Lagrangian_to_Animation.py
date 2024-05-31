
from scipy import interpolate
from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *

# Single pendulum exclusive.....

# Initialisation du modèle théorique

F_ext = sp.symbols("F_ext")

theta = sp.Function("theta")
theta_d = sp.Function("theta_d")
theta_dd = sp.Function("theta_dd")
t = sp.symbols("t")

Symbol_matrix = np.array(
    [[F_ext], [theta(t)], [theta_d(t)], [theta_dd(t)]])

m, l, g = sp.symbols("m l g")

L = 0.2
Substitution = {"g": 9.81, "l": L, "m": 0.1}

Time_end = 14

#----------------External Forces--------------------

F_ext_time = np.array([0, 2, 4, 6, 8, Time_end])
F_ext_Value = np.array([[0, 1, -1, 1, 1, -1]]) * 0.0  # De la forme (k,...)

F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)
# ---------------------------

Y0 = np.array([[2, 0]])  # De la forme (k,2)

L_System = m*l**2/2*theta_d(t)**2+sp.cos(theta(t))*l*m*g

Dynamics_system = Dynamics_f(Lagrangian_to_Acc_func(L_System, Symbol_matrix, t, Substitution, fluid_f=0.0), F_ext_func)

t_values, theta_values = Run_RK45(Dynamics_system, Y0, Time_end)

Animate_Single_pendulum(L, theta_values, t_values)
