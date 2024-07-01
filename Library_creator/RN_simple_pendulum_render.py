
from scipy import interpolate
from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *
from function.Catalog_gen import *

# Single pendulum exclusive.....

# Initialisation du modèle théorique

t = sp.symbols("t")

Symb = Symbol_Matrix_g(1,t)

theta = Symb[1,0]
theta_d = Symb[2,0]
theta_dd = Symb[3,0]

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

L_System = m*l**2/2*theta_d**2+sp.cos(theta)*l*m*g

Acc_func,_ = Lagrangian_to_Acc_func(L_System, Symb, t, Substitution, fluid_f=[-0.0])

Dynamics_system = Dynamics_f(Acc_func, F_ext_func)

t_values, theta_values = Run_RK45(Dynamics_system, Y0, Time_end)

Animate_Single_pendulum(L, theta_values, t_values)
