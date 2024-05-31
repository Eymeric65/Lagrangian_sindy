
from scipy import interpolate

from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *

# Double pendulum exclusive.....

# Initialisation du modèle théorique

F_ext1 = sp.symbols("F_ext1")

theta1 = sp.Function("theta1")
theta1_d = sp.Function("theta1_d")
theta1_dd = sp.Function("theta1_dd")

F_ext2 = sp.symbols("F_ext2")

theta2 = sp.Function("theta2")
theta2_d = sp.Function("theta2_d")
theta2_dd = sp.Function("theta2_dd")

t = sp.symbols("t")

Symbol_matrix = np.array(
    [[F_ext1, F_ext2], [theta1(t), theta2(t)], [theta1_d(t), theta2_d(t)], [theta1_dd(t), theta2_dd(t)]])

m1, l1, m2, l2, g = sp.symbols("m1 l1 m2 l2 g")

L1t = 0.2
L2t = 0.2
Lt = L1t + L2t
Substitution = {"g": 9.81, "l1": L1t, "m1": 0.1, "l2": L2t, "m2": 0.1}

Time_end = 14

#----------------External Forces--------------------

F_ext_time = np.array([0, 2, 4, 6, 8, Time_end])
F_ext_Value = np.array([[0, 0, 0, 0, 0, 0],[0, 1, -1, 1, 1, -1]]) * 0.0  # De la forme (k,...)

F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)
# ---------------------------

#Y0 = np.array([[0, 0], [0.0, 0]])  # De la forme (k,2)
Y0 = np.array([[2, 0], [0.1, 0]])  # De la forme (k,2)

L = 1 / 2 * (m1 + m2) * l1 ** 2 * theta1_d(t) ** 2 + 1 / 2 * m2 * l2 ** 2 * theta2_d(t) ** 2 + m2 * l1 * l2 * theta1_d(
    t) * theta2_d(t) * sp.cos(theta1(t) - theta2(t)) + (m1 + m2) * g * l1 * sp.cos(theta1(t)) + m2 * g * l2 * sp.cos(
    theta2(t))

Dynamics_system = Dynamics_f(Lagrangian_to_Acc_func(L, Symbol_matrix, t, Substitution, fluid_f=0.0), F_ext_func)

t_values, theta_values = Run_RK45(Dynamics_system, Y0, Time_end)

Animate_double_pendulum(L1t, L2t, theta_values, t_values)
