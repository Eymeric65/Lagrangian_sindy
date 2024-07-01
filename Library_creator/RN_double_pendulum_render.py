
from scipy import interpolate

from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *
from function.Catalog_gen import *

# Double pendulum exclusive.....

# Initialisation du modèle théorique

L1t = 1.
L2t = 1.
m_1 = .8
m_2 = .8
Y0 = np.array([[2, 0], [0, 0]])  # De la forme (k,2)
Frotement = [-0.4,-0.02]

t = sp.symbols("t")

Coord_number = 2
Symb = Symbol_Matrix_g(Coord_number,t)

# Ideal model creation

theta1 = Symb[1,0]
theta1_d = Symb[2,0]
theta1_dd = Symb[3,0]

theta2 = Symb[1,1]
theta2_d = Symb[2,1]
theta2_dd = Symb[3,1]

m1, l1, m2, l2, g = sp.symbols("m1 l1 m2 l2 g")

Lt = L1t + L2t
Substitution = {"g": 9.81, "l1": L1t, "m1": m_1, "l2": L2t, "m2": m_2}

L = (1 / 2 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 1 / 2 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
     * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(
    theta2))

Time_end = 14

#----------------External Forces--------------------

F_ext_time = np.array([0, 2, 4, 6, 8, Time_end])
F_ext_Value = np.array([[0, 0, 0, 0, 0, 0],[0, 1, -1, 1, 1, -1]]) * 0.0  # De la forme (k,...)

F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)
# ---------------------------

#Y0 = np.array([[0, 0], [0.0, 0]])  # De la forme (k,2)
Y0 = np.array([[2, 0], [0.1, 0]])  # De la forme (k,2)


Acc_func,_ = Lagrangian_to_Acc_func(L, Symb, t, Substitution, fluid_f=Frotement)

Dynamics_system = Dynamics_f(Acc_func, F_ext_func)

t_values, theta_values = Run_RK45(Dynamics_system, Y0, Time_end)

Animate_double_pendulum(L1t, L2t, theta_values, t_values)
