from scipy import interpolate
from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *
from function.Catalog_gen import *

from function.ray_env_creator import *

# Single pendulum exclusive.....

# Initialisation du modèle théorique

t = sp.symbols("t")

CoordNumb = 1

Symb = Symbol_Matrix_g(CoordNumb,t)

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

Acc_func,_ = Lagrangian_to_Acc_func(L_System, Symb, t, Substitution, fluid_f=[-0.01])

Dynamics_system = Dynamics_f_extf(Acc_func)

EnvConfig = {
    "coord_numb": CoordNumb,
    "target":np.array([-0.5,0]),
    "dynamics_function_h":Dynamics_system,
    "h":0.01
}

Environment = MyFunctionEnv(EnvConfig)

s = -1 

while True:
    # if (Environment.state[] /0.5) % 2 > 1 and s==-1 :
    #     s=1

    # if (Environment.time /0.5) % 2 < 1 and s==1 :
    #     s=-1

    

    state, reward, done, truncated,_ =Environment.step(np.array([0.15*s]))

    s = - np.sign(Environment.state[1])
    print(s)

    #print(state, reward, done, truncated)

    Environment.render()

