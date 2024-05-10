import numpy as np

from function.Catalog_gen import *

from scipy import interpolate
from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *
from function.Optimization import  *

import seaborn as sn

t = sp.symbols("t")


Coord_number = 2
Symb = Symbol_Matrix_g(Coord_number,t)


theta1 = Symb[1,0]
theta1_d = Symb[2,0]
theta1_dd = Symb[3,0]

theta2 = Symb[1,1]
theta2_d = Symb[2,1]
theta2_dd = Symb[3,1]

m1, l1, m2, l2, g = sp.symbols("m1 l1 m2 l2 g")

L1t = 0.2
L2t = 0.2
Lt = L1t + L2t
Substitution = {"g": 9.81, "l1": L1t, "m1": 0.1, "l2": L2t, "m2": 0.1}

Time_end = 30

#----------------External Forces--------------------

F_ext_time = np.array([0, 2, 4, 6, 8, Time_end])
F_ext_Value = np.array([[0, 1, -1, 0, 0, 0],[0, 1, -1, 0, 0, 0]]) * 0.1  # De la forme (k,...)

F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)

def F_ext(t):
    return np.array([np.cos(t*np.cos(t*2*np.pi/Time_end)*2*np.pi/10)*0.1,np.sin(t*np.cos(t*2*np.pi/Time_end)*2*np.pi/10)*0.1])

F_ext_func = F_ext
# ---------------------------

Y0 = np.array([[0, 0], [0, 0]])  # De la forme (k,2)

L = (1 / 2 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 1 / 2 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
     * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(
    theta2))


#Ajout du bruit

Degree_function = 4

Puissance_model= 2

function_catalog = [
    lambda x : Symb[1,x],
    lambda x : Symb[2,x],
    lambda x : sp.sin(Symb[1,x]),
    lambda x : sp.cos(Symb[1,x])
]

Catalog = Catalog_gen(function_catalog,Coord_number,Degree_function,puissance=Puissance_model)

Solution_ideal = Make_Solution_vec(sp.expand_trig(L.subs(Substitution)),Catalog)#,Frottement=Frotement)

S_index = (Solution_ideal != 0).nonzero()[0]

add_new_comp = 5

count = np.array(range(len(Catalog)))

count = np.delete(count,S_index)

indices = np.random.randint(0,len(count)-1,add_new_comp)

res = np.concatenate((count[indices],S_index))

print(res)

print("Sol ideal",Solution_ideal.shape)
print("Sol ideal fact",len((Solution_ideal != 0).nonzero()[0]))