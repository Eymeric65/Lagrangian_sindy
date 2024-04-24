from function.Catalog_gen import *

from scipy import interpolate
from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *

t = sp.symbols("t")

#Generate Experience data :

F_ext = sp.symbols("F_ext")

theta = sp.Function("theta")
theta_d = sp.Function("theta_d")
theta_dd = sp.Function("theta_dd")

Symbol_matrix = np.array(
    [[F_ext], [theta(t)], [theta_d(t)], [theta_dd(t)]])

m, l, g = sp.symbols("m l g")

L = 0.2
Substitution = {"g": 9.81, "l": L, "m": 0.1}

Time_end = 14

#----------------External Forces--------------------

F_ext_time = np.array([0,2,4,6,8,Time_end])
F_ext_Value = np.array([0,1,-1,1,0,0])*0.03

F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)
# ---------------------------

Y0 = np.array([[0, 0]])  # De la forme (k,2)

L_System = m*l**2/2*theta_d(t)**2+sp.cos(theta(t))*l*m*g



Dynamics_system = Dynamics_f(Lagrangian_to_Acc_func(L_System, Symbol_matrix, t, Substitution,fluid_f=0.0),F_ext_func)

t_values, thetas_values = Run_RK45(Dynamics_system, Y0, Time_end,max_step=0.01)

# Generate Catalog

Coord_number = 1
Degree_function = 2

Symb = Symbol_Matrix_g(Coord_number,t)

function_catalog = [
    lambda x : Symb[1,x],
    lambda x : Symb[2,x],
    lambda x : sp.sin(Symb[1,x]),
    lambda x : sp.cos(Symb[1,x])
]

Catalog = Catalog_gen(function_catalog,Coord_number,2)
print(Catalog)
# Fitting

Nb_t = len(t_values)

theta_v = thetas_values[:,0]

Exp_matrix = Catalog_to_experience_matrix(Nb_t,Coord_number,Catalog,Symb,t,theta_v)

print(Exp_matrix.shape,t_values.shape,F_ext_func(t_values).shape)

Forces_vec = Forces_vector(F_ext_func,t_values)

print(Forces_vec.shape)

Solution,residual,rank,_ = np.linalg.lstsq(Exp_matrix,Forces_vec,rcond=None)

#Retrieve the model !

Recup_Threshold = 0.1

max_coeff = np.max(Solution)

Solution_thr = np.where(Solution > max_coeff*Recup_Threshold, Solution, 0)

Modele_fit = 0
Modele_ideal = 0

Solution_ideal=np.zeros((len(Catalog),1))

Solution_ideal[4,0] = 0.002
Solution_ideal[3,0] = 0.1962

for i in range(len(Catalog)):
    if(Solution_thr[i]!=0):
        print(Catalog[i])
    Modele_fit = Modele_fit + Catalog[i]*Solution_thr[i]

    Modele_ideal = Modele_ideal + Catalog[i]*Solution_ideal[i]



Modele_fit = sp.simplify(Modele_fit)

print("Lagrangien de base : ",sp.simplify( L_System.subs(Substitution)))
print("Lagrangien fit : ",Modele_fit)
print("Lagrangien ideal : ",Modele_ideal)

# Render

plt.figure(0)
plt.plot(t_values,theta_v,label="True Model")

plt.figure(2)
plt.plot(t_values,Forces_vec[:,0],label="Fext")
plt.plot(t_values,Exp_matrix@Solution_ideal,label="fit")

plt.figure(3)
plt.plot(t_values,Exp_matrix@Solution_ideal-Forces_vec,label="Fext")

plt.figure(1)
plt.plot(Solution)

plt.show()