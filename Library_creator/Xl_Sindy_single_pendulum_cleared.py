import numpy as np

from function.Catalog_gen import *

from scipy import interpolate
from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *
from function.Optimization import  *

import seaborn as sn



#np.random.seed(124)

t = sp.symbols("t")

#Generate Experience data :

Coord_number = 1
Symb = Symbol_Matrix_g(Coord_number,t)

theta = Symb[1,0]
theta_d = Symb[2,0]
theta_dd = Symb[3,0]

#print("Symbol matrix",Symb)

m, l, g = sp.symbols("m l g")

L = 0.5
Substitution = {"g": 9.81, "l": L, "m": 0.1}

Time_end = 30

#----------------External Forces--------------------

F_ext_time = np.array([0,2,4,6,8,10,15,20,Time_end])
F_ext_Value = np.array([0,1,-1,1,-1,0.1,1,-1,0.1])*0.4

#F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)
F_ext_func = interpolate.PchipInterpolator(F_ext_time, F_ext_Value)
# ---------------------------

Y0 = np.array([[0, 0]])  # De la forme (k,2)

L_System = m*l**2/2*theta_d**2+sp.cos(theta)*l*m*g

#print("Base Lagrangian",L_System)

#print("Expanded Lagrangian",sp.expand(L_System.subs(Substitution)).args)

Frotement = [-0.3]

Acc_func,_ = Lagrangian_to_Acc_func(L_System, Symb, t, Substitution,fluid_f=Frotement)

Dynamics_system = Dynamics_f(Acc_func,F_ext_func)

t_values_w, thetas_values_w = Run_RK45(Dynamics_system, Y0, Time_end,max_step=0.005)

q_d_v = np.gradient(thetas_values_w[:,0], t_values_w)
print("Deviation gradient",np.linalg.norm(q_d_v - thetas_values_w[:,1])) #tres important


# Restriction de la data

T_cut= Time_end*0.7

t_values = t_values_w[t_values_w<T_cut]
thetas_values = thetas_values_w[t_values_w<T_cut,:]

# Ajout du bruit

Noise_sigma=  3*10**-2 *0
thetas_values_n = thetas_values

Degree_function = 2

#print("Matrix Symbol : ",Symb)

function_catalog = [
    lambda x : Symb[1,x],
    lambda x : Symb[2,x],
    lambda x : sp.sin(Symb[1,x]),
    lambda x : sp.cos(Symb[1,x])
]

Catalog = Catalog_gen(function_catalog,Coord_number,Degree_function)

print(Catalog)


Nb_t = len(t_values)

Forces_vec = Forces_vector(F_ext_func,t_values)

Is_Frottement = Frotement!=0

Exp_matrix = Catalog_to_experience_matrix(Nb_t,Coord_number,Catalog,Symb,t,thetas_values_n[:,0],t_values,noise=Noise_sigma,Frottement=Is_Frottement)

# Fitting

Exp_norm,reduction,Variance = Normalize_exp(Exp_matrix)


coeff = Lasso_reg(Forces_vec,Exp_norm)

Solution = Un_normalize_exp(coeff,Variance,reduction,Exp_matrix)
Frottement_coeff = [Solution[-1,0]]

Modele_fit = Make_Solution_exp(Solution[:-1,0],Catalog)


print("Lagrangien de base : ",sp.simplify( L_System.subs(Substitution)))


Solution_ideal = Make_Solution_vec(L_System.subs(Substitution),Catalog,Frottement=Frotement)

print("Lagrangien fit : ",Modele_fit)


#Plotting

fig, axs = plt.subplots(3, 3)

fig.suptitle("Resultat Experience"+str(Noise_sigma))

#Simulation temporelle

axs[0,0].set_title("Resultat temporelle")

axs[0,0].plot(t_values_w,thetas_values_w[:,0],label="extended")
axs[0,0].plot(t_values,thetas_values[:,0]+np.random.normal(0,Noise_sigma,thetas_values[:,0].shape),label="fit")

Acc_func2 , Model_Valid =  Lagrangian_to_Acc_func(Modele_fit, Symb, t, Substitution,fluid_f=Frottement_coeff)

if(Model_Valid):

    Dynamics_system_2 = Dynamics_f(Acc_func2,F_ext_func)
    t_values_v, thetas_values_v = Run_RK45(Dynamics_system_2, Y0, Time_end,max_step=0.01)

    axs[0,0].plot(t_values_v,thetas_values_v[:,0],"--",label="Exp")

axs[0,0].legend()

#Erreur sur temporelle

axs[1,0].set_title("Erreur temporelle")
if(Model_Valid):

    interp_other_sim = np.interp(t_values_w,t_values_v,thetas_values_v[:,0])

    axs[1,0].plot(t_values_w,thetas_values_w[:,0]-interp_other_sim)

#Erreur de regression

axs[0,1].set_title("Erreur de regression")

axs[0,1].plot(t_values,(Exp_matrix@Solution_ideal-Forces_vec),label="Solution ideal")
axs[0,1].plot(t_values,(Exp_matrix@Solution-Forces_vec),label="Solution fit")
axs[0,1].legend()

#Optimisation Step

axs[1,1].set_title("Optimisation Step")

Bar_height_ideal = np.abs(Solution_ideal)/np.max(np.abs(Solution_ideal))
Bar_height_found = np.abs(Solution)/np.max(np.abs(Solution))
axs[1,1].bar(np.arange(len(Solution_ideal)),Bar_height_ideal[:,0],width=1,label="True model")
axs[1,1].bar(np.arange(len(Solution_ideal)),Bar_height_found[:,0],width=0.5,label="Model Found")

axs[1,1].legend()

plt.show()


