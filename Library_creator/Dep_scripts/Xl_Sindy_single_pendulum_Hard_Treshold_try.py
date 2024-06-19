import numpy as np

from function.Catalog_gen import *

from scipy import interpolate
from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *
from function.Optimization import  *

import seaborn as sn

np.random.seed(124)

t = sp.symbols("t")

#Generate Experience data :

Coord_number = 1
Symb = Symbol_Matrix_g(Coord_number,t)

theta = Symb[1,0]
theta_d = Symb[2,0]
theta_dd = Symb[3,0]

print("Symbol matrix",Symb)

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

print("Base Lagrangian",L_System)

print("Expanded Lagrangian",sp.expand(L_System.subs(Substitution)).args)

Acc_func,_ = Lagrangian_to_Acc_func(L_System, Symb, t, Substitution,fluid_f=0.0)

Dynamics_system = Dynamics_f(Acc_func,F_ext_func)

t_values_w, thetas_values_w = Run_RK45(Dynamics_system, Y0, Time_end,max_step=0.005)

# Restriction de la data

T_cut= 20

t_values = t_values_w[t_values_w<T_cut]
thetas_values = thetas_values_w[t_values_w<T_cut,:]

# Ajout du bruit

#Noise_sigma= 10**-2
Noise_sigma= 0
thetas_values_n = thetas_values #+ np.random.normal(0,Noise_sigma,thetas_values.shape)

# Generate Catalog


Degree_function = 3

print("Matrix Symbol : ",Symb)

function_catalog = [
    lambda x : Symb[1,x],
    lambda x : Symb[2,x],
    lambda x : sp.sin(Symb[1,x]),
    lambda x : sp.cos(Symb[1,x])
]

Catalog = Catalog_gen(function_catalog,Coord_number,Degree_function)
print(Catalog)


# Fitting

Recup_Threshold = 0.2

#Catalog_overfit = 20

subs = 1 # On réduit le nombre de données sans corrompre les dérivées du gradient

theta_v_s = thetas_values[::subs,0]
t_values_s = t_values[::subs]

Nb_t = len(t_values)

q_d_v = np.gradient(thetas_values_n[:,1], t_values)
print("Deviation gradient",np.linalg.norm(q_d_v - thetas_values_n[:,1])/Nb_t) #tres important


Exp_matrix = Catalog_to_experience_matrix(Nb_t,Coord_number,Catalog,Symb,t,thetas_values_n[:,0],t_values,subsample=subs,noise=Noise_sigma)

#Correlation = np.corrcoef(np.transpose(Exp_matrix))

#Correlation_t = np.corrcoef(Exp_matrix)

# Correlation = np.cov(np.transpose(Exp_matrix))
#
# Correlation_t = np.cov(Exp_matrix)


Forces_vec_s = Forces_vector(F_ext_func,t_values_s)

Forces_vec = Forces_vector(F_ext_func,t_values)

Modele_fit,Solution,_,opt_step = Hard_treshold_sparse_regression(Exp_matrix,Forces_vec_s,Catalog,Recup_Threshold=Recup_Threshold)

print("Solution Shape : ",Solution.shape)

print("Lagrangien de base : ",sp.simplify( L_System.subs(Substitution)))


Solution_ideal = Make_Solution_vec(L_System.subs(Substitution),Catalog)

print("Lagrangien fit : ",Modele_fit[0])


#Plotting

fig, axs = plt.subplots(3, 3)

fig.suptitle("Resultat Experience"+str(Noise_sigma))

#Simulation temporelle
axs[0,0].set_title("Resultat temporelle")

axs[0,0].plot(t_values_w,thetas_values_w[:,0],label="extended")
axs[0,0].plot(t_values,thetas_values[:,0]+np.random.normal(0,Noise_sigma,thetas_values[:,0].shape),label="fit")

Acc_func2 , Model_Valid =  Lagrangian_to_Acc_func(Modele_fit[0], Symb, t, Substitution,fluid_f=0.0)

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

axs[0,1].plot(t_values_s,(Exp_matrix@Solution_ideal-Forces_vec_s),label="Solution ideal")
axs[0,1].plot(t_values_s,(Exp_matrix@Solution-Forces_vec_s),label="Solution fit")
axs[0,1].legend()

#Optimisation Step


axs[1,1].set_title("Optimisation Step")

axs[1,2].set_title("Optimisation Step,Cond view")

W_mult = len(opt_step)+2

for i in range(len(opt_step)):

    Solution_s,Cond,Solution_ind_s = opt_step[i]
    Bar_height = np.abs(Solution_s)/np.max(np.abs(Solution_s))

    Bar_height_cond = Cond/np.max(Cond)

    axs[1,1].bar(Solution_ind_s,Bar_height[:,0],width=1/(i+1),label=str(i))
    axs[1, 2].bar(Solution_ind_s, Bar_height_cond, width=1 / (i + 1), label=str(i))


# sn.heatmap(Correlation,ax=axs[2,0])
#
# sn.heatmap(Correlation_t,ax=axs[2,1])

Bar_height = np.abs(Solution_ideal)/np.max(np.abs(Solution_ideal))
axs[1,1].bar(np.arange(len(Solution_ideal)),Bar_height[:,0],width=1/W_mult,label="True model")

axs[1,1].legend()
axs[1,2].legend()

axs[0,2].set_title("Experience Matrix power")
axs[0,2].bar(np.arange(len(Solution_ideal)),np.linalg.norm(Exp_matrix, axis=0)**2)

axs[2,2].bar(np.arange(len(Solution_ideal)),np.var(Exp_matrix, axis=0),width=0.5)

plt.show()


