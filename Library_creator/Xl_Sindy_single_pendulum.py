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

Frotement = 0.3

Acc_func,_ = Lagrangian_to_Acc_func(L_System, Symb, t, Substitution,fluid_f=Frotement)

Dynamics_system = Dynamics_f(Acc_func,F_ext_func)

t_values_w, thetas_values_w = Run_RK45(Dynamics_system, Y0, Time_end,max_step=0.005)

# Restriction de la data

T_cut= Time_end*0.7

t_values = t_values_w[t_values_w<T_cut]
thetas_values = thetas_values_w[t_values_w<T_cut,:]

# Ajout du bruit

#Noise_sigma= 10**-2
#Noise_sigma=  2*10**-1
Noise_sigma=  3*10**-2
thetas_values_n = thetas_values #+ np.random.normal(0,Noise_sigma,thetas_values.shape)

# Generate Catalog

Degree_function = 2

print("Matrix Symbol : ",Symb)

function_catalog = [
    lambda x : Symb[1,x],
    lambda x : Symb[2,x],
    lambda x : sp.sin(Symb[1,x]),
    lambda x : sp.cos(Symb[1,x])
]

Catalog = Catalog_gen(function_catalog,Coord_number,Degree_function)

print(Catalog)

Forces_vec = Forces_vector(F_ext_func,t_values)

# Creation de la matrice de regression

subs = 1 # On réduit le nombre de données sans corrompre les dérivées du gradient

theta_v_s = thetas_values[::subs,0]
t_values_s = t_values[::subs]

Nb_t = len(t_values)

# q_d_v = np.gradient(thetas_values_n[:,1], t_values)
# print("Deviation gradient",np.linalg.norm(q_d_v - thetas_values_n[:,1])/Nb_t) #tres important

Forces_vec_s = Forces_vector(F_ext_func,t_values_s)



Is_Frottement = Frotement!=0

Exp_matrix = Catalog_to_experience_matrix(Nb_t,Coord_number,Catalog,Symb,t,thetas_values_n[:,0],t_values,subsample=subs,noise=Noise_sigma,Frottement=Is_Frottement)

# Fitting

# Normalisation de la Data

# Variance = np.var(Exp_matrix,axis=0)
# Mean = np.mean(Exp_matrix,axis=0)
#
# print(Variance)
#
# #Erase Nul Variance column
#
# reduction = np.argwhere(Variance != 0)
#
# print(Variance[reduction[:,0]])
# print(reduction)


Exp_norm,reduction,Variance = Normalize_exp(Exp_matrix)

# Exp_matrix_r = Exp_matrix[:,reduction[:,0]]
#
# Exp_norm = ( Exp_matrix_r - Mean[reduction[:,0]] )/Variance[reduction[:,0]]

print("Reducted shape : ",Exp_norm.shape)

# Y = Forces_vec[:,0]
#
# model = LassoCV(cv=5, random_state=0, max_iter=10000)
#
# # Fit model
# model.fit(Exp_norm, Y)
#
# alpha = model.alpha_
#
# print("Lasso alpha : ",alpha)
#
# # Set best alpha
# lasso_best = Lasso(alpha=model.alpha_)
# lasso_best.fit(Exp_norm, Y)
#
# coeff = lasso_best.coef_

coeff = Lasso_reg(Forces_vec,Exp_norm)

print("lasso coeff : ",coeff)

#print(coeff,coeff[:-1])

# if(Is_Frottement):
#
#     Solution_r = coeff[:-1]/Variance[reduction[:-1,0]]
#
#     Frottement_coeff = -coeff[-1]/Variance[reduction[-1,0]]
#
# else:
#     Solution_r = coeff/Variance[reduction[:,0]]

#Solution_r = np.reshape(Solution_r,(-1,1))

# Solution_r = coeff[:]/Variance[reduction[:,0]]
#
# Frottement_coeff = -Solution_r[-1]
#
# #Solution_r = np.reshape(Solution_r,(-1,1))
#
# Solution = np.zeros((len(Catalog)+1,1))
# #Solution[reduction[:-1,0],0]=Solution_r
# Solution[reduction[:,0],0]=Solution_r
Solution = Un_normalize_exp(coeff,Variance,reduction,Exp_matrix)
Frottement_coeff = -Solution[-1,0]

print("Solution et coeff",Solution,Frottement_coeff)

# Hard Treshold.....
# Recup_Threshold = 0.2
#
# _,Solution,_,opt_step = Hard_treshold_sparse_regression(Exp_matrix,Forces_vec_s,Catalog,Recup_Threshold=Recup_Threshold)
#----------------




Modele_fit = Make_Solution_exp(Solution[:-1,0],Catalog)

print("Solution Shape : ",Solution.shape)

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

#Solution_fr = np.concatenate((Solution,[[Frottement_coeff]]),axis=0)
Solution_fr = Solution
print(Solution_fr)

axs[0,1].plot(t_values_s,(Exp_matrix@Solution_ideal-Forces_vec_s),label="Solution ideal")
axs[0,1].plot(t_values_s,(Exp_matrix@Solution_fr-Forces_vec_s),label="Solution fit")
axs[0,1].legend()

#Optimisation Step


axs[1,1].set_title("Optimisation Step")

axs[1,2].set_title("Optimisation Step,Cond view")

# W_mult = len(opt_step)+2
#
# for i in range(len(opt_step)):
#
#     Solution_s,Cond,Solution_ind_s = opt_step[i]
#     Bar_height = np.abs(Solution_s)/np.max(np.abs(Solution_s))
#
#     Bar_height_cond = Cond/np.max(Cond)
#
#     axs[1,1].bar(Solution_ind_s,Bar_height[:,0],width=1/(i+1),label=str(i))
#     axs[1, 2].bar(Solution_ind_s, Bar_height_cond, width=1 / (i + 1), label=str(i))


# sn.heatmap(Correlation,ax=axs[2,0])
#
# sn.heatmap(Correlation_t,ax=axs[2,1])

Bar_height_ideal = np.abs(Solution_ideal)/np.max(np.abs(Solution_ideal))
Bar_height_found = np.abs(Solution_fr)/np.max(np.abs(Solution_fr))
axs[1,1].bar(np.arange(len(Solution_ideal)),Bar_height_ideal[:,0],width=1,label="True model")
axs[1,1].bar(np.arange(len(Solution_ideal)),Bar_height_found[:,0],width=0.5,label="Model Found")

axs[1,1].legend()
axs[1,2].legend()

# axs[0,2].set_title("Experience Matrix power")
# axs[0,2].bar(np.arange(len(Solution_ideal)),np.linalg.norm(Exp_matrix, axis=0)**2)
#
# axs[2,2].bar(np.arange(len(Solution_ideal)),np.var(Exp_matrix, axis=0),width=0.5)

plt.show()


