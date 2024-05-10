import numpy as np

from function.Catalog_gen import *

from scipy import interpolate
from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *
from function.Optimization import  *

import seaborn as sn

np.random.seed(1230)

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

# F_ext_time = np.array([0, 2, 4, 6, 8, Time_end])
# F_ext_Value = np.array([[0, 1, -1, 0, 0, 0],[0, 1, -1, 0, 0, 0]]) * 0.1  # De la forme (k,...)
#
# F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)

def F_ext(t):
    return np.array([np.cos(t*np.cos(t*2*np.pi/Time_end)*2*np.pi/10)*0.1,np.sin(t*np.cos(t*2*np.pi/Time_end)*2*np.pi/10)*0.1])

def F_ext2(t):
    return np.array([np.cos(t*3*np.cos(t*1.5*np.pi/Time_end)*2*np.pi/10)*0.11,np.sin(t*3*np.sin(t*1.5*np.pi/Time_end)*2*np.pi/10)*0.08])

def F_ext3(t):
    return np.array([np.cos(t*np.cos(t*2*np.pi/Time_end)*2*np.pi/10)*0.1,0.0*t])


F_ext_func = F_ext2
# ---------------------------

Y0 = np.array([[0, 0], [0, 0]])  # De la forme (k,2)

L = (1 / 2 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 1 / 2 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
     * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(
    theta2))

Frotement = [-0.0,-0.0]

Acc_func,_ = Lagrangian_to_Acc_func(L, Symb, t, Substitution,fluid_f=Frotement)

Dynamics_system = Dynamics_f(Acc_func,F_ext_func)

t_values_w, thetas_values_w = Run_RK45(Dynamics_system, Y0, Time_end,max_step=0.0005)

q_d_v = np.gradient(thetas_values_w[:,::2], t_values_w,axis=0)

test = thetas_values_w[:,1::2]

thetas_values_w = thetas_values_w[:,::2]


print("Deviation gradient q0",np.linalg.norm(q_d_v - test)) #tres important




# Restriction de la data

T_cut= Time_end*0.5

t_values = t_values_w[t_values_w<T_cut]
thetas_values = thetas_values_w[t_values_w<T_cut,:]



#Ajout du bruit

Noise_sigma=  3*10**-2 * 0
thetas_values_n = thetas_values

Degree_function = 4

Puissance_model= 2

function_catalog = [
    lambda x : Symb[1,x],
    lambda x : Symb[2,x],
    lambda x : sp.sin(Symb[1,x]),
    lambda x : sp.cos(Symb[1,x])
]

Catalog = np.array(Catalog_gen(function_catalog,Coord_number,Degree_function,puissance=Puissance_model))

Solution_ideal = Make_Solution_vec(sp.expand_trig(L.subs(Substitution)),Catalog)#,Frottement=Frotement)

S_index = (Solution_ideal != 0).nonzero()[0]

#add_new_comp = 15
add_new_comp = 60

count = np.array(range(len(Catalog)))

count = np.delete(count,S_index)

indices = np.random.randint(0,len(count)-1,add_new_comp)

res = np.concatenate((count[indices],S_index))

Solution_ideal = Solution_ideal[res,:]

#Solution_ideal[5,0] = 0

Catalog = Catalog[res]

print("Sol ideal",Solution_ideal.shape)
print("Sol ideal fact",len((Solution_ideal != 0).nonzero()[0]))



Nb_t = len(t_values)

print("Nombre de temps",Nb_t)





Is_Frottement = Frotement!=0

Surfacteur=len(Catalog)*5

Subsample = Nb_t//Surfacteur

print(Subsample)

#Subsample = 50

t_values_s = t_values[::Subsample]



Forces_vec = Forces_vector(F_ext_func,t_values_s)

print("forces_vec_shape",Forces_vec.shape)

Exp_matrix = Catalog_to_experience_matrix(Nb_t,Coord_number,Catalog,Symb,t,thetas_values_n,t_values,subsample=Subsample)

print(Exp_matrix.shape)

Exp_norm,reduction,Variance = Normalize_exp(Exp_matrix,null_effect=True)

coeff = Lasso_reg(Forces_vec,Exp_norm)

Solution = Un_normalize_exp(coeff,Variance,reduction,Exp_matrix)

#Hardtreshold
tr =  10**-3

Solution[np.abs(Solution)< np.max(np.abs(Solution))*tr] = 0

print("Erreur de resolution coeff :",np.linalg.norm( Solution-Solution_ideal)/np.linalg.norm(Solution_ideal))

#print(Solution,Solution_ideal)


Modele_fit = Make_Solution_exp(Solution[:,0],Catalog)

print("Lagrangien de base : ",sp.simplify( L.subs(Substitution)))

print("Lagrangien expanded : ",sp.expand_trig(L.subs(Substitution)))

print("Equation mouvement : ",Euler_lagranged(L,Symb,t,0)) #Juste
print("Equation mouvement : ",Euler_lagranged(L,Symb,t,1)) #Juste


print("Modele fit",Modele_fit)

print("Shape matrice experience",Exp_matrix.shape)

fig, axs = plt.subplots(3, 3)

fig.suptitle("Resultat Experience Double pendule"+str(Noise_sigma))


print("Sol ",Solution.shape)

Acc_func2 , Model_Valid =  Lagrangian_to_Acc_func(Modele_fit, Symb, t, Substitution,fluid_f=Frotement)

#Simulation temporelle

axs[0,0].set_title("Resultat temporelle q0")
axs[1,0].set_title("Resultat temporelle q1")

axs[0,0].plot(t_values_w,thetas_values_w[:,0],label="extended")
axs[0,0].plot(t_values,thetas_values[:,0]+np.random.normal(0,Noise_sigma,thetas_values[:,0].shape),label="fit")


axs[1,0].plot(t_values_w,thetas_values_w[:,1],label="extended")
axs[1,0].plot(t_values,thetas_values[:,1]+np.random.normal(0,Noise_sigma,thetas_values[:,0].shape),label="fit")



if (Model_Valid):
    Dynamics_system_2 = Dynamics_f(Acc_func2, F_ext_func)
    t_values_v, thetas_values_v = Run_RK45(Dynamics_system_2, Y0, Time_end, max_step=0.01)

    axs[0, 0].plot(t_values_v, thetas_values_v[:, 0], "--", label="Exp")

    axs[1, 0].plot(t_values_v, thetas_values_v[:, 2], "--", label="Exp")

axs[1,2].set_title("Erreur temporelle")
if(Model_Valid):

    interp_other_sim = np.interp(t_values_w,t_values_v,thetas_values_v[:,0])
    amp0 = thetas_values_w[:,0].max() -thetas_values_w[:,0].min()
    axs[1,2].plot(t_values_w,(thetas_values_w[:,0]-interp_other_sim )/amp0, label="Q0")

    interp_other_sim = np.interp(t_values_w,t_values_v,thetas_values_v[:,2])
    amp1 = thetas_values_w[:, 1].max() - thetas_values_w[:, 1].min()
    axs[1,2].plot(t_values_w,(thetas_values_w[:,1]-interp_other_sim)/amp1,label="Q1")

axs[0,0].legend()
axs[1,0].legend()

axs[1,2].legend()

axs[0,1].set_title("Erreur de regression")

print("Woaw",np.repeat(t_values,Coord_number).shape )


q_dd_v = np.gradient(q_d_v, t_values_w,axis=0)

axs[2,0].plot(t_values_w,q_d_v)
axs[2,0].plot(t_values_w,q_dd_v)

axs[0,1].plot(np.repeat(t_values_s,Coord_number)*2,(Exp_matrix@Solution_ideal-Forces_vec),label="Solution ideal")
axs[0,1].plot(np.repeat(t_values_s,Coord_number)*2,(Exp_matrix@Solution-Forces_vec),label="Solution fit")
axs[0,1].legend()

axs[0,2].set_title("Erreur de regression, comparaison")

axs[0,2].plot(np.repeat(t_values_s,Coord_number)*2,(Exp_matrix@Solution_ideal),label="Solution ideal")
axs[0,2].plot(np.repeat(t_values_s,Coord_number)*2,(Exp_matrix@Solution),label="Solution fit")
axs[0,2].plot(np.repeat(t_values_s,Coord_number)*2,(Forces_vec),label="Forces")
axs[0,2].legend()

axs[1,1].set_title("Optimisation Step")

Bar_height_ideal = np.abs(Solution_ideal)/np.max(np.abs(Solution_ideal))
Bar_height_found = np.abs(Solution)/np.max(np.abs(Solution))
axs[1,1].bar(np.arange(len(Solution_ideal)),Bar_height_ideal[:,0],width=1,label="True model")
axs[1,1].bar(np.arange(len(Solution_ideal)),Bar_height_found[:,0],width=0.5,label="Model Found")

axs[1,1].legend()


plt.show()