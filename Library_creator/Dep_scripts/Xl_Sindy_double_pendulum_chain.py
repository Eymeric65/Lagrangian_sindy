import numpy as np

from function.Catalog_gen import *

from scipy import interpolate
from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *
from function.Optimization import  *

import seaborn as sn

#np.random.seed(1230)

# Setup probleme

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

L = (1 / 2 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 1 / 2 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
     * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(
    theta2))

Y0 = np.array([[0, 0], [0, 0]])  # De la forme (k,2)

Frotement = [-0.12,-0.12]

# -------------------------------

# Creation catalogue

Degree_function = 4

Puissance_model= 2

#Suited catalog creation

function_catalog_1 = [
    lambda x: Symb[2,x]
]

function_catalog_2 = [
     lambda x : sp.sin(Symb[1,x]),
     lambda x : sp.cos(Symb[1,x])
]

Catalog_sub_1 = np.array(Catalog_gen(function_catalog_1,Coord_number,2))

Catalog_sub_2 = np.array(Catalog_gen(function_catalog_2,Coord_number,2))

Catalog_crossed = np.outer(Catalog_sub_2,Catalog_sub_1)

print(Catalog_sub_1.shape,Catalog_sub_2.shape,Catalog_crossed.shape)


Catalog = np.concatenate((Catalog_crossed.flatten(),Catalog_sub_1,Catalog_sub_2))

Solution_ideal = Make_Solution_vec(sp.expand_trig(L.subs(Substitution)),Catalog,Frottement=Frotement)#,Frottement=Frotement)

# function_catalog = [
#     lambda x : Symb[1,x],
#     lambda x : Symb[2,x],
#     lambda x : sp.sin(Symb[1,x]),
#     lambda x : sp.cos(Symb[1,x])
# ]
#
# Catalog = np.array(Catalog_gen(function_catalog,Coord_number,Degree_function,puissance=Puissance_model))
#

#
# S_index = (Solution_ideal[:-len(Frotement)] != 0).nonzero()[0]
#
# #add_new_comp = 15
# add_new_comp = 100 # 9 marchait bien
#
# count = np.array(range(len(Catalog)))
#
# count = np.delete(count,S_index)
#
# indices = np.random.randint(0,len(count)-1,add_new_comp)
#
# #res = np.concatenate((count[indices],S_index,np.arange(len(Frotement))+len(Catalog)))
# res = np.concatenate((count,S_index,np.arange(len(Frotement))+len(Catalog)))
#
# Solution_ideal = Solution_ideal[res,:]
#
#
#
# Catalog = Catalog[res[:-len(Frotement)]]

Cat_len = len(Catalog)

#--------------------------

# Creation des forces

# Parametre
Surfacteur=Cat_len*25 # La base

periode = 0.8 #

#Time_end = periode*10

N_periode = 10# In one periode they will be Surfacteur*N_Periode/Cat_len time tick

#Time_end = periode * 100
Time_end = periode * Cat_len/N_periode
#Time_end = periode * 120

print("Temps de l'experience {} et longueur du Catalogue {} ".format(Time_end,Cat_len))

NbTry = 1

#Noise_list=[10**-4,10**-3,10**-2,5*10**-2,10**-1,3*10**-1,6*10**-1,1]

Try_list = np.ones((NbTry,))

T_cut = Time_end * 0.7

#M_span=1.2
M_span = 3 # Max span

periode_shift = 0.1

#------------------

np.random.seed()

def F_gen(M_span,periode_shift,Time_end,periode):

    F_ext_time = np.arange(0,Time_end,periode)

    f_nbt = len(F_ext_time)

    F_ext_time = F_ext_time + (np.random.random((f_nbt,))-0.5)*2*periode_shift

    F_ext_Value = (np.random.random((Coord_number,f_nbt))-0.5)*2*M_span

    return interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)

def concat_f(arr):

    def ret(t):

        out = arr[0](t)

        for f in arr[1:]:

            out += f(t)

        return out

    return ret


for jhk in range(len(Try_list)):

#----------------External Forces--------------------

    f_arr = []

    aug = 50

    for i in range(1,aug):

        f_arr += [F_gen(M_span/(1+np.log(aug))/(i),periode_shift/i,Time_end,periode/i)]




    F_ext_func = concat_f(f_arr)

    # ---------------------------
    troncature = 5
    # Creation des schema de simulation

    Acc_func,_ = Lagrangian_to_Acc_func(L, Symb, t, Substitution, fluid_f=Frotement)

    Dynamics_system = Dynamics_f(Acc_func,F_ext_func)

    t_values_w, thetas_values_w = Run_RK45(Dynamics_system, Y0, Time_end,max_step=0.0005)

    q_d_v = np.gradient(thetas_values_w[:,::2], t_values_w,axis=0,edge_order=2)

    test = thetas_values_w[:,1::2]

    thetas_values_w = thetas_values_w[:,::2]

    print("Deviation gradient q0",np.linalg.norm(q_d_v[troncature:] - test[troncature:])/np.linalg.norm(test[troncature:])) #tres important

    # Restriction de la data

    t_values = t_values_w[t_values_w<T_cut]
    thetas_values = thetas_values_w[t_values_w<T_cut,:]

    #Ajout du bruit

    Noise_sigma=  10**-3
    #Noise_sigma =Noise_list[jhk]

    thetas_values_n = thetas_values

    Nb_t = len(t_values)

    Is_Frottement = Frotement!=0

    Subsample = Nb_t//Surfacteur

    t_values_s = t_values[troncature::Subsample]

    Forces_vec = Forces_vector(F_ext_func,t_values_s)

    Exp_matrix = Catalog_to_experience_matrix(Nb_t,Coord_number,Catalog,Symb,t,thetas_values_n,t_values,subsample=Subsample,Frottement=(len(Frotement)>0),troncature=troncature,noise=Noise_sigma)

    Exp_norm,reduction,Variance = Normalize_exp(Exp_matrix,null_effect=True)

    coeff = Lasso_reg(Forces_vec,Exp_norm)

    Solution = Un_normalize_exp(coeff,Variance,reduction,Exp_matrix)

    #Hardtreshold
    tr =  10**-3

    Solution[np.abs(Solution)< np.max(np.abs(Solution))*tr] = 0

    Erreur = np.linalg.norm( Solution/np.max(Solution)-Solution_ideal/np.max(Solution_ideal))/np.linalg.norm(Solution_ideal/np.max(Solution_ideal))

    print("Erreur de resolution coeff :",Erreur)
    print("sparsity : ",np.sum(np.where(Solution > 0,1,0)))

    Try_list[jhk] = Erreur

    # if Erreur < 0.08:
    #     print("yipee")
    #     break



print("experience finale : ",Try_list)
print("experience finale MEAN : ",np.nanmean(Try_list))


Modele_fit = Make_Solution_exp(Solution[:,0],Catalog,Frottement=len(Frotement))

print("Modele fit",Modele_fit)

Modele_ideal = Make_Solution_exp(Solution_ideal[:,0],Catalog,Frottement=len(Frotement))

print("Modele ideal",Modele_ideal)

fig, axs = plt.subplots(3, 3)

fig.suptitle("Resultat Experience Double pendule"+str(Noise_sigma))


Acc_func2 , Model_Valid = Lagrangian_to_Acc_func(Modele_fit, Symb, t, Substitution,
                                                 fluid_f=Solution[-len(Frotement):, 0])



#Simulation temporelle


axs[0,0].set_title("q0")
axs[1,0].set_title("q1")

axs[0,0].plot(t_values_w,thetas_values_w[:,0],label="extended")
axs[0,0].plot(t_values,thetas_values[:,0]+np.random.normal(0,Noise_sigma,thetas_values[:,0].shape),label="fit")


axs[1,0].plot(t_values_w,thetas_values_w[:,1],label="extended")
axs[1,0].plot(t_values,thetas_values[:,1]+np.random.normal(0,Noise_sigma,thetas_values[:,0].shape),label="fit")



if (Model_Valid):
    Dynamics_system_2 = Dynamics_f(Acc_func2, F_ext_func)
    t_values_v, thetas_values_v = Run_RK45(Dynamics_system_2, Y0, Time_end, max_step=0.05)

    axs[0, 0].plot(t_values_v, thetas_values_v[:, 0], "--", label="found model")

    axs[1, 0].plot(t_values_v, thetas_values_v[:, 2], "--", label="found model")

axs[1,2].set_title("temporal error")
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

axs[0,1].set_title("Regression error")

axs[2,1].plot(t_values_w,F_ext_func(t_values_w).T)


q_dd_v = np.gradient(q_d_v, t_values_w,axis=0)

axs[2,0].plot(t_values_w,q_d_v)
axs[2,0].plot(t_values_w,q_dd_v)

axs[0,1].plot(np.repeat(t_values_s,Coord_number)*2,(Exp_matrix@Solution_ideal-Forces_vec),label="ideal solution")
axs[0,1].plot(np.repeat(t_values_s,Coord_number)*2,(Exp_matrix@Solution-Forces_vec),label="fit solution")
axs[0,1].legend()

axs[0,2].set_title("Regression error, comparison")

axs[0,2].plot(np.repeat(t_values_s,Coord_number)*2,(Exp_matrix@Solution_ideal),label="ideal solution")
axs[0,2].plot(np.repeat(t_values_s,Coord_number)*2,(Exp_matrix@Solution),label="fit solution")
axs[0,2].plot(np.repeat(t_values_s,Coord_number)*2,(Forces_vec),label="forces")
axs[0,2].legend()

axs[1,1].set_title("Model retrieved")

Bar_height_ideal = np.abs(Solution_ideal)/np.max(np.abs(Solution_ideal))
Bar_height_found = np.abs(Solution)/np.max(np.abs(Solution))
axs[1,1].bar(np.arange(len(Solution_ideal)),Bar_height_ideal[:,0],width=1,label="True model")
axs[1,1].bar(np.arange(len(Solution_ideal)),Bar_height_found[:,0],width=0.5,label="Model Found")

axs[1,1].legend()


plt.show()