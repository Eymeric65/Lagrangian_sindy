from function.Simulation import *
import matplotlib.pyplot as plt
# Setup problem

L1t = .2
L2t = .2
m_1 = .1
m_2 = .1

Y0 = np.array([[0, 0], [0, 0]])  # De la forme (k,2)

Frotement = [-1.62,-1.62]

# Force variable
periode = 3 #
periode_shift = 0.5

M_span = [0.0,0.0] # Max span

N_periode = 4# In one periode they will be Surfacteur*N_Periode/Cat_len time tick


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

L1t = 1
L2t = 1
Lt = L1t + L2t
Substitution = {"g": 9.81, "l1": L1t, "m1": 1, "l2": L2t, "m2": 1}

L = (1 / 2 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 1 / 2 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
     * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(
    theta2))

Y0 = np.array([[0, 0], [0, 0]])  # De la forme (k,2)


Solve = False

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

Catalog = np.concatenate((Catalog_crossed.flatten(),Catalog_sub_1,Catalog_sub_2))

Solution_ideal = Make_Solution_vec(sp.expand_trig(L.subs(Substitution)),Catalog,Frottement=Frotement)#,Frottement=Frotement)

Cat_len = len(Catalog)

#--------------------------

# Creation des forces

# Parametre
Surfacteur=Cat_len*30 # La base



#Time_end = periode*10


#Time_end = periode * 100
Time_end = periode * Cat_len/N_periode
#Time_end = periode * 120

print("Temps de l'experience {} et longueur du Catalogue {} ".format(Time_end,Cat_len))

NbTry = 1

T_cut = Time_end * 0.7

#M_span=1.2



#----------------External Forces--------------------


np.random.seed(123)
F_ext_func = F_gen_opt(Coord_number,M_span,Time_end,periode,periode_shift,aug=50)
#F_ext_func = F_gen_c(M_span,periode_shift,Time_end,periode,Coord_number,aug=14)

# ---------------------------
troncature = 5
# Creation des schema de simulation

Acc_func,_ = Lagrangian_to_Acc_func(L, Symb, t, Substitution, fluid_f=Frotement)

Dynamics_system = Dynamics_f(Acc_func,F_ext_func)

t_values_w, thetas_values_w = Run_RK45(Dynamics_system, Y0, Time_end,max_step=0.002)

q_d_v_g = np.gradient(thetas_values_w[:,::2], t_values_w,axis=0,edge_order=2)

q_d_v = thetas_values_w[:,1::2]

thetas_values_w = thetas_values_w[:,::2]

print("Deviation gradient q0",np.linalg.norm(q_d_v[troncature:] - q_d_v_g[troncature:])/np.linalg.norm(q_d_v[troncature:])) #tres important

#Ajout du bruit

#Regression
Noise_sigma=  10**-3*0

Nb_t = len(t_values_w)

Subsample = Nb_t//Surfacteur

print(Subsample)
np.random.seed(123)
Solution,Exp_matrix,t_values_s = Execute_Regression(t_values_w,thetas_values_w,t,Symb,Catalog,F_ext_func,Subsample=Subsample,q_d_v=q_d_v)

#Fin Regression

Erreur = np.linalg.norm( Solution/np.max(Solution)-Solution_ideal/np.max(Solution_ideal))/np.linalg.norm(Solution_ideal/np.max(Solution_ideal))

print("Erreur de resolution coeff :",Erreur)
print("sparsity : ",np.sum(np.where(np.abs(Solution) > 0,1,0)))


Modele_fit = Make_Solution_exp(Solution[:,0],Catalog,Frottement=len(Frotement))

print("Modele fit",Modele_fit)

Modele_ideal = Make_Solution_exp(Solution_ideal[:,0],Catalog,Frottement=len(Frotement))

print("Modele ideal",Modele_ideal)

fig, axs = plt.subplots(3, 3)

fig.suptitle("Resultat Experience Double pendule"+str(Noise_sigma))


if Solve :

    Acc_func2 , Model_Valid = Lagrangian_to_Acc_func(Modele_fit, Symb, t, Substitution,
                                                     fluid_f=Solution[-len(Frotement):, 0])

else :
    Model_Valid = False

#Simulation temporelle

axs[0,0].set_title("q0")
axs[1,0].set_title("q1")

axs[0,0].plot(t_values_w,thetas_values_w[:,0],label="extended")
axs[0,0].plot(t_values_w,thetas_values_w[:,0]+np.random.normal(0,Noise_sigma,thetas_values_w[:,0].shape),label="fit")


axs[1,0].plot(t_values_w,thetas_values_w[:,1],label="extended")
axs[1,0].plot(t_values_w,thetas_values_w[:,1]+np.random.normal(0,Noise_sigma,thetas_values_w[:,0].shape),label="fit")



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

Forces_vec= Forces_vector(F_ext_func,t_values_s)

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