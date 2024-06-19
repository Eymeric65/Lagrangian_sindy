from function.Simulation import *
import matplotlib.pyplot as plt
# Setup problem

regression = True

#Exp1
# L1t = .2
# L2t = .2
# m_1 = .1
# m_2 = .1
# Y0 = np.array([[2, 0], [0, 0]])  # De la forme (k,2)
# Frotement = [-.4,-.2]
# M_span = [0.8,0.5] # Max span
# periode = 1 #
# periode_shift = 0.2
# Surfacteur=30 # La base
# N_periode = 5# In one periode they will be Surfacteur*N_Periode/Cat_len time tick

L1t = 1.
L2t = 1.
m_1 = .8
m_2 = .8
Y0 = np.array([[2, 0], [0, 0]])  # De la forme (k,2)
Frotement = [-1.4,-1.2]
M_span = [15.8,4.5] # Max span
periode = 1 #
periode_shift = 0.2
Surfacteur=10 # La base
N_periode = 5# In one periode they will be Surfacteur*N_Periode/Cat_len time tick

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



Lt = L1t + L2t
Substitution = {"g": 9.81, "l1": L1t, "m1": m_1, "l2": L2t, "m2": m_2}

L = (1 / 2 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 1 / 2 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
     * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(
    theta2))

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

Time_end = periode * Cat_len/N_periode

print("Temps de l'experience {} et longueur du Catalogue {} ".format(Time_end,Cat_len))

#----------------External Forces--------------------

F_ext_func = F_gen_opt(Coord_number,M_span,Time_end,periode,periode_shift,aug=15)
#F_ext_func = F_gen_c(M_span,periode_shift,Time_end,periode,Coord_number,aug=14)

# ---------------------------
troncature = 5
# Creation des schema de simulation

Acc_func,_ = Lagrangian_to_Acc_func(L, Symb, t, Substitution, fluid_f=Frotement)

Dynamics_system = Dynamics_f(Acc_func,F_ext_func)

t_values_w, phase = Run_RK45(Dynamics_system, Y0, Time_end,max_step=0.001)

q_d_v_g = np.gradient(phase[:,::2], t_values_w,axis=0,edge_order=2)

q_d_v = phase[:,1::2]

thetas_values_w = phase[:,::2]

print("Deviation gradient q0",np.linalg.norm(q_d_v[troncature:] - q_d_v_g[troncature:])/np.linalg.norm(q_d_v[troncature:])) #tres important

#Ajout du bruit

# --------------------------------------- Regression 1 -----------------------------------

distance_subsampling = 0.5

Indices_sub = Optimal_sampling(phase[:,:2],distance_subsampling)


fig, axs = plt.subplots(3, 4)


#Simulation temporelle

axs[0,0].set_title("q0")
axs[1,0].set_title("q1")

axs[0,0].plot(t_values_w,thetas_values_w[:,0])
axs[1,0].plot(t_values_w,thetas_values_w[:,1])

axs[1,2].set_title("temporal error")

# if(Model_Valid):
#
#     interp_other_sim = np.interp(t_values_w,t_values_v,thetas_values_v[:,0])
#     amp0 = thetas_values_w[:,0].max() -thetas_values_w[:,0].min()
#     axs[1,2].plot(t_values_w,(thetas_values_w[:,0]-interp_other_sim )/amp0, label="Q0")
#
#     interp_other_sim = np.interp(t_values_w,t_values_v,thetas_values_v[:,2])
#     amp1 = thetas_values_w[:, 1].max() - thetas_values_w[:, 1].min()
#     axs[1,2].plot(t_values_w,(thetas_values_w[:,1]-interp_other_sim)/amp1,label="Q1")



axs[2,2].set_title("Forces")
axs[2,2].plot(t_values_w,F_ext_func(t_values_w).T,label=["F_1", "F_2"])

axs[0,1].set_title("q0_d")
axs[1,1].set_title("q1_d")
axs[0,1].plot(t_values_w,q_d_v[:,0])
axs[1,1].plot(t_values_w,q_d_v[:,1])

axs[0,2].set_title("Phase q0")
axs[1,2].set_title("Phase q1")
axs[0,2].plot(thetas_values_w[:,0],q_d_v[:,0])
axs[1,2].plot(thetas_values_w[:,1],q_d_v[:,1])

axs[0,2].scatter(thetas_values_w[Indices_sub,0],q_d_v[Indices_sub,0])
axs[1,2].scatter(thetas_values_w[Indices_sub,1],q_d_v[Indices_sub,1])





# Regression error


plt.show()