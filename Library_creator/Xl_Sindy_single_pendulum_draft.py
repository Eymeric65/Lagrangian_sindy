import numpy as np

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

print("Symbol matrix",Symbol_matrix)

m, l, g = sp.symbols("m l g")

L = 0.5
Substitution = {"g": 9.81, "l": L, "m": 0.1}

Time_end = 14

#----------------External Forces--------------------

F_ext_time = np.array([0,2,4,6,8,10,Time_end])
F_ext_Value = np.array([0,1,-1,1,-1,0.1,0.1])*0.4

#F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)
F_ext_func = interpolate.PchipInterpolator(F_ext_time, F_ext_Value)
# ---------------------------

Y0 = np.array([[0, 0]])  # De la forme (k,2)

L_System = m*l**2/2*theta_d(t)**2+sp.cos(theta(t))*l*m*g

print("Base Lagrangian",L_System)

Dynamics_system = Dynamics_f(Lagrangian_to_Acc_func(L_System, Symbol_matrix, t, Substitution,fluid_f=0.0),F_ext_func)

t_values_w, thetas_values_w = Run_RK45(Dynamics_system, Y0, Time_end,max_step=0.005)

T_cut= 8

t_values = t_values_w[t_values_w<T_cut]
thetas_values = thetas_values_w[t_values_w<T_cut,:]

Noise_sigma= 10**-6

thetas_values = thetas_values + np.random.normal(0,Noise_sigma,thetas_values.shape)


# Generate Catalog

Coord_number = 1
Degree_function = 2

Symb = Symbol_Matrix_g(Coord_number,t)

print("Matrix Symbol : ",Symb)

function_catalog = [
    lambda x : Symb[1,x],
    lambda x : Symb[2,x],
    lambda x : sp.sin(Symb[1,x]),
    lambda x : sp.cos(Symb[1,x])
]

Catalog = Catalog_gen(function_catalog,Coord_number,2)
print(Catalog)


# Fitting

Recup_Threshold = 0.03

Catalog_subsample = 20

subs = len(t_values)//(Catalog_subsample*len(Catalog))



theta_v_s = thetas_values[::subs,0]
theta_v=thetas_values[:,0]
t_values_s = t_values[::subs]

Nb_t = len(t_values)

print("nombre de sample",Nb_t)


Exp_matrix = Catalog_to_experience_matrix(Nb_t,Coord_number,Catalog,Symb,t,theta_v,t_values,subsample=subs)

Puissance_coeff = np.linalg.norm(Exp_matrix,axis=0)**2

print("Coeff puissance",Puissance_coeff)

Forces_vec_s = Forces_vector(F_ext_func,t_values_s)
Forces_vec = Forces_vector(F_ext_func,t_values)

Solution_r,residual,rank,_ = np.linalg.lstsq(Exp_matrix,Forces_vec_s,rcond=None)

Solution = Solution_r

Solution_ideal = np.zeros(Solution.shape)
Solution_ideal[8,0] = 0.0125 # Ideal
Solution_ideal[3,0] = 0.4905
#Solution_ideal[3,0] = -3.05

plt.figure(3)
plt.plot(t_values_s,(Exp_matrix@Solution_ideal-Forces_vec_s),label="Fext")



Sol_ind = np.arange(len(Solution))

Nbind = len(Solution)
Nbind_prec = Nbind+1

#Retrieve the model !

Modele_fit = 0

for i in Sol_ind:
    Modele_fit = Modele_fit + Catalog[i] * Solution_r[i]
print("Model before fitting : ", Modele_fit[0])


while Nbind<Nbind_prec :
    Nbind_prec = Nbind

    poid = np.linalg.norm(Exp_matrix,axis=0)**2
    #print("weight",poid)

    Condition_value = np.abs(poid*Solution[:,0])
    #Condition_value = np.abs(Solution[:, 0])
    #print("cond",Condition_value)

    indices = np.argwhere(Condition_value> np.max(Condition_value)*Recup_Threshold)
    indices = indices[:,0]
    Sol_ind=Sol_ind[indices]
    Exp_matrix = Exp_matrix[:,indices]

    Solution, residual, rank, _ = np.linalg.lstsq(Exp_matrix, Forces_vec_s, rcond=None)

    Modele_fit = 0

    for i in range(len(Sol_ind)):
        Modele_fit = Modele_fit + Catalog[Sol_ind[i]] * Solution[i]
    print("Model fitting : ",Modele_fit[0])


    Nbind = len(Sol_ind)




#L_fit =Euler_lagranged(Modele_fit, Symb, t, 0)

print("Lagrangien de base : ",sp.simplify( L_System.subs(Substitution)))
print("Lagrangien fit : ",Modele_fit[0])

# Render

# plt.figure(0)
# plt.plot(t_values,theta_v,label="True Model")



q_d_v = np.gradient(thetas_values[:,0], t_values)


print("Deviation gradient",np.linalg.norm(q_d_v - thetas_values[:,1])) #tres important

q_dd_v = np.gradient(q_d_v, t_values)

plt.figure(0)

plt.plot(t_values_w,thetas_values_w[:,0],label="extended")
plt.plot(t_values,thetas_values[:,0],label="fit")
#plt.plot(t_values,q_d_v,label="fit")
#plt.plot(t_values,q_dd_v,label="fit")

# Dynamics_system_2 = Dynamics_f(Lagrangian_to_Acc_func(Modele_fit[0], Symb, t, Substitution,fluid_f=0.0),F_ext_func)
# t_values_v, thetas_values_v = Run_RK45(Dynamics_system_2, Y0, Time_end,max_step=0.01)
#
# plt.plot(t_values_v,thetas_values_v[:,0],"--",label="Exp")

plt.legend()


plt.figure(3)
#plt.plot(t_values,Forces_vec[:,0],label="Fext")
plt.plot(t_values_s,(Exp_matrix@Solution-Forces_vec_s),label="fit")

plt.legend()

#plt.plot(t_values,q_dd_v*40,label="fit2")




#
# plt.figure(1)
# plt.plot(Solution)

plt.show()