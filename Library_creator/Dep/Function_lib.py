import sympy as sp
import time
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import RK45
import numpy as np

#Il faut implémenter le hard threshold multiple, voir avec LASSO :O

General_Coord_Nb = 1

Degree_function = 2

Function_catalog = "{}(t) {}_dot(t) sp.sin({}(t)) sp.cos({}(t))"

Function_catalog = Function_catalog.split(" ")

Var_letter = [ "q{}".format(i) for i in range(General_Coord_Nb)]

Var_letter_d = list(map(lambda x : x+"_dot",Var_letter))

Var_letter_dd = list(map(lambda x : x+"_ddot",Var_letter))

Var_symbols = sp.symbols(Var_letter+Var_letter_d+Var_letter_dd)

t = sp.symbols("t")

#Maybe better way to Use symbollic https://stackoverflow.com/questions/7006626/how-to-calculate-expression-using-sympy-in-python
Symb_l = []
for symb in Var_letter:

    exec("{0} = sp.Function(\"{0}\")".format(symb))
    Symb_l += [eval("{0}(t)".format(symb))]



Symb_d_l = []
for symb in Var_letter_d:

    exec("{0} = sp.Function(\"{0}\")".format(symb))
    Symb_d_l += [eval("{0}(t)".format(symb))]

Symb_dd_l = []
for symb in Var_letter_dd:

    exec("{0} = sp.Function(\"{0}\")".format(symb))
    Symb_dd_l += [eval("{0}(t)".format(symb))]

Symbol_matrix = np.array([Symb_l,Symb_d_l,Symb_dd_l])

print(Symbol_matrix)

print(Var_symbols)

Catalog = []

#Implementation naive de la creation de bibliotheque... Une approche plus fine pourrait permettre d'atteindre plus de degres de liberte
def Catalog_gen_int(f_cat,v_cat,prof): # MDR la multiplication c'est transitif

    if prof==0 :
        return [""] #,f_cat[f].format(v_cat[v])
    else:
        ret = []
        for i in range(len(f_cat)):
            for j in range(len(v_cat)):
                res = Catalog_gen(f_cat,v_cat,prof-1)

                res_add = [res[l]+("*" if len(res[l]) else "" )+f_cat[i].format(v_cat[j]) for l in range(len(res))]

                ret += res_add

        return ret

def Catalog_gen(f_cat,v_cat,prof,im,jm): # MDR la multiplication c'est transitif, solved

    if prof==0 :
        return [""] #,f_cat[f].format(v_cat[v])
    else:
        ret = []
        for i in range(im,len(f_cat)):
            for j in range(jm,len(v_cat)):
                res = Catalog_gen(f_cat,v_cat,prof-1,i,j)

                res_add = [res[l]+("*" if len(res[l]) else "" )+f_cat[i].format(v_cat[j]) for l in range(len(res))]

                ret += res_add

        return ret

def Euler_lagranged(expr,q,q_d,q_dd): #Fonction qui fait bien le taff

    dL_dq = sp.diff(expr, q)

    dL_dq_d = sp.diff(expr,q_d)

    d_dt = sp.diff(dL_dq_d,t).replace(sp.Derivative(q,t),q_d).replace(sp.Derivative(q_d,t),q_dd)

    return dL_dq - d_dt

def Dynamics_f(Acc,Fext):

    def func(t,State):

        State = np.reshape(State,(-1,2))

        ret = np.zeros(State.shape)
        ret[:,0]=State[:,1]
        ret[:,1]=Acc(State[:,0],State[:,1],Fext(t))
        return np.reshape(ret,(-1,))

    return func

start_time = time.time()

Catalog = []

for i in range(Degree_function):
    Catalog+= Catalog_gen(Function_catalog,Var_letter,i+1,0,0)

Catalog = list(map(lambda x : eval(x),Catalog))

print("Catalog len : %s " % len(Catalog),"--- %s seconds ---" % (time.time() - start_time))

# Experience Single Pendulum

F_ext = sp.symbols("F_ext")

theta = sp.Function("theta")
theta_d = sp.Function("theta_d")
theta_dd = sp.Function("theta_dd")

m,l,g = sp.symbols("m l g")

Substitution = {"g":9.81,"l":0.2,"m":0.1}

Time_end = 14

F_ext_time = np.array([0,2,4,6,8,Time_end])
F_ext_Value = np.array([0,1,-1,1,0,0])*0.03
F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)

Y0 = np.array([[0,0]])
Y0 = np.reshape(Y0,(-1,))

L = m*l**2/2*theta_d(t)**2+sp.cos(theta(t))*l*m*g

Dynamics = sp.simplify(Euler_lagranged(L,theta(t),theta_d(t),theta_dd(t)))+F_ext #- 0.1*theta_d(t)

Acc = sp.solve(Dynamics,theta_dd(t))[0]

Acc_sub = Acc.subs(Substitution)

print("Exp Model : ",Acc_sub)

print(Catalog)

Acc_lambda = sp.lambdify([theta(t), theta_d(t), F_ext], Acc_sub)

System_func = Dynamics_f(Acc_lambda, F_ext_func)

Model = RK45(System_func,0,Y0,Time_end ,0.01, 0.001, np.e**-6)

# collect data
t_values = []
theta_values = []
for i in range(1000):
    # get solution step state
    Model.step()
    t_values.append(Model.t)
    theta_values.append(Model.y[0])
    # break loop after modeling is finished
    if Model.status == 'finished':
        print("End step : ",i)
        break

t_values = np.array(t_values)
theta_d_values = np.gradient(theta_values)
theta_dd_values = np.gradient(theta_d_values)

# Generation du Dataset pour regression

Nb_t = len(t_values)

Exp_Mat = np.zeros((len(t_values)*General_Coord_Nb,len(Catalog)))

for i in range(General_Coord_Nb):

    Catalog_lagranged = list(map(lambda x : Euler_lagranged(x,Symb_l[i],Symb_d_l[i],Symb_dd_l[i]),Catalog))
    Catalog_lambded = list(map(lambda x : sp.lambdify([Symbol_matrix],x,modules="numpy"),Catalog_lagranged))


    # print(Catalog_lagranged)
    # print([Symb_l,Symb_d_l,Symb_dd_l])
    # print(Catalog_lambded)
    #
    # fun_l = Catalog_lambded[0]
    # print(fun_l([np.array([10,10])],[np.array([10,10])],[np.array([10,10])]))

    for j in range(len(Catalog_lambded)) :
         #print(str(signature(Catalog_lambded[j])))
         Exp_Mat[i*Nb_t:(i+1)*(Nb_t),j] = Catalog_lambded[j](np.array([[theta_values],[theta_d_values],[theta_dd_values]]))

print("te",F_ext_func(t_values).shape)

Solution,residual,rank,_ = np.linalg.lstsq(Exp_Mat,F_ext_func(t_values),rcond=None)

#Retrieve the model !

Recup_Threshold = 0.1

max_coeff = np.max(Solution)

Solution_thr = np.where(Solution > max_coeff*Recup_Threshold, Solution, 0)

Modele_fit = 0

for i in range(len(Catalog)):

    Modele_fit = Modele_fit + Catalog[i]*Solution_thr[i]

Modele_fit = sp.simplify(Modele_fit)

print("Lagrangien de base : ",L.subs(Substitution))
print("Lagrangien fit : ",Modele_fit)



plt.figure(0)
plt.plot(t_values,theta_values,label="True Model")

plt.figure(2)
plt.plot(t_values,F_ext_func(t_values),label="Fext")
plt.plot(t_values,Exp_Mat@Solution,label="fit")

plt.figure(3)
plt.plot(t_values,Exp_Mat@Solution-F_ext_func(t_values),label="Fext")

plt.figure(1)
plt.plot(Solution)

plt.show()


