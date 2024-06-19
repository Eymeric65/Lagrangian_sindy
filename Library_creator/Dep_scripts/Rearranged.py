import sympy as sp
import time
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import RK45
import numpy as np

# Paramètre de la génération !

General_Coord_Nb = 1

Degree_function = 2

Function_catalog = "{}(t) {}_dot(t) sp.sin({}(t)) sp.cos({}(t))"

Function_catalog = Function_catalog.split(" ")

#Génération des symboles

Var_letter = [ "q{}".format(i) for i in range(General_Coord_Nb)]
Var_letter_d = list(map(lambda x : x+"_dot",Var_letter))
Var_letter_dd = list(map(lambda x : x+"_ddot",Var_letter))

Var_symbols = sp.symbols(Var_letter+Var_letter_d+Var_letter_dd)

t = sp.symbols("t")

# Inscription des symboles dans une matrice de symbole et dans l'environnement courant
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

#Generation du Catalogue polynomiale
def Catalog_gen(f_cat,v_cat,prof,im,jm): # Generation de tous les couples polynomiaux
    if prof==0 :
        return [""]
    else:
        ret = []
        for i in range(im,len(f_cat)):
            for j in range(jm,len(v_cat)):
                res = Catalog_gen(f_cat,v_cat,prof-1,i,j)
                res_add = [res[l]+("*" if len(res[l]) else "" )+f_cat[i].format(v_cat[j]) for l in range(len(res))]
                ret += res_add
        return ret


Catalog = []

for i in range(Degree_function):
    Catalog+= Catalog_gen(Function_catalog,Var_letter,i+1,0,0)

Catalog = list(map(lambda x : eval(x),Catalog)) # Conversion en symbolic
print("Catalog len : %s " % len(Catalog))


#Generation de l'expérience de base (Pourra être remplacer par Mujoco plus tard)

def Euler_lagranged(expr,q,q_d,q_dd,t): #Euler Lagrange en symbolique
    dL_dq = sp.diff(expr, q)
    dL_dq_d = sp.diff(expr,q_d)
    d_dt = sp.diff(dL_dq_d,t).replace(sp.Derivative(q,t),q_d).replace(sp.Derivative(q_d,t),q_dd)
    return dL_dq - d_dt

def Dynamics_f(Acc,Fext): # Createur de la fonction a mettre dans l'intégrateur

    def func(t,State):

        State = np.reshape(State,(-1,2))
        ret = np.zeros(State.shape)
        ret[:,0]=State[:,1]
        ret[:,1]=Acc(State[:,0],State[:,1],Fext(t))
        return np.reshape(ret,(-1,))

    return func