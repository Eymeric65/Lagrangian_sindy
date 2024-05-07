import sympy as sp
import numpy as np

def Catalog_gen_p(f_cat,qk,prof,im,jm): # MDR la multiplication c'est transitif, solved

    if prof==0 :
        return [1] #,f_cat[f].format(v_cat[v])
    else:
        ret = []
        for i in range(im,len(f_cat)):
            for j in range(jm,qk):
                res = Catalog_gen_p(f_cat,qk,prof-1,i,j)
                fun_p = f_cat[i]
                res_add = [res[l]*fun_p(j) for l in range(len(res))]

                ret += res_add

        return ret

def Catalog_gen(f_cat,qk,degre):

    Catalog = []

    for i in range(degre):
        Catalog += Catalog_gen_p(f_cat, qk, i + 1, 0, 0)

    return Catalog

def Symbol_Matrix_g(Coord_number,t):

    ret = np.zeros((4,Coord_number),dtype='object')
    ret[0,:]= [sp.Function("Fext{}".format(i))(t) for i in range(Coord_number)]
    ret[1, :] = [sp.Function("q{}".format(i))(t) for i in range(Coord_number)]
    ret[2, :] = [sp.Function("q{}_d".format(i))(t) for i in range(Coord_number)]
    ret[3, :] = [sp.Function("q{}_dd".format(i))(t) for i in range(Coord_number)]
    return ret

def Forces_vector(F_fun,t_v):

    return np.transpose(np.reshape(F_fun(t_v),(1,-1)))

def Make_Solution_vec(exp,Catalog,Frottement=0):

    exp_arg = sp.expand(exp).args

    Solution = np.zeros((len(Catalog)+int(Frottement!=0),1))

    for i in range(len(exp_arg)):

        for v in range(len(Catalog)):

            test = exp_arg[i]/Catalog[v]

            if(len(test.args)==0):

                Solution[v,0] = test

    if Frottement != 0:
        Solution[-1,0] = Frottement

    return Solution

def Make_Solution_exp(Solution,Catalog):

    Modele = 0

    for i in range(len(Solution)):

        Modele += Solution[i]*Catalog[i]

    return Modele





