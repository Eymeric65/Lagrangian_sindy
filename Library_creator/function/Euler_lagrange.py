import sympy as sp
import numpy as np

def Euler_lagranged(expr, Smatrix, t, qi):  #Euler Lagrange en symbolique take an expression of the lagrangian, an Symbol matrix, the symbolic for t, and the indice for the transformation
    dL_dq = sp.diff(expr, Smatrix[1, qi])
    dL_dq_d = sp.diff(expr, Smatrix[2, qi])
    d_dt = (sp.diff(dL_dq_d, t))

    for j in range(Smatrix.shape[1]):  # Time derivative replacement d/dt q -> q_d

        d_dt = d_dt.replace(sp.Derivative(Smatrix[1, j], t), Smatrix[2, j])
        d_dt = d_dt.replace(sp.Derivative(Smatrix[2, j], t), Smatrix[3, j])

    return dL_dq - d_dt

def Lagrangian_to_Acc_func(L, Symbol_matrix, t, Substitution,fluid_f = 0): # Turn the Lagrangian into the complete Array function
    Qk = Symbol_matrix.shape[1]
    Acc = np.zeros((Qk, 1), dtype="object")

    Valid = True

    for i in range(Qk):  # Derive the k expression for dynamics

        Dyn = Euler_lagranged(L, Symbol_matrix, t, i) - Symbol_matrix[0, i] - fluid_f*Symbol_matrix[2, i] # Add the Fext term
        #print("Dyn",Dyn)

        if( Symbol_matrix[3,i] in Dyn.atoms(sp.Function) ):

            Acc_s = sp.solve(Dyn, Symbol_matrix[3, i])
            #print("Acc_s",Acc_s)
            Acc_s = Acc_s[0]

            Acc[i, 0] = Acc_s.subs(Substitution)

        else:
            Valid = False
            break

        #Valid = Valid and  ( Symbol_matrix[3,i] in Dyn.atoms(sp.Function) )

    #print(Acc)

    Acc_lambda = sp.lambdify([Symbol_matrix], Acc)  # Lambdify under the input of Symbol_matrix

    return Acc_lambda,Valid

def Catalog_to_experience_matrix(Nt,Qt,Catalog,Sm,t,q_v,q_t,subsample=1,noise=0):

    Nt_s = Nt//subsample +1

    Exp_Mat = np.zeros(((Nt_s) * Qt, len(Catalog)))

    q_d_v = np.gradient(q_v,q_t)
    q_dd_v= np.gradient(q_d_v,q_t)

    q_matrix = np.zeros((Sm.shape[0],Sm.shape[1],Nt_s))

    q_matrix[1, :, :] = q_v[::subsample]
    q_matrix[2, :, :] = q_d_v[::subsample]
    q_matrix[3, :, :] = q_dd_v[::subsample]

    q_matrix = q_matrix + np.random.normal(0,noise,q_matrix.shape)

    for i in range(Qt):

        Catalog_lagranged = list(map(lambda x: Euler_lagranged(x, Sm,t,i), Catalog))
        Catalog_lambded = list(map(lambda x: sp.lambdify([Sm], x, modules="numpy"), Catalog_lagranged))

        # print(Catalog_lagranged)
        # print([Symb_l,Symb_d_l,Symb_dd_l])
        # print(Catalog_lambded)
        #
        # fun_l = Catalog_lambded[0]
        # print(fun_l([np.array([10,10])],[np.array([10,10])],[np.array([10,10])]))

        for j in range(len(Catalog_lambded)):
            # print(str(signature(Catalog_lambded[j])))
            Func_pick = Catalog_lambded[j]
            Exp_Mat[i * Nt_s:(i + 1) * (Nt_s), j] = Func_pick(q_matrix)



    return Exp_Mat