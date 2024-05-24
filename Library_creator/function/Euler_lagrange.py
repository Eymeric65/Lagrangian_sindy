import numpy as np
import sympy as sp


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

    if len(fluid_f) ==0 :
        fluid_f = [0 for i in range(Qk)]
    Acc = np.zeros((Qk, 1), dtype="object")

    dyn = np.zeros((Qk,), dtype="object")

    Valid = True

    for i in range(Qk):  # Derive the k expression for dynamics #LOL les equations sont couplées en gros mdrrrrrrr

        Dyn = Euler_lagranged(L, Symbol_matrix, t, i) - Symbol_matrix[0, i] # + fluid_f[i]*Symbol_matrix[2, i]  # Add the Fext term

        # Hardcodage matrice de dissipation pour un systeme en chaine (interagit deux a deux)

        Dyn += fluid_f[i]*Symbol_matrix[2, i]

        if(i<(Qk-1)):
            Dyn += fluid_f[i+1]*Symbol_matrix[2, i]
            Dyn += - fluid_f[i+1]*Symbol_matrix[2, i+1]

        if(i>0):

            Dyn+= - fluid_f[i]*Symbol_matrix[2, i-1]

        # ------------------------------------

        if( Symbol_matrix[3,i] in Dyn.atoms(sp.Function) ):

            dyn[i] = Dyn.subs(Substitution)

        else:
            Valid = False
            break

    if(Valid):
        Solution_S = sp.solve(dyn,Symbol_matrix[3, :])
        if isinstance(Solution_S,dict) :

            Sol = list(Solution_S.values())
        else:
            Sol = list(Solution_S[0].values())

        Acc[:,0]= Sol #LA SOLUTION

    Acc_lambda = sp.lambdify([Symbol_matrix], Acc)  # Lambdify under the input of Symbol_matrix

    return Acc_lambda,Valid

def Catalog_to_experience_matrix(Nt,Qt,Catalog,Sm,t,q_v,q_t,subsample=1,noise=0,Frottement=False,troncature=0,q_d_v=[],q_dd_v=[]):
    #print(Nt)
    #print(Nt//subsample)
    #print(Nt/2 != Nt//2)

    Nt_s = len(q_v[troncature::subsample])

    Exp_Mat = np.zeros(((Nt_s) * Qt, len(Catalog)+int(Frottement)*Qt))

    if len(q_d_v) == 0 :
        q_d_v = np.gradient(q_v,q_t,axis=0)
    if len(q_dd_v) == 0:
        q_dd_v= np.gradient(q_d_v,q_t,axis=0)

    q_matrix = np.zeros((Sm.shape[0],Sm.shape[1],Nt_s))

    q_matrix[1, :, :] = np.transpose(q_v[troncature::subsample])
    q_matrix[2, :, :] = np.transpose(q_d_v[troncature::subsample])
    q_matrix[3, :, :] = np.transpose(q_dd_v[troncature::subsample])

    q_matrix = q_matrix + np.random.normal(0,noise,q_matrix.shape)

    for i in range(Qt):

        #print("Valeur Qt : ",i)
        Catalog_lagranged = list(map(lambda x: Euler_lagranged(x, Sm,t,i), Catalog))

        #print(Catalog_lagranged)

        #print("--------------")

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

        if(Frottement):
            Exp_Mat[i * Nt_s:(i + 1) * (Nt_s), len(Catalog_lambded)+i] = q_d_v[troncature::subsample,i]



    return Exp_Mat