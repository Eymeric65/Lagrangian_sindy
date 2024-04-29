import numpy as np

def Condition_val(Exp,solution):

    return np.abs((np.linalg.norm(Exp, axis=0)**2 * solution[:, 0]* solution[:, 0]))

def Hard_treshold_sparse_regression(Exp_matrix,Forces_vec_s,Catalog,cond=Condition_val,Recup_Threshold = 0.03):

    Solution_r, residual, rank, _ = np.linalg.lstsq(Exp_matrix, Forces_vec_s, rcond=None)

    Solution = Solution_r

    ret_sol = np.zeros(Solution.shape)

    Sol_ind = np.arange(len(Solution))

    Nbind = len(Solution)
    Nbind_prec = Nbind + 1

    step = []

    print("Recup treshold",Recup_Threshold)

    while Nbind < Nbind_prec:
        Nbind_prec = Nbind
        # print("weight",poid)

        Condition_value = cond(Exp_matrix,Solution)


        # Condition_value = np.abs(Solution[:, 0])
        # print("cond",Condition_value)

        step += [(Solution, Condition_value, Sol_ind)]

        indices = np.argwhere(Condition_value/np.max(Condition_value) > Recup_Threshold)
        indices = indices[:, 0]
        Sol_ind = Sol_ind[indices]
        Exp_matrix = Exp_matrix[:, indices]

        Solution, residual, rank, _ = np.linalg.lstsq(Exp_matrix, Forces_vec_s, rcond=None)

        print("sol_len",Condition_value/np.max(Condition_value))
        print("indices",indices)



        Modele_fit = 0

        for i in range(len(Sol_ind)):
            Modele_fit = Modele_fit + Catalog[Sol_ind[i]] * Solution[i]
            ret_sol[Sol_ind[i]] = Solution[i]

        #print("Model fitting : ", Modele_fit[0])

        Nbind = len(Sol_ind)

    reduction = len(Solution_r)-Nbind


    return Modele_fit,ret_sol,reduction,step