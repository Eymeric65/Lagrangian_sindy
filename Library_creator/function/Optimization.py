import numpy as np

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
# def Condition_val(Exp,solution):
#
#     return np.abs((np.linalg.norm(Exp, axis=0)**2 * solution[:, 0]))

def Condition_val(Exp,solution):

    return np.abs((np.var(Exp, axis=0) * solution[:, 0]))

def Hard_treshold_sparse_regression(Exp_matrix,Forces_vec_s,Catalog,cond=Condition_val,Recup_Threshold = 0.03):

    Solution_r, residual, rank, _ = np.linalg.lstsq(Exp_matrix, Forces_vec_s, rcond=None)

    Solution = Solution_r

    ret_sol = np.zeros(Solution.shape)

    Sol_ind = np.arange(len(Solution))

    Nbind = len(Solution)
    Nbind_prec = Nbind + 1

    step = []

    #print("Recup treshold",Recup_Threshold)

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

        #print("sol_len",Condition_value/np.max(Condition_value))
        #print("indices",indices)

        Nbind = len(Sol_ind)

    Modele_fit = 0

    for i in range(len(Sol_ind)):
        Modele_fit = Modele_fit + Catalog[Sol_ind[i]] * Solution[i]
        ret_sol[Sol_ind[i]] = Solution[i]

        #print("Model fitting : ", Modele_fit[0])

    reduction = len(Solution_r)-Nbind

    return Modele_fit,ret_sol,reduction,step

def Lasso_reg(F_vec,Exp_norm,m_iter=10**6,tol=10**-6,eps=5*10**-6):

    Y = F_vec[:, 0]

    model = LassoCV(cv=5, random_state=0, max_iter=m_iter,eps=eps,tol=tol)

    # Fit model
    model.fit(Exp_norm, Y)

    alpha = model.alpha_

    #print("Lasso alpha : ", alpha)

    # Set best alpha
    lasso_best = Lasso(alpha=model.alpha_,max_iter=m_iter,tol=tol)
    lasso_best.fit(Exp_norm, Y)

    coeff = lasso_best.coef_

    return coeff

def Normalize_exp(Exp_matrix,null_effect=False):

    Variance = np.var(Exp_matrix, axis=0) * int(not null_effect) + int(null_effect)
    Mean = np.mean(Exp_matrix, axis=0) * int(not null_effect)

    reduction = np.argwhere(Variance != 0)

    #print("Reduction shape : ",Variance.shape,reduction.shape)

    Exp_matrix_r = Exp_matrix[:, reduction[:, 0]]
    Exp_norm = (Exp_matrix_r - Mean[reduction[:, 0]]) / Variance[reduction[:, 0]]

    return Exp_norm,reduction,Variance

def Un_normalize_exp(coeff,Variance,reduction,Exp_mat):

    Solution_r = coeff[:] / Variance[reduction[:, 0]]

    Frottement_coeff = -Solution_r[-1]

    Solution = np.zeros((Exp_mat.shape[1], 1)) # No generalise

    Solution[reduction[:, 0], 0] = Solution_r

    return Solution


