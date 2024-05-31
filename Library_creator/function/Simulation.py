import numpy as np
from .Dynamics_modeling import *
from .Catalog_gen import *
from .Euler_lagrange import *
from .Optimization import *


def Execute_Regression(t_values,thetas_values,t,Symb,Catalog,F_ext_func,Noise=0,troncature=5,Subsample=1,Hard_tr=10**-3,q_d_v=[],q_dd_v=[],reg=True):

    if Subsample == 0:
        Subsample =1

    Nb_t = len(t_values)

    Coord_number = thetas_values.shape[1]

    t_values_s = t_values[troncature::Subsample]

    Forces_vec = Forces_vector(F_ext_func,t_values_s)

    Exp_matrix = Catalog_to_experience_matrix(Nb_t,Coord_number,Catalog,Symb,t,thetas_values,t_values,subsample=Subsample,Frottement=True,troncature=troncature,noise=Noise,q_d_v=q_d_v,q_dd_v=q_dd_v)

    if reg:

        Exp_norm,reduction,Variance = Normalize_exp(Exp_matrix,null_effect=True)

        coeff = Lasso_reg(Forces_vec,Exp_norm)

        Solution = Un_normalize_exp(coeff,Variance,reduction,Exp_matrix)

        Solution[np.abs(Solution)< np.max(np.abs(Solution))*Hard_tr] = 0

    else:
        Solution = None

    return Solution,Exp_matrix,t_values_s