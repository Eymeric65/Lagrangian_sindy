import numpy as np

from function.Euler_lagrange import *
from function.Catalog_gen import *
import time

# Creation catalogue

#np.random.seed(1032)

t = sp.symbols("t")

Coord_number = 2
Symb = Symbol_Matrix_g(Coord_number,t)

Degree_function = 4

Puissance_model= 2

# ------ Suited catalog creation ------

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

Cat_len = len(Catalog)

print(Cat_len)

Test = 2

# Test 1 is to know wether or not there is q_k_dd**2 in Lagrangian representation of our fonction

if Test == 1:

    for k in (0,1):

        for j in Catalog:

            exp = Euler_lagranged(j,Symb, t, k)

            test1 = str(Symb[3,0]) in str(exp)

            test2 = str(Symb[3,1]) in str(exp)

            if(test1 or test2):

                print(str(exp))

#Test 2 is to calculate the time to inverse a random Lagrangian expression

elif Test == 2 :

    Validation = True

    for _ in range(100) :

        Number_Term = 3

        L = np.sum( np.random.choice(Catalog,Number_Term,replace=False) )

        Start_T = time.time()

        Modele,Valid = Lagrangian_to_Acc_func(L, Symb, t, {}, fluid_f=[], Verbose=False)

        if(Valid):

            #print(L)
            print("Exec Time of solve ",time.time()-Start_T)

        #Validation
        if Validation and Valid :

            Start_T = time.time()
            Modele_V, Valid_V = Lagrangian_to_Acc_func(L, Symb, t, {}, fluid_f=[], Verbose=False,Clever_Solve=False)
            print("Exec Time of solve old ", time.time() - Start_T)

            Input = np.random.random(Symb.shape)

            Start_T = time.time()
            Inf1 =Modele(Input)
            print("Exec Time Inference new ", time.time() - Start_T)


            Start_T = time.time()
            Inf2 =Modele_V(Input)
            print("Exec Time Inference old ", time.time() - Start_T)

            print("Error :",np.linalg.norm(Inf1-Inf2)/np.linalg.norm(Inf2))

            print(Inf1,Inf2)

            break

#--------------------------