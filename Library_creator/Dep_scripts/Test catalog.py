from function.Catalog_gen import *

t = sp.symbols("t")

Coord_number = 2
Symb = Symbol_Matrix_g(Coord_number,t)

Degree_function = 4

function_catalog = [
    lambda x : Symb[1,x],
    lambda x : Symb[2,x],
    lambda x : sp.sin(Symb[1,x]),
    lambda x : sp.cos(Symb[1,x])
]

Puissance = 2

Catalog = Catalog_gen(function_catalog,Coord_number,Degree_function,puissance=Puissance)

print(Concat_Func_var(function_catalog,Coord_number))

print(len(Catalog))
#
print(Catalog)

for c in range(len(Catalog)):

    for j in range(c+1,len(Catalog)):

        if Catalog[c]==Catalog[j]:

            print(j,c)
            print(Catalog[j],Catalog[c])