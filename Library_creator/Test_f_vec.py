import numpy as np

from function.Dynamics_modeling import *
import matplotlib.pyplot as plt

T_end = 100

#F_ext = F_gen_v(1000,0.0,T_end,1,1) #,aug=2,method="v")

F_ext = F_gen_opt(2,200,T_end,1,0.2,aug=50)

t = np.arange(0,T_end,0.01)

print(F_ext(0))

res = F_ext(t)



print((np.max(res)-np.min(res))/2)
print(np.std(res),np.var(res))
plt.plot(t,res[0,:])
plt.show()

