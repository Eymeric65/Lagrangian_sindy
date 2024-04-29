import numpy as np
from scipy.integrate import RK45
def Dynamics_f(Acc, Fext): # Transform a function Array like this [[q0_dd(q0,q0_d,f0)],...,[qk_dd(qk,qk_d,fk)]] into a function Dynamics(t,[q0,q0_d,...,qk,qk_d])
    def func(t, State):
        #State is a list [q0,q0_d,...,qk,qk_d]
        State = np.reshape(State, (-1, 2))
        State_f = np.transpose(State)

        # Input is the same as Symbol matrix
        Input = np.zeros((State_f.shape[0] + 2, State_f.shape[1]))
        Input[1:3, :] = State_f
        Input[0, :] = Fext(t)

        ret = np.zeros(State.shape)

        ret[:, 0] = State[:, 1]
        ret[:, 1] = Acc(Input)[:, 0]
        return np.reshape(ret, (-1,))

    return func

def Run_RK45(dynamics, Y0, Time_end,max_step=0.05):  #Run a RK45 integration on our model Y0 is still under the form (k,2)

    Y0_f = np.reshape(Y0, (-1,))  # de la forme(2*k,)
    Model = RK45(dynamics, 0, Y0_f, Time_end, max_step, 0.001, np.e ** -6)

    # collect data
    t_v = []
    q_v = []
    for i in range(8000):
        # get solution step state
        Model.step()
        t_v.append(Model.t)
        q_v.append(Model.y)
        # break loop after modeling is finished
        if Model.status == 'finished':
            print("End step : ", i)
            break

    q_v = np.array(q_v)  # Output as (k,len(t_values)) with line like this : q0,q0_d,...,qk,qk_d
    t_v = np.array(t_v)
    return t_v, q_v
