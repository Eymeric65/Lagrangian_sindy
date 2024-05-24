import time

import mujoco
import mujoco.viewer
import numpy as np

from function.Render import *

from function.Simulation import *

#Lol le monde reel est pas tres sympa...
# Les forces de dissipitation doivent etre placer de meilleure maniere que sur mon systeme ideal....
# Les forces doivent etre applique de maniere symetrique car il y a l'action reaction tout ca tout ca...

q_pos = []
q_vel = []
q_accel = []

time_l = []

q0_init = 0
q1_init = 0


m = mujoco.MjModel.from_xml_path('2D_double_pendulum_point_mass.xml')
d = mujoco.MjData(m)

d.qpos[0] = q0_init + np.pi
d.qpos[1] = q1_init - q0_init

Reg = False


def controller(model,data):

  global F_ext_func

  F = F_ext_func(data.time)

  # data.ctrl[0] = -F[0]
  # data.ctrl[1] = -F[1]
  data.qfrc_applied[0] = -F[0]-F[1] # A etudier
  data.qfrc_applied[1] = -F[1]

  time_l.append(data.time)

  q_pos.append([data.qpos[0]-np.pi, data.qpos[1]+data.qpos[0]-np.pi])
  q_vel.append([data.qvel[0],data.qvel[1]+data.qvel[0]])

  q_accel.append([data.qacc[0],data.qacc[1]+data.qacc[0]])

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()

  # Creation catalogue

  t = sp.symbols("t")

  Coord_number = 2
  Degree_function = 4
  Puissance_model = 2

  Symb = Symbol_Matrix_g(Coord_number, t)

  # Suited catalog creation

  function_catalog_1 = [
    lambda x: Symb[2, x]
  ]

  function_catalog_2 = [
    lambda x: sp.sin(Symb[1, x]),
    lambda x: sp.cos(Symb[1, x])
  ]

  Catalog_sub_1 = np.array(Catalog_gen(function_catalog_1, Coord_number, 2))
  Catalog_sub_2 = np.array(Catalog_gen(function_catalog_2, Coord_number, 2))
  Catalog_crossed = np.outer(Catalog_sub_2, Catalog_sub_1)

  Catalog = np.concatenate((Catalog_crossed.flatten(), Catalog_sub_1, Catalog_sub_2))

  Cat_len = len(Catalog)

  # --------------------------

  # Creation des forces

  # Parametre
  Surfacteur = Cat_len * 10  # La base
  periode = 3  #
  N_periode = 15  # In one periode they will be Surfacteur*N_Periode/Cat_len time tick
  Time_end = periode * Cat_len / N_periode

  print("Temps de l'experience {} et longueur du Catalogue {} ".format(Time_end, Cat_len))


  # ----------------External Forces--------------------

  M_span = [8, 1]  # Max span

  periode_shift = 0.5

  # ----------------External Forces--------------------
  np.random.seed(123)
  F_ext_func = F_gen_opt(Coord_number, M_span, Time_end, periode, periode_shift, aug=50)

  # def F_test(t):
  #
  #   t=np.array(t)
  #
  #   if(len(np.array(t).shape)==0):
  #
  #     return np.array([0.0*t, 0.1* t])
  #
  #   else:
  #     return np.array([0.0*t,0.1*t])
  #
  # print(F_test(0),F_test([0,1]))
  #
  #
  # F_ext_func = F_test
  mujoco.set_mjcb_control(controller)




  while viewer.is_running() and d.time < Time_end:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)


    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    # if time_until_next_step > 0:
    #   time.sleep(time_until_next_step)


  theta_v = np.array(q_pos)
  theta_d_v = np.array(q_vel)
  theta_dd_v = np.array(q_accel)

  print(theta_v.shape)

  time_l = np.array(time_l)

  Nb_t = len(time_l)

  Subsample = Nb_t // Surfacteur

  np.random.seed(123)

  if Reg:

    Solution, Exp_matrix, t_values_s = Execute_Regression(time_l, theta_v, t, Symb, Catalog, F_ext_func,Subsample=Subsample,q_d_v=theta_d_v,q_dd_v=theta_dd_v)

  fig, axs = plt.subplots(2, 3)

  if Reg:
    Modele_fit = Make_Solution_exp(Solution[:, 0], Catalog, Frottement=Coord_number)


  # Ideal Simulation

  theta1 = Symb[1, 0]
  theta1_d = Symb[2, 0]
  theta1_dd = Symb[3, 0]

  theta2 = Symb[1, 1]
  theta2_d = Symb[2, 1]
  theta2_dd = Symb[3, 1]

  m1, l1, m2, l2, g = sp.symbols("m1 l1 m2 l2 g")

  L1t = 1
  L2t = 1
  Lt = L1t + L2t
  Substitution = {"g": 9.81, "l1": L1t, "m1": 1, "l2": L2t, "m2": 1}

  L = (1 / 2 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 1 / 2 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
       * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(
            theta2))

  Y0 = np.array([[q0_init, 0], [q1_init, 0]])  # De la forme (k,2)

  Frotement = [-0.3, -0.5]

  # -------------------------------

  Solution_ideal = Make_Solution_vec(sp.expand_trig(L.subs(Substitution)), Catalog,
                                     Frottement=Frotement)  # ,Frottement=Frotement)
  # --------------------------


  Acc_func, _ = Lagrangian_to_Acc_func(L, Symb, t, Substitution, fluid_f=Frotement)

  Dynamics_system = Dynamics_f(Acc_func, F_ext_func)

  t_values_w, thetas_values_w = Run_RK45(Dynamics_system, Y0, Time_end, max_step=0.05)

  Frotement = [-0.3, -0.3]

  Acc_func, _ = Lagrangian_to_Acc_func(L, Symb, t, Substitution, fluid_f=Frotement)

  Dynamics_system = Dynamics_f(Acc_func, F_ext_func)

  t_values_w_2, thetas_values_w_2 = Run_RK45(Dynamics_system, Y0, Time_end, max_step=0.05)

  # ------------------------------


  if Reg:
    print("Modele fit", Modele_fit)
    print("sparsity : ", np.sum(np.where(np.abs(Solution) > 0, 1, 0)))

  axs[0,0].set_title("q0")
  axs[1,0].set_title("q1")

  axs[0,0].plot(time_l,theta_v[:,0],label="Mujoco")
  axs[0, 0].plot(t_values_w, thetas_values_w[:,0], label="Simu1")
  axs[0, 0].plot(t_values_w_2, thetas_values_w_2[:, 0], label="Simu2")

  axs[1,0].plot(time_l,theta_v[:,1],label="Mujoco")
  axs[1, 0].plot(t_values_w, thetas_values_w[:,2], label="Simu1")
  axs[1, 0].plot(t_values_w_2, thetas_values_w_2[:, 2], label="Simu2")

  axs[0,0].legend()
  axs[1, 0].legend()

  axs[0,1].set_title("q0_d")
  axs[1,1].set_title("q1_d")

  axs[0,1].plot(time_l,theta_d_v[:,0],label="Mujoco")
  axs[0, 1].plot(t_values_w, thetas_values_w[:,1], label="Simu1")
  axs[0, 1].plot(t_values_w_2, thetas_values_w_2[:, 1], label="Simu2")

  axs[1,1].plot(time_l,theta_d_v[:,1],label="Mujoco")
  axs[1, 1].plot(t_values_w, thetas_values_w[:,3], label="Simu1")
  axs[1, 1].plot(t_values_w_2, thetas_values_w_2[:, 3], label="Simu2")

  axs[0,1].legend()
  axs[1, 1].legend()

  axs[0,2].set_title("F0")
  axs[1,2].set_title("F1")

  Forces = F_ext_func(time_l)

  axs[0,2].plot(time_l,Forces[0,:],label="extended")
  axs[1,2].plot(time_l,Forces[1,:],label="extended")



  if Reg:
    axs[1, 2].set_title("Model retrieved")
    Bar_height_found = np.abs(Solution) / np.max(np.abs(Solution))
    axs[1, 2].bar(np.arange(len(Solution)), Bar_height_found[:, 0], width=0.5, label="Model Found")

  plt.show()
