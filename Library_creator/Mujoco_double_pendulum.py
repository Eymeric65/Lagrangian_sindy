import time

import mujoco
import mujoco.viewer

from function.Render import *

from function.Simulation import *


qact1 = []
qact2 = []
time_l = []

q0_init = -np.pi
q1_init = 0


m = mujoco.MjModel.from_xml_path('2D_double_pendulum.xml')
d = mujoco.MjData(m)

d.qpos[0] = q0_init
d.qpos[1] = q1_init



def controller(model,data):

  global F_ext_func

  F = F_ext_func(data.time)

  data.ctrl[0] = F[0]
  data.ctrl[1] = F[1]

  time_l.append(data.time)
  qact1.append(data.qpos[0])
  qact2.append(data.qpos[1])

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
  Surfacteur = Cat_len * 25  # La base
  periode = 0.8  #
  N_periode = 10  # In one periode they will be Surfacteur*N_Periode/Cat_len time tick
  Time_end = periode * Cat_len / N_periode

  print("Temps de l'experience {} et longueur du Catalogue {} ".format(Time_end, Cat_len))

  M_span = 20  # Max span
  periode_shift = 0.1

  # ----------------External Forces--------------------

  F_ext_func = F_gen_c(M_span, periode_shift, Time_end, periode, Coord_number)

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
  qact2 = np.array(qact2)
  qact1 = np.array(qact1)

  theta_v = np.concatenate((qact1,qact2))
  theta_v = np.reshape(theta_v,(2,-1)).T

  time_l = np.array(time_l)

  Nb_t = len(time_l)

  Subsample = Nb_t // Surfacteur

  Solution, Exp_matrix, t_values_s = Execute_Regression(time_l, theta_v, t, Symb, Catalog, F_ext_func,Subsample=Subsample)

  fig, axs = plt.subplots(2, 2)

  Modele_fit = Make_Solution_exp(Solution[:, 0], Catalog, Frottement=Coord_number)

  print("Modele fit", Modele_fit)

  axs[0,0].set_title("q0")
  axs[1,0].set_title("q1")

  axs[0,0].plot(time_l,theta_v[:,0],label="extended")
  axs[1,0].plot(time_l,theta_v[:,1],label="extended")

  axs[0,1].set_title("F0")
  axs[1,1].set_title("F1")

  Forces = F_ext_func(time_l)

  axs[0,1].plot(time_l,Forces[0,:],label="extended")
  axs[1,1].plot(time_l,Forces[1,:],label="extended")

  axs[1, 1].set_title("Model retrieved")

  Bar_height_found = np.abs(Solution) / np.max(np.abs(Solution))
  axs[1, 1].bar(np.arange(len(Solution)), Bar_height_found[:, 0], width=0.5, label="Model Found")

  plt.show()
