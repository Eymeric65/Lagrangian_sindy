import mujoco.viewer
import numpy as np

def Mujoco_simulation(model,Y0,F_ext_func,Time_end,Coor):


    q_phase = []

    time_l = []

    m = mujoco.MjModel.from_xml_path(model)
    d = mujoco.MjData(m)

    for i in range(Coor):

        d.qpos[i] = Y0[i,0] - np.sum(Y0[0:i,0]) # Local angle

        print("init pos ",i,d.qpos[i])

    def controller(model, data):

        F = F_ext_func(data.time)

        for i in range(Coor):
            data.qfrc_applied[i] = -F[i]  # -F[1] # A etudier

        time_l.append(data.time)

        q_phase.append([item for i in range(Coor) for item in [np.sum(data.qpos[0:i+1]),np.sum(data.qvel[0:i+1]),np.sum(data.qacc[0:i + 1])]]) # Local angle


    with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
        mujoco.set_mjcb_control(controller)

        while viewer.is_running() and d.time < Time_end:

            mujoco.mj_step(m, d)
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

    mujoco.set_mjcb_control(None)

    time_l = np.array(time_l)

    q_phase = np.array(q_phase)





    return time_l,q_phase