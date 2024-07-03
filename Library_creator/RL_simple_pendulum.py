from scipy import interpolate
from function.Dynamics_modeling import *
from function.Euler_lagrange import *
from function.Render import *
from function.Catalog_gen import *

from function.ray_env_creator import *

from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

# Single pendulum exclusive.....

# Initialisation du modèle théorique

t = sp.symbols("t")

CoordNumb = 1

Symb = Symbol_Matrix_g(CoordNumb,t)

theta = Symb[1,0]
theta_d = Symb[2,0]
theta_dd = Symb[3,0]

m, l, g = sp.symbols("m l g")

L = 0.2
Substitution = {"g": 9.81, "l": L, "m": 0.1}

Time_end = 14

#----------------External Forces--------------------

F_ext_time = np.array([0, 2, 4, 6, 8, Time_end])
F_ext_Value = np.array([[0, 1, -1, 1, 1, -1]]) * 0.0  # De la forme (k,...)

F_ext_func = interpolate.CubicSpline(F_ext_time, F_ext_Value, axis=1)
# ---------------------------

Y0 = np.array([[2, 0]])  # De la forme (k,2)

L_System = m*l**2/2*theta_d**2+sp.cos(theta)*l*m*g

Acc_func,_ = Lagrangian_to_Acc_func(L_System, Symb, t, Substitution, fluid_f=[-0.02])

Dynamics_system = Dynamics_f_extf(Acc_func)

EnvConfig = {
    "coord_numb": CoordNumb,
    "target":np.array([np.pi,0]),
    "dynamics_function_h":Dynamics_system,
    "h":0.05
}

ray.init(
  num_cpus=16,
  num_gpus=1,
  include_dashboard=False,
  ignore_reinit_error=True,
  log_to_driver=False,
)


# Create an RLlib Algorithm instance from a PPOConfig object.
config = (
    PPOConfig().environment(
        # Env class to use (here: our gym.Env sub-class from above).
        env=MyFunctionEnv,
        env_config=EnvConfig,
    )
    .framework("torch")
    .resources(num_cpus_per_worker=1, num_gpus_per_worker=1 / 16)
    # Parallelize environment rollouts.
    .env_runners(num_env_runners=10)
)
# Construct the actual (PPO) algorithm object from the config.
algo = config.build()

for i in range(10):
    results = algo.train()
    print(f"Iter: {i}; avg. return={results['env_runners']['episode_return_mean']}")

#Confirmation experience

stop = False
Environment = MyFunctionEnv(EnvConfig)

while not stop:

    action = algo.compute_single_action(Environment.state)

    state, reward, stop, truncated,_ = Environment.step(action)

    print(state, reward, stop, truncated)

    Environment.render()
