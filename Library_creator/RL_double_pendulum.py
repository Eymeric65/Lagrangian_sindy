from function.Simulation import *
from function.ray_env_creator import *
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

#exeray.init()

L1t = 0.7
L2t = 0.7
m_1 = .5
m_2 = .5
Frotement = [-1.2,-1.0]

t = sp.symbols("t")

Coord_number = 2
Symb = Symbol_Matrix_g(Coord_number,t)

# Ideal model creation

theta1 = Symb[1,0]
theta1_d = Symb[2,0]
theta1_dd = Symb[3,0]

theta2 = Symb[1,1]
theta2_d = Symb[2,1]
theta2_dd = Symb[3,1]

m1, l1, m2, l2, g = sp.symbols("m1 l1 m2 l2 g")

h=0.1

Lt = L1t + L2t
Substitution = {"g": 9.81, "l1": L1t, "m1": m_1, "l2": L2t, "m2": m_2}

L = (1 / 2 * (m1 + m2) * l1 ** 2 * theta1_d ** 2 + 1 / 2 * m2 * l2 ** 2 * theta2_d ** 2 + m2 * l1 * l2 * theta1_d
     * theta2_d * sp.cos(theta1 - theta2) + (m1 + m2) * g * l1 * sp.cos(theta1) + m2 * g * l2 * sp.cos(
    theta2))

Acc_func,_ = Lagrangian_to_Acc_func(L, Symb, t, Substitution, fluid_f=Frotement)

Dynamics_system = Dynamics_f_extf(Acc_func)

target = np.array([0.0, 0.0, 0.0, 0.0])  # Define your target vector

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
        env_config={"coord_numb": Coord_number,"target":target,"dynamics_function_h":Acc_func,"h":h},
    )
    .framework("torch")
    .resources(num_cpus_per_worker=1, num_gpus_per_worker=1 / 16)
    # Parallelize environment rollouts.
    .env_runners(num_env_runners=3)
)
# Construct the actual (PPO) algorithm object from the config.
algo = config.build()

for i in range(30):
    results = algo.train()
    print(f"Iter: {i}; avg. return={results['env_runners']['episode_return_mean']}")

env = MyFunctionEnv({"coord_numb": Coord_number,"target":target,"dynamics_function_h":Acc_func,"h":h})

obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0

while not terminated and not truncated:
    # Compute a single action, given the current observation
    # from the environment.
    action = algo.compute_single_action(obs)
    # Apply the computed action in the environment.
    #print(action)

    obs, reward, terminated, truncated, info = env.step(action)

    print(reward)
    # Sum up rewards for reporting purposes.
    total_reward += reward
# Report results.
print(f"Played 1 episode; total-reward={total_reward}")