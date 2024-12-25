import numpy as np
from airfoil_env import AirfoilControlEnv
import tqdm

env = AirfoilControlEnv(model_path="best_weights.keras")
policy_probs = np.full((env.observation_space.nvec[0], env.observation_space.nvec[1], env.action_space.n), 0.25)

def policy(state: tuple):
    return policy_probs[state]

state_values = np.zeros((env.observation_space.nvec[0], env.observation_space.nvec[1]))

def value_iteration(policy_probs, state_values):
    num_mach_states = env.observation_space.nvec[0]
    num_cl_states = env.observation_space.nvec[1]
    num_actions = env.action_space.n
    
    for row in range(num_mach_states):
        for col in range(num_cl_states):
            # old_value = state_values[(row, col)]
            action_probs = None
            max_qsa = float('-inf')

            for action in range(num_actions):
                state, reward, _, _ = env.step((row, col), action)
                qsa = reward
                if qsa > max_qsa:
                    max_qsa = qsa
                    action_probs = np.zeros(num_actions)
                    action_probs[action] = 1.
            print('One state done')
            state_values[(row, col)] = max_qsa
            policy_probs[(row, col)] = action_probs
    return policy_probs, state_values

# Initialize environment
env = AirfoilControlEnv(model_path="best_weights.keras")

# Perform Value Iteration
optimal_policy, value_function = value_iteration(policy_probs, state_values)

# Save the optimal policy
np.save("optimal_policy.npy", optimal_policy)