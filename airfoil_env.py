import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import spaces


class AirfoilControlEnv(gym.Env):
    def __init__(self, model_path):
        super(AirfoilControlEnv, self).__init__()
        
        # Load the trained DNN model
        self.model = tf.keras.models.load_model(model_path)
        
        # State Space: Discrete Mach numbers and discrete target lift coefficients
        self.mach_values = np.arange(0.17, .31, 0.01)
        self.target_cl_values = np.arange(-0.5, 2.0, 0.1)

        # Action Space: Discrete angles of attack
        self.angle_of_attack_values = np.arange(-4.5, 15.9, 0.25)
        
        # Observation and Action Spaces
        self.num_mach_states = len(self.mach_values)
        self.num_cl_states = len(self.target_cl_values)
        self.observation_space = spaces.MultiDiscrete([self.num_mach_states, self.num_cl_states,])
        self.action_space = spaces.Discrete(len(self.angle_of_attack_values))
        
        # Initial state
        self.state = None
    
    def reset(self):
        """Reset environment state."""
        # Randomly select indices for Mach and Target Lift Coefficient
        mach_index = np.random.choice(self.num_mach_states)
        target_cl_index = np.random.choice(self.num_cl_states)
        
        self.state = np.array([
            mach_index, 
            target_cl_index
        ], dtype=np.int32)
        
        return self.state
    
    def step(self, state: tuple, action: int):
        """Execute an action."""
        angle_of_attack = self.angle_of_attack_values[action]
        
        # Get actual Mach and Target Lift Coefficient values
        mach = self.mach_values[state[0]]
        target_cl = self.target_cl_values[state[1]]
        
        # Predict Lift (CL) using the DNN model
        predicted_cl = self.model.predict(np.array([[mach, angle_of_attack]]), verbose=0)[0][0]
        
        # Reward based on proximity to target lift coefficient
        error = abs(predicted_cl - target_cl)
        reward = -error  # Negative error as reward
        
        '''# Done condition
        done = error < 0.01  # Episode ends when error is minimal
        '''
        done = None
        
        # State remains unchanged
        self.state = np.array([
            state[0], 
            state[1]
        ], dtype=np.int32)
        
        return self.state, reward, done, {}
    
    def render(self, mode='human'):
        mach_index, target_cl_index = self.state
        mach = self.mach_values[mach_index]
        target_cl = self.target_cl_values[target_cl_index]
        print(f"State: Mach={mach:.2f}, Target CL={target_cl:.2f}")