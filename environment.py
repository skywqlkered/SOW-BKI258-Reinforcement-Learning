import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
    # Define action and observation spaces
        self.action_space = spaces.Discrete(2) # Example: two actions
        self.observation_space = spaces.Box(low=0, high=2, shape=(7,7), dtype=np.int8)
        self.state = None # Initialize state
    
    def reset(self, seed, options): # type: ignore
        """Reset the environment to an initial state."""
        self.state = np.array([5.0]) # Example initial state
        return self.state, {}
            
    def step(self, action):
        """Apply an action and return results."""
        reward = 1 if action == 1 else 0
        if self.state is not None:
            self.state = self.state + (action- 0.5)
            done = self.state[0] > 10 or self.state[0] < 0
            return self.state, reward, done, False, {}
        else: raise ValueError("State is not defined")
        
    def render(self):
        """Render the environment (optional)."""
        print(f"Current state: {self.state}")
    
    def close(self):
        """Clean up resources (optional)."""
        pass


# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 1 2 3 0 d 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0

# self.worm: list[tuple] = [(),(), ()]
# self.dirt: tuple = () 
