import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any

class MouseEnv(gym.Env):
    def __init__(self):
        "Defines the action and observation spaces"
        super(MouseEnv, self).__init__()

        # Action space: 0:Up, 1:Right, 2:Down, 3:Left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: (4*3) mouse positions * ((4*3)-1) cheese positions + 1 win state
        self.cols = 4
        self.rows = 3
        self.observation_space = spaces.Discrete((self.cols * self.rows) * (self.cols * self.rows - 1) + 1)

        # Initialize state
        self.mouse_pos = None
        self.cheese_pos = None
        self.won = False

        # Define rewards and punishment
        self.step_punishment = -1.0
        self.impossible_action_punishment = -2.0
        self.lose_punishment = -50
        self.win_reward = 100

    def _get_obs(self):
        """
        Converts the current board state into a single integer.
        Total states = (rows * cols mouse positions * (rows * cols - 1) cheese positions) + 1 goal state.
        Returned values are in the range [0, observation_space.n - 1].
        """
        # If the cheese is in the hole, succesful terminal state (highest index)
        if self.won:
            return (self.cols * self.rows) * (self.cols * self.rows - 1)

        # Convert 2D coordinates to 1D indices
        m_idx = self.mouse_pos[0] * self.cols + self.mouse_pos[1] # type: ignore
        c_idx = self.cheese_pos[0] * self.cols + self.cheese_pos[1] # type: ignore

        # Lower cheese index by one if it is bigger than the mouse index, 
        # to remove the state in which they are on top of each other.
        if c_idx > m_idx:
            c_idx -= 1
            
        # Return the final index
        return (m_idx * (self.cols * self.rows - 1)) + c_idx

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to an initial state.
        Randomly places the cheese in valid cells (rows < n-1, columns > 0) and the mouse in any cell that doesn't overlap with the cheese.
        
        Args:
            seed: Seed for random number generator (optional).
            options: Additional options for reset (optional).
        
        Returns:
            obs: Initial observation as an integer.
            info: Empty dictionary.
        """
        super().reset(seed=seed, options=options)

        # Define valid cheese cells (Row < n-1, Col > 0) (not game over)
        valid_cheese_cells = [
            [r, c] for r in range(self.rows-1) for c in range(1, self.cols)
        ]

        # Pick random cheese position
        self.cheese_pos = valid_cheese_cells[self.np_random.choice(len(valid_cheese_cells))]
        
        # Pick random mouse position
        while True:
            m_row = self.np_random.integers(0, self.rows)
            m_col = self.np_random.integers(0, self.cols)
            self.mouse_pos = [m_row, m_col]
            
            # If this position is not occupied by cheese, continue
            if self.mouse_pos != self.cheese_pos:
                break
        
        # Start as not game over
        self.terminated = False

        # Return obs, info
        return self._get_obs(), {}
            
    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Apply an action to the environment.
        Args:
            action: Int, specifies the direction the mouse moves in.
        
        Returns:
            Any: idk
            Float: The reward the mouse got for the action
            Bool: True if the mouse got the cheese in the hole, false otherwise
            Bool: Truncated is always false since the mouse cannot go out-of-bounce and there is no timelimit.
            dict: empty dictionary, there is no need for auxiliary diagnostic information yet.


        0: Up, 1: Right, 2: Down, 3: Left
        """
        # Define movement: {action: (row_change, col_change)}
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = moves[action]

        # Calculate where the mouse wants to go
        new_m = [self.mouse_pos[0] + dr, self.mouse_pos[1] + dc] # type: ignore

        # Give a small punishment each step to incentivize shortest path
        reward = self.step_punishment
        done = False


        # Mouse hit a wall -> impossible, do nothing
        if not (0 <= new_m[0] < self.rows and 0 <= new_m[1] < self.cols):
            reward = self.impossible_action_punishment
        
        # If the mouse tries to push cheese
        elif new_m == self.cheese_pos:

            # Calculate where the cheese would go
            new_c = [self.cheese_pos[0] + dr, self.cheese_pos[1] + dc] # type: ignore
            
            # If the cheese gets pushed to the win position
            if new_c == [-1, self.cols - 1]:
                reward = self.win_reward
                done = True

                self.mouse_pos = new_m
                self.cheese_pos = new_c
                self.won = True

            # If cheese stays on board
            elif (0 <= new_c[0] < self.rows and 0 <= new_c[1] < self.cols):
                self.cheese_pos = new_c
                self.mouse_pos = new_m
                
                # If cheese gets pushed to a fatal state (first column or last row)
                if self.cheese_pos[1] == 0 or self.cheese_pos[0] == self.rows - 1:
                    reward = self.lose_punishment
                    done = True
                
            # Cheese hit a wall -> impossible, do nothing
            else:
                reward = self.impossible_action_punishment
        
        # Normal Mouse Move (no push)
        else:
            self.mouse_pos = new_m

        return self._get_obs(), reward, done, False, {}

    def render(self):
        grid = np.full((self.rows, self.cols), "0", dtype=str)
        if not self.won:
            grid[self.cheese_pos[0], self.cheese_pos[1]] = "c" # type: ignore
        grid[self.mouse_pos[0], self.mouse_pos[1]] = "m" # type: ignore
        print("      ╭───╮")
        print("╭─────╯ h │")
        for row in grid:
            print("│ " + " ".join(row) + " │")
        print("╰─────────╯")

    def close(self):
        """Cleanup (not required for this simple environment)."""
        pass