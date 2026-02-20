import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any

class MouseEnv(gym.Env):
    cols = 4
    rows = 3

    # Define movement: {action: (row_change, col_change)}
    moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    # Define rewards and punishment
    step_punishment = -1
    impossible_action_punishment = -2
    lose_punishment = -50
    win_reward = 100

    def __init__(self):
        "Defines the action and observation spaces"
        super(MouseEnv, self).__init__()

        # Action space: 0:Up, 1:Right, 2:Down, 3:Left
        self.num_of_actions = 4
        self.action_space = spaces.Discrete(self.num_of_actions)
        
        # Observation space: (4*3) mouse positions * ((4*3)-1) cheese positions + 1 win state
        self.num_of_states = (self.cols * self.rows) * (self.cols * self.rows - 1) + 1
        self.observation_space = spaces.Discrete(self.num_of_states)
        
        # Initialize state
        self.mouse_pos = None
        self.cheese_pos = None
        self.won = False

    def _get_obs(self):
        """
        Converts the current board state into a single integer.
        Total states = (rows * cols mouse positions * (rows * cols - 1) cheese positions) + 1 goal state.
        Returned values are in the range [0, observation_space.n - 1].
        """
        # If the cheese is in the hole, successful terminal state (highest index)
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
    
    @classmethod
    def get_state_from_obs(cls, obs: int) -> tuple[list[int] | None, list[int] | None, bool]:
        """
        Converts an observation integer back into the corresponding mouse and cheese positions, and win state.
        This is the inverse of _get_obs.
        
        Args:
            obs (int): The observation integer to convert.
        
        Returns:
            mouse_pos (list[int] | None): The [row, col] position of the mouse, or None if in win state.
            cheese_pos (list[int] | None): The [row, col] position of the cheese, or None if in win state.
            won (bool): True if the observation corresponds to the win state, False otherwise.

        """
        if obs == (cls.cols * cls.rows) * (cls.cols * cls.rows - 1):
            return None, None, True
        
        m_idx = obs // (cls.cols * cls.rows - 1)
        c_idx = obs % (cls.cols * cls.rows - 1)

        if c_idx >= m_idx:
            c_idx += 1

        m_row, m_col = divmod(m_idx, cls.cols)
        c_row, c_col = divmod(c_idx, cls.cols)

        return [m_row, m_col], [c_row, c_col], False

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

    @classmethod
    def get_reward(cls, state: int, action: int) -> float:
        """
        Calculate the reward for the current state and action.
        Implements the same reward structure as in step, but without changing the state.
        """        
        # Convert state back to mouse and cheese positions
        (m_row, m_col), (c_row, c_col), won = cls.get_state_from_obs(state)

        # If already in win state, no reward for any action
        if won:
            return 0
        
        # Define movement: {action: (row_change, col_change)}
        dr, dc = cls.moves[action]

        # Calculate where the mouse wants to go
        new_m = [m_row + dr, m_col + dc]
        # Give a small punishment each step to incentivize shortest path
        reward = cls.step_punishment
        # Mouse hit a wall -> impossible, do nothing
        if not (0 <= new_m[0] < cls.rows and 0 <= new_m[1] < cls.cols):
            reward = cls.impossible_action_punishment
        # If the mouse tries to push cheese
        elif new_m == [c_row, c_col]:
            # Calculate where the cheese would go
            new_c = [c_row + dr, c_col + dc]
            # If the cheese gets pushed to the win position
            if new_c == [-1, cls.cols - 1]:
                reward = cls.win_reward
            # If cheese stays on board
            elif (0 <= new_c[0] < cls.rows and 0 <= new_c[1] < cls.cols):
                # If cheese gets pushed to a fatal state (first column or last row)
                if new_c[1] == 0 or new_c[0] == cls.rows - 1:
                    reward = cls.lose_punishment
            # Cheese hit a wall -> impossible, do nothing
            else:
                reward = cls.impossible_action_punishment
        
        return reward

    @classmethod
    def get_next_state(cls, obs: int, action: int) -> int:
        """
        Calculate the next step for the current state and action.
        Implements the same step structure as in step, but without changing the state.
        """

        mouse_pos, cheese_pos, won = MouseEnv.get_state_from_obs(obs)
        dr, dc = cls.moves[action]

        # Calculate where the mouse wants to go
        new_m = [mouse_pos[0] + dr, mouse_pos[1] + dc]

        # Mouse hit a wall -> impossible, do nothing
        if not (0 <= new_m[0] < cls.rows and 0 <= new_m[1] < cls.cols):
            return cls.get_obs(mouse_pos, cheese_pos, won)

        # If the mouse tries to push cheese
        elif new_m == cheese_pos:

            # Calculate where the cheese would go
            new_c = [cheese_pos[0] + dr, cheese_pos[1] + dc]

            # If the cheese gets pushed to the win position
            if new_c == [-1, cls.cols - 1]:
                return cls.get_obs(new_m, new_c, True)

            # If cheese stays on board
            elif 0 <= new_c[0] < cls.rows and 0 <= new_c[1] < cls.cols:
                return cls.get_obs(new_m, new_c, won)

            # Cheese hit a wall -> impossible, do nothing
            else:
                return cls.get_obs(mouse_pos, cheese_pos, won)

        # Normal Mouse Move (no push)
        else:
            return cls.get_obs(new_m, cheese_pos, won)

    @classmethod
    def get_obs(cls, mouse_pos: list[int], cheese_pos: list[int], won: bool) -> int:
        """
        Converts the current board state into a single integer.
        Total states = (rows * cols mouse positions * (rows * cols - 1) cheese positions) + 1 goal state.
        Returned values are in the range [0, observation_space.n - 1].
        """
        # If the cheese is in the hole, successful terminal state (highest index)
        if won:
            return (cls.cols * cls.rows) * (cls.cols * cls.rows - 1)

        # Convert 2D coordinates to 1D indices
        m_idx = mouse_pos[0] * cls.cols + mouse_pos[1]  # type: ignore
        c_idx = cheese_pos[0] * cls.cols + cheese_pos[1]  # type: ignore

        # Lower cheese index by one if it is bigger than the mouse index,
        # to remove the state in which they are on top of each other.
        if c_idx > m_idx:
            c_idx -= 1

        # Return the final index
        return (m_idx * (cls.cols * cls.rows - 1)) + c_idx

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Apply an action to the environment.
        Args:
            action: Int, specifies the direction the mouse moves in.
        
        Returns:
            Int: The observation of the new state using self._get_obs()
            Float: The reward the mouse got for the action
            Bool: True if the mouse got the cheese in the hole, false otherwise
            Bool: Truncated is always false since the mouse cannot go out-of-bounds and there is no time-limit.
            Dict: empty dictionary, there is no need for auxiliary diagnostic information yet.


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

if __name__ == "__main__":
    print(MouseEnv.get_reward(0, 0))