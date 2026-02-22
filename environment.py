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
        """Return the observation for the current state."""
        return self.get_obs(self.mouse_pos, self.cheese_pos, self.won) # type: ignore
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment to an initial state.
        Randomly places the cheese in valid cells (rows < n-1, columns > 0) and the mouse in any cell that doesn't overlap with the cheese.
        
        Args:
            seed: Seed for random number generator (optional).
            options: Additional options for reset (optional).
        
        Returns:
        - obs: Initial observation as an integer.
        - info: Empty dictionary.
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
        - Int: The observation of the new state using self._get_obs()
        - Float: The reward the mouse got for the action
        - Bool: True if the mouse got the cheese in the hole, false otherwise
        - Bool: Truncated is always false since the mouse cannot go out-of-bounds and there is no time-limit.
        - Dict: empty dictionary, there is no need for auxiliary diagnostic information yet.


        0: Up, 1: Right, 2: Down, 3: Left
        """
        next_mouse, next_cheese, next_won, reward, done = self._simulate_transition(self.mouse_pos, self.cheese_pos, self.won, action)

        self.mouse_pos = next_mouse
        self.cheese_pos = next_cheese
        self.won = next_won

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

    @classmethod
    def get_state_from_obs(cls, obs: int) -> tuple[list[int] | None, list[int] | None, bool]:
        """
        Converts an observation integer back into the corresponding mouse and cheese positions, and win state.
        This is the inverse of get_obs.
        
        Args:
            obs (int): The observation integer to convert.
        
        Returns:
        - mouse_pos: (list[int] | None): The [row, col] position of the mouse, or None if in win state.
        - cheese_pos: (list[int] | None): The [row, col] position of the cheese, or None if in win state.
        - won (bool): True if the observation corresponds to the win state, False otherwise.
        
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

    @classmethod
    def is_terminal_obs(cls, obs: int) -> bool:
        """Check whether an observation is terminal (win or lose).

        A state is terminal when either:
        - the cheese is in the hole (win state), or
        - the cheese is in the first column / last row (lose state).

        Args:
            obs (int): Observation index.

        Returns:
            bool: True if the observation is terminal, else False.
        """
        mouse_pos, cheese_pos, won = cls.get_state_from_obs(obs)

        # Win state
        if won or mouse_pos is None or cheese_pos is None:
            return True
        
        # Lose state
        return cheese_pos[1] == 0 or cheese_pos[0] == cls.rows - 1

    @classmethod
    def _is_terminal_state(cls, cheese_pos: list[Any] | None, won: bool) -> bool:
        """Check terminal condition from decoded state components."""
        if won or cheese_pos is None:
            return True
        return cheese_pos[1] == 0 or cheese_pos[0] == cls.rows - 1

    @classmethod
    def _simulate_transition(cls, mouse_pos: list[Any] | None, cheese_pos: list[Any] | None, won: bool, action: int,) -> tuple[list[Any] | None, list[Any] | None, bool, float, bool]:
        """Simulate one transition from decoded state values.

        Returns next mouse position, next cheese position, next won flag, reward, and terminal flag.
        """
        # Terminal state
        if cls._is_terminal_state(cheese_pos, won):
            return mouse_pos, cheese_pos, won, 0, True

        if mouse_pos is None or cheese_pos is None:
            raise ValueError("Missing positions for non-terminal transition simulation.")

        # Determine the new positions
        dr, dc = cls.moves[action]
        new_m = [mouse_pos[0] + dr, mouse_pos[1] + dc]

        # Initialize return values
        next_mouse = mouse_pos
        next_cheese = cheese_pos
        next_won = won
        reward = cls.step_punishment
        done = False

        # Mouse hit a wall -> impossible, do nothing
        if not (0 <= new_m[0] < cls.rows and 0 <= new_m[1] < cls.cols):
            reward = cls.impossible_action_punishment

        # If the mouse tries to push cheese
        elif new_m == cheese_pos:
            new_c = [cheese_pos[0] + dr, cheese_pos[1] + dc]

            # If the cheese gets pushed to the win position
            if new_c == [-1, cls.cols - 1]:
                next_mouse = new_m
                next_cheese = new_c
                next_won = True
                reward = cls.win_reward
                done = True

            # If cheese stays on board
            elif 0 <= new_c[0] < cls.rows and 0 <= new_c[1] < cls.cols:
                next_mouse = new_m
                next_cheese = new_c

                # If cheese gets pushed to a fatal state (first column or last row)
                if new_c[1] == 0 or new_c[0] == cls.rows - 1:
                    reward = cls.lose_punishment
                    done = True
                    
            # Cheese hit a wall -> impossible, do nothing
            else:
                reward = cls.impossible_action_punishment
        # Normal mouse move (no push)
        else:
            next_mouse = new_m

        return next_mouse, next_cheese, next_won, reward, done

    @classmethod
    def get_reward(cls, state: int, action: int) -> float:
        """
        Calculate the reward for the current state and action.
        Implements the same reward structure as in step, but without changing the state.

        Args:
            state (int): the observation of the current state.
            action (int): the action to take.

        Returns:
            reward (float): the reward for the transition.
        """        
        mouse_pos, cheese_pos, won = cls.get_state_from_obs(state)
        _, _, _, reward, _ = cls._simulate_transition(mouse_pos, cheese_pos, won, action)
        return reward

    @classmethod
    def get_next_state(cls, obs: int, action: int) -> int:
        """
        Calculate the next step for the current state and action.
        Implements the same step structure as in step, but without changing the state.

        Args:
            obs (int): The observation of the current state.
            action (int): the action to take.

        Returns:
            new_obs (int): The observation of the new state.
        """
        mouse_pos, cheese_pos, won = cls.get_state_from_obs(obs)
        next_mouse, next_cheese, next_won, _, _ = cls._simulate_transition(mouse_pos, cheese_pos, won, action)

        # Win state is represented by a dedicated terminal observation.
        if next_won:
            return cls.get_obs([0, 0], [0, 0], True)

        if next_mouse is None or next_cheese is None:
            raise ValueError("Missing positions when encoding non-win next state.")

        return cls.get_obs(next_mouse, next_cheese, next_won)

    @classmethod
    def get_obs(cls, mouse_pos: list[int], cheese_pos: list[int], won: bool) -> int:
        """
        Converts the current board state into a single integer.
        Total states = (rows * cols mouse positions * (rows * cols - 1) cheese positions) + 1 goal state.
        Returned values are in the range [0, observation_space.n - 1].

        Args:
            mouse_pos ([int, int]): A list containing the row and column index of the mouse.
            cheese_pos ([int, int]): A list containing the row and column index of the cheese.
            won (bool): True if the observation corresponds to the win state, False otherwise.

        Returns:
            obs (int): The observation of the given state.
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