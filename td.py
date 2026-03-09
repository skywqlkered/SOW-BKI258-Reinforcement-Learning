from environment import MouseEnv
import numpy as np


def epsilon_greedy(state: int, epsilon: float, Qtable: np.typing.NDArray) -> int:
    """Returns an action based on the epsilon value

    Args:
        state (int): number of current stateF
        epsilon (float): the epsilon value

    Returns:
        action (int) the next action to take

    """
    if np.random.random() < epsilon:
        return np.random.randint(low=0, high=MouseEnv.num_of_actions)
    else:
        return int(np.argmax(Qtable[state]))

def greedy(state: int, Qtable: np.typing.NDArray) -> int:
    """Returns a greedy action

    Args:
        state (int): number of current state
        Qtable (np.typing.NDArray): the Qtable

    Returns:
        action (int): the next action to take
    """
    return np.argmax(Qtable[state])

def run_qlearning(
    num_of_episodes: int,
    learning_rate: float,
    discount_rate: float,
    epsilon: float) -> np.typing.NDArray:
    """Runs the Q-learning algorithm and returns Q-values

        Args:
            num_of_episodes (int): amount of episode runs
            learning_rate (float): the learning rate
            discount_rate (float): the discount factor
            epsilon (float): the epsilon greedy value

        Returns:
            Q-values (np.typing.NDArray): A table of Q values of state and action
        """

    Q = np.zeros((MouseEnv.num_of_states, MouseEnv.num_of_actions))
    for _ in range(num_of_episodes):

        state = np.random.randint(0, MouseEnv.num_of_states)
        action = epsilon_greedy(state=state, epsilon=epsilon, Qtable=Q)

        while not MouseEnv.is_terminal_obs(state):

            reward = MouseEnv.get_reward(state, action)  # reward of taking action in current state

            next_state = MouseEnv.get_next_state(state, action=action)
            next_b_action = epsilon_greedy(state=next_state, epsilon=epsilon, Qtable=Q)
            next_p_action = greedy(state=next_state, Qtable=Q)

            td_target = reward + discount_rate * Q[next_state, next_p_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += learning_rate * td_error

            state = next_state
            action = next_b_action
    return Q  # type: ignore

def qlearning(num_of_episodes, alpha, discount, epsilon):
    """Runs the sarsa learning algorithm and returns a policy

    Returns:
        policy (dict): mapping of state to action
    """
    Q_qlearning = run_qlearning(
        num_of_episodes,
        learning_rate=alpha,
        discount_rate=discount,
        epsilon=epsilon
    )
    qlearning_policy = np.argmax(Q_qlearning, axis=1)
    policy = {}
    for i, value in enumerate(qlearning_policy):
        policy[i] = int(value)
    return policy, Q_qlearning

def run_SARSA(
    num_of_episodes: int,
    learning_rate: float,
    discount_rate: float,
    epsilon: float,
) -> np.typing.NDArray:
    """Runs the SARSA learning algorithm and returns Q-values

    Args:
        env (MouseEnv): the environment
        num_of_episodes (int): amount of episode runs
        learning_rate (float): the learning rate
        discount_factor (float): the discount factor
        epsilon (float): the epsilon greedy value

    Returns:
        Q-values (np.typing.NDArray): A table of Q values of state and action

    """
    Q = np.zeros((MouseEnv.num_of_states, MouseEnv.num_of_actions))
    for _ in range(num_of_episodes):

        state = np.random.randint(0, MouseEnv.num_of_states)
        action = epsilon_greedy(state=state, epsilon=epsilon, Qtable=Q)
        total_reward = 0
        done = False

        while not done:

            done = MouseEnv.is_terminal_obs(state)
            reward = MouseEnv.get_reward(
                state, action
            )  # reward of  taking action in current state
            next_state = MouseEnv.get_next_state(state, action=action)
            next_action = epsilon_greedy(state=next_state, epsilon=epsilon, Qtable=Q)

            TD_target = reward + discount_rate * Q[next_state, next_action]
            TD_error = TD_target - Q[state, action]
            Q[state, action] += learning_rate * TD_error

            state = next_state
            action = next_action

            total_reward += reward
    return Q  # type: ignore


def SARSA(num_of_episodes, alpha, discount, epsilon):
    """Runs the sarsa learning algorithm and returns a policy

    Returns:
        policy (dict): mapping of state to action
    """
    Q_sarsa = run_SARSA(
        num_of_episodes,
        learning_rate=alpha,
        discount_rate=discount,
        epsilon=epsilon,
    )
    SARSA_policy = np.argmax(Q_sarsa, axis=1)
    policy = {}
    for i, value in enumerate(SARSA_policy):
        policy[i] = int(value)
    return policy, Q_sarsa