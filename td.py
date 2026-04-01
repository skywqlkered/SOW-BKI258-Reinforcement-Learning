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
    return int(np.argmax(Qtable[state]))


def run_qlearning(
    num_of_episodes: int,
    learning_rate: float,
    discount_rate: float,
    epsilon: float,
    epsilon_decay: float,
    track_history: bool = False,
    track_rewards: bool = False,
    convergence_check: bool = False,
    convergence_range: int = 100):
    """Runs the Q-learning algorithm and returns Q-values

        Args:
            num_of_episodes (int): amount of episode runs
            learning_rate (float): the learning rate
            discount_rate (float): the discount factor
            epsilon (float): the epsilon greedy value
            epsilon_decay (float): the decay rate for epsilon after each episode

    Returns:
        Q-values (np.typing.NDArray): A table of Q values of state and action
    """

    Q = np.zeros((MouseEnv.num_of_states, MouseEnv.num_of_actions))
    reward_list: list[float] = []
    list_of_Q: list[np.ndarray] = []
    previous_policy: dict[int, int] = {}
    convergence_counter: int = 0

    for _ in range(num_of_episodes):
        total_reward: float = 0

        state = np.random.randint(0, MouseEnv.num_of_states)
        action = epsilon_greedy(state=state, epsilon=epsilon, Qtable=Q)
        epsilon *= epsilon_decay

        while not MouseEnv.is_terminal_obs(state):

            reward = MouseEnv.get_reward(
                state, action
            )  # reward of taking action in current state

            next_state = MouseEnv.get_next_state(state, action=action)
            next_b_action = epsilon_greedy(state=next_state, epsilon=epsilon, Qtable=Q)
            next_p_action = greedy(state=next_state, Qtable=Q)

            td_target = reward + discount_rate * Q[next_state, next_p_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += learning_rate * td_error

            state = next_state
            action = next_b_action
            total_reward += reward

        if convergence_check:
            qlearning_policy = np.argmax(Q, axis=1)
            policy = {}
            for i, value in enumerate(qlearning_policy):
                policy[i] = int(value)
            if policy == previous_policy:
                convergence_counter += 1
            else:
                convergence_counter = 0
            if convergence_counter >= convergence_range:
                break
            previous_policy = policy.copy()

        if track_rewards:
            reward_list.append(total_reward)
        if track_history:
            list_of_Q.append(Q.copy())

    if track_history or track_rewards:
        return Q, list_of_Q, reward_list

    return Q  # type: ignore

def qlearning(num_of_episodes,
              alpha,
              discount,
              epsilon,
              epsilon_decay=1.0,
              track_history: bool = False,
              track_rewards: bool = False,
              convergence_check: bool = False,
              convergence_range: int = 100):
    """Runs the sarsa learning algorithm and returns a policy

    Returns:
        policy (dict): mapping of state to action
    """
    qlearning_result = run_qlearning(
        num_of_episodes,
        learning_rate=alpha,
        discount_rate=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        track_history=track_history,
        track_rewards=track_rewards,
        convergence_check=convergence_check,
        convergence_range=convergence_range,
    )
    q_history: list[np.ndarray] = []
    reward_history: list[float] = []
    if track_history or track_rewards:
        Q_qlearning, q_history, reward_history = qlearning_result  # type: ignore
    else:
        Q_qlearning = qlearning_result  # type: ignore
    qlearning_policy = np.argmax(Q_qlearning, axis=1)  # type: ignore
    policy = {}
    for i, value in enumerate(qlearning_policy):
        policy[i] = int(value)

    if track_history or track_rewards:
        return policy, Q_qlearning, q_history, reward_history

    return policy, Q_qlearning


def run_SARSA(
    num_of_episodes: int,
    learning_rate: float,
    discount_rate: float,
    epsilon: float,
    epsilon_decay: float,
    track_history: bool = False,
    track_rewards: bool = False,
    convergence_check: bool = False,
    convergence_range: int = 100,
):
    """Runs the SARSA learning algorithm and returns Q-values

    Args:
        env (MouseEnv): the environment
        num_of_episodes (int): amount of episode runs
        learning_rate (float): the learning rate
        discount_factor (float): the discount factor
        epsilon (float): the epsilon greedy value
        epsilon_decay (float): the decay rate for epsilon after each episode

    Returns:
        Q-values (np.typing.NDArray): A table of Q values of state and action

    """
    Q = np.zeros((MouseEnv.num_of_states, MouseEnv.num_of_actions))
    reward_list: list[float] = []
    list_of_Q: list[np.ndarray] = []
    previous_policy: dict[int, int] = {}
    convergence_counter: int = 0

    for _ in range(num_of_episodes):

        state = np.random.randint(0, MouseEnv.num_of_states)
        action = epsilon_greedy(state=state, epsilon=epsilon, Qtable=Q)
        epsilon *= epsilon_decay
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

        if convergence_check:
            SARSA_policy = np.argmax(Q, axis=1)
            policy = {}
            for i, value in enumerate(SARSA_policy):
                policy[i] = int(value)
            if policy == previous_policy:
                convergence_counter += 1
            else:
                convergence_counter = 0
            if convergence_counter >= convergence_range:
                break

            previous_policy = policy.copy()

        if track_rewards:
            reward_list.append(total_reward)
        if track_history:
            list_of_Q.append(Q.copy())

    if track_history or track_rewards:
        return Q, list_of_Q, reward_list

    return Q  # type: ignore


def SARSA(num_of_episodes: int,
          alpha: float,
          discount: float,
          epsilon: float,
          epsilon_decay: float = 1.0,
          track_history: bool = False,
          track_rewards: bool = False,
          convergence_check: bool = False,
          convergence_range: int = 100):
    """Runs the sarsa learning algorithm and returns a policy

    Returns:
        policy (dict): mapping of state to action
    """
    sarsa_result = run_SARSA(
        num_of_episodes,
        learning_rate=alpha,
        discount_rate=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        track_history=track_history,
        track_rewards=track_rewards,
        convergence_check=convergence_check,
        convergence_range=convergence_range,
    )
    q_history: list[np.ndarray] = []
    reward_history: list[float] = []
    if track_history or track_rewards:
        Q_sarsa, q_history, reward_history = sarsa_result  # type: ignore
    else:
        Q_sarsa = sarsa_result  # type: ignore

    SARSA_policy = np.argmax(Q_sarsa, axis=1)  # type: ignore
    policy = {}
    for i, value in enumerate(SARSA_policy):
        policy[i] = int(value)

    if track_history or track_rewards:
        return policy, Q_sarsa, q_history, reward_history

    return policy, Q_sarsa
