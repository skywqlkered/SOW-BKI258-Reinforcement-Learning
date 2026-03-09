from environment import MouseEnv
import numpy as np



def epsilon_greedy(state: int, epsilon: float, Qtable: np.typing.NDArray, env) - > int:
    """Returns an action based on the epsilon value

    Args:
        state (int): number of current stateF
        epsilon (float): the epsilon value

    Returns:
        action (int) the next action to take

    """
    if np.random.random() < epsilon:
        return np.random.randint(low=0, high=env.num_of_actions)
    else:
        return int(np.argmax(Qtable[state]))

def run_SARSA(env: MouseEnv, num_of_episodes: int, learning_rate: float, discount_rate: float, epsilon: float)
    Q = np.zeros((env.num_of_states, env.num_of_actions))

    for episode in range(num_of_episodes):
        steps = 0
        state = np.random.randint(0, env.num_of_states)
        action = epsilon_greedy(state=state, epsilon=epsilon, Qtable=Q, env=env)
        total_reward = 0
        done = False

        while not done:
            next_state = env.get_next_state(state, action=action)
            next_action = epsilon_greedy(
            state=next_state, epsilon=epsilon, Qtable=Q, env=env)
            done = env.is_terminal_obs(state)
            reward = env.get_reward(state, action)
            
            Q[state, action] = learning_rate * (reward + discount_rate * Q[next_state, next_action] - Q[state, action])
            steps += 1
            state = next_state
            action = next_action
            total_reward += reward
    return Q


def SARSA(env, num_of_episodes, alpha, discount, epsilon):
    """

    Returns:
        tuple:
            - dict[int, int]: Converged policy mapping from state index to action.
            - dict[int, float]: Final state-value function mapping from state index to estimated value.
    """
    # Initialize policy randomly for all states
    policy = {
        state: np.random.randint(0, MouseEnv.num_of_actions - 1)
        for state in range(MouseEnv.num_of_states)
    }

    # Alternate between policy evaluation and policy improvement until convergence
    while True:
        values = run_SARSA(env, num_of_episodes, alpha, discount, epsilon)
        new_policy = control(values, discount)
        # Convergence check: stop if the policy is stable (unchanged)
        if new_policy == policy:
            break
        policy = new_policy.copy()

    return policy, values

    


