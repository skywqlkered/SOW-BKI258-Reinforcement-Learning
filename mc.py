from dp import get_action_value
from environment import MouseEnv
import random


def montecarlo_prediction(policy: dict[int, int], num_of_episodes: int, gamma: float):
    returns: dict[int, list[float]] = {state: [] for state in range(MouseEnv.num_of_states)}

    for _ in range(num_of_episodes):

        episode = generate_episode(policy)
        g = 0

        visited_states = [state[0] for state in episode]

        while episode:
            state, action, reward = episode.pop()
            g = gamma * g + reward
            if state not in visited_states[:len(episode)]:
                returns[state].append(g)

    values: dict[int, float] = {}
    for state in range(MouseEnv.num_of_states):
        if returns[state]:
            values[state] = sum(returns[state]) / len(returns[state])
        else:
            values[state] = 0.0  # not visited state
    return values


def generate_episode(policy, max_steps=100) -> list[tuple[int, int, float]]:
    state = random.randrange(0, MouseEnv.num_of_states)
    steps = 0
    episode = []
    while not MouseEnv.is_terminal_obs(state) and steps < max_steps:
        action = policy[state]
        reward = MouseEnv.get_reward(state, action)
        episode.append((state, action, reward))

        next_state = MouseEnv.get_next_state(state, action)
        state = next_state

        steps += 1

    return episode


def control(values: dict[int, float], discount: float):
    # Loop over states
    # for each state loop over actions
    # get the value for the action from the montecarlo prediction
    # Select the highest value from the actions for each state
    # return the policy

    policy: dict[int, int] = {}

    for state in range(MouseEnv.num_of_states):
        best_action = None
        best_value = float('-inf')
        for action in range(MouseEnv.num_of_actions):
            # get the value for the action from the montecarlo prediction
            value = get_action_value(state, action, values, discount)

            if value > best_value:
                best_value = value
                best_action = action

        policy[state] = best_action

    return policy


def montecarlo(num_of_episodes, discount):
    policy = {state: random.randint(0, MouseEnv.num_of_actions-1) for state in range(MouseEnv.num_of_states)}

    while True:
        values = montecarlo_prediction(policy, num_of_episodes, discount)
        new_policy = control(values, discount)
        if new_policy == policy:
            break
        policy = new_policy.copy()
        print("Policy updated")
        print(new_policy)

    return policy, values


if __name__ == "__main__":
    montecarlo()
