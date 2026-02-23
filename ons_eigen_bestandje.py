from environment import MouseEnv
from numpy import inf
from typing import Mapping


def run_value_iteration(environment: MouseEnv, theta: float = 1e-8, gamma: float = 1.0) -> dict[int, float]:
    """Compute optimal state values with value iteration."""
    states = range(environment.num_of_states)
    values = {state: 0.0 for state in states}

    delta = float("inf")
    while delta >= theta:
        delta = 0.0
        for state in states:
            if environment.is_terminal_obs(state):
                values[state] = 0.0
                continue

            v = values[state]
            values[state] = max(
                get_action_value(environment, state, action, values, gamma)
                for action in range(environment.num_of_actions)
            )
            delta = max(delta, abs(v - values[state]))

    return values


def get_action_value(environment: MouseEnv, state, action, values):
    action_value = 0
    for next_state, probability in environment.get_transition_probabilities(state, action):
        reward = environment.get_reward(state, action, next_state)
        action_value += probability * (reward + environment.gamma * values[next_state])
    return action_value


def create_policy(env: MouseEnv, values: dict[int, float]) -> dict[int, int]:
    """
    Returns a policy based on the result of the value iteration algorithm

    Args:
        env (MouseEnv): the environment
        values (dict[int, float]): the value iteration results

    Returns:
        policy (dict[int, int]): the policy, which returns the action for each state with the max value
    """
    policy: dict[int, int] = {}
    for state in range(env.num_of_states):
        best_action: tuple[int, float] = (-1, -inf)
        for action in range(env.num_of_actions):
            action_value = values[MouseEnv.get_next_state(state, action)]
            if action_value > best_action[1]:
                best_action = (action, action_value)
        policy[state] = best_action[0]

    return policy


if __name__ == "__main__":
    env = MouseEnv()
    # Check if the get_state_from_obs function works correctly
    states = range(env.num_of_states)
    for state in states:
        obs = env.get_observation_from_state(state)
        assert env.get_state_from_obs(obs) == state, f"State {state} does not match observation {obs}"
