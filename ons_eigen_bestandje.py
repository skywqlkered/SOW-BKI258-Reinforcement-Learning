    
from environment import MouseEnv

def run_value_iteration(environment: MouseEnv):
    states = range(environment.num_of_states)
    print("Running value iteration...")

    print("Actions:", environment.action_space)

    values = {state : 0.0 for state in states}
    delta = 0
    while delta >= 0:
        delta = 0
        for state in states:
            v = values[state]

            values[state] = max(get_action_value(environment, state, action, values) for action in range(environment.num_of_actions))
            delta = max(delta, abs(v - values[state]))
    return values

def get_action_value(environment: MouseEnv, state, action, values):
    action_value = 0
    for next_state, probability in environment.get_transition_probabilities(state, action):
        reward = environment.get_reward(state, action, next_state)
        action_value += probability * (reward + environment.gamma * values[next_state])
    return action_value



if __name__ == "__main__":
    env = MouseEnv()
    # Check if the get_state_from_obs function works correctly
    states = range(env.num_of_states)
    for state in states:
        obs = env.get_observation_from_state(state)
        assert env.get_state_from_obs(obs) == state, f"State {state} does not match observation {obs}"
