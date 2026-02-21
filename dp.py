from environment import MouseEnv
import random
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_policy(policy, env: MouseEnv, theta, gamma):
    Vs: dict[float] = {state: 0.0 for state in range(env.num_of_states)} # type:ignore
    
    delta = 1000
    delta_history = []
    while not delta < theta:
        delta = 0
        for state in range(env.num_of_states):
            v = Vs[state]
            if state == env.num_of_states - 1:
                continue
            
            state_total = 0
            action = policy[state]
            reward = env.get_reward(state, action)
            state_prime = env.get_next_state(state, action)
            
            if reward == env.lose_punishment or reward == env.win_reward:
                action_total = reward
            else: 
                action_total = reward + (gamma * Vs[state_prime]) # p(state | action) = 1, so this isnt included
            state_total = action_total
            
            Vs[state] = state_total
            delta = max(delta, abs(v - Vs[state]))
        delta_history.append(delta)
    return Vs, delta_history
    
def evaluator_inator():
    env = MouseEnv()
    policy = {state: random.randint(0, 3) for state in range(env.num_of_states)}
    theta = 1e-4
    discount_rate = 0.8
    print(evaluate_policy(policy, env, theta, discount_rate))

def improve_policy(value_function: dict, policy: dict, env: MouseEnv, gamma: float):
    # Mechanism: Maximization
    # Goal: Control (Improve Policy)
    # Tra probabilities: 
    policy_stable = True
    refined_policy = {}
    for state in range(env.num_of_states):
        old_action = policy[state]
        action_values = {}
        for action in range(env.num_of_actions):
            action_total = 0
            for state_prime in range(env.num_of_states):
                action_total +=  + gamma * value_function[state_prime]
            action_values[action] = action_total
        best_action = max(action_values, key=lambda x: action_values[x])
        if best_action != old_action:
            policy_stable = False
        refined_policy[state] = best_action
    return refined_policy, policy_stable


if __name__ == "__main__":
    evaluator_inator()