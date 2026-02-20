from environment import MouseEnv
import random
def evaluate_policy(policy, env: MouseEnv, theta, gamma):
    Vs: dict[float] = {state: 0.0 for state in range(env.num_of_states)} # type:ignore
    
    delta = 100
    
    while delta < theta:
        delta = 0
        for state in range(env.num_of_states):
            previous_value = None
            v = Vs[state]
            
            state_total = 0
            for action in range(env.num_of_actions):
                action_total = 0
                for state_prime in range(env.num_of_states):
                    action_total += 1 * -1 + gamma * Vs[state_prime]
                state_total = policy[state] * action_total
            
            Vs[state] = round(state_total, 1)
            delta = max(delta, v - Vs[v])
    return Vs
    
def evaluator_inator():
    env = MouseEnv()
    policy = {state: random.randint(0, 3) for state in range(env.num_of_states)}
    theta = 0.01
    discount_rate = 0.5
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


evaluator_inator()