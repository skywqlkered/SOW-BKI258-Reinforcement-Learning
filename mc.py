from environment import MouseEnv
import random

def montecarlo_prediction(policy: dict[int, int], env: MouseEnv, num_of_episodes: int, gamma: float = 0.5):
    returns: dict[int, list[float]] = {state : [] for state in range(env.num_of_states)}

    for _ in range(num_of_episodes):
        
        episode = generate_episode(policy, env)
        g = 0
        
        visited_states = [state[0] for state in episode]
        
        while episode:
            state, action, reward = episode.pop()
            g = gamma * g + reward
            if state not in visited_states[:len(episode)]:
                returns[state].append(g)
                

    values: dict[int, float]  = {} 
    for state in range(env.num_of_states):    
        if returns[state]:
            values[state] = sum(returns[state]) / len(returns[state])
        else:
            values[state] = 0.0 # not visited state
    return values

def generate_episode(policy, env: MouseEnv, max_steps = 100) -> list[tuple[int, int, float]]:
    state = random.randrange(0, env.num_of_states)
    steps = 0
    episode = []
    while not env.is_terminal_obs(state) and steps < max_steps:
        action = policy[state]
        reward = env.get_reward(state, action)
        episode.append((state, action, reward))

        next_state = env.get_next_state(state, action)
        state = next_state 

        steps += 1
    
    return episode


def predictinator():
    env = MouseEnv()
    policy = {state: random.randint(0, env.num_of_actions-1) for state in range(env.num_of_states)}
    num_of_episodes = 1000
    discount = 0.5

    print(montecarlo_prediction(policy, env, num_of_episodes, discount))
    
if __name__ == "__main__":
    predictinator()