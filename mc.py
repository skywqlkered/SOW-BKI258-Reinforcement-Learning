from environment import MouseEnv
import random

def montecarlo_prediction(policy: dict[int, int], env: MouseEnv, num_of_episodes: int, gamma: float = 1.0):
    values: dict[float] = {state: 0.0 for state in range(env.num_of_states)} # type:ignore
    returns = [[] for _ in range(env.num_of_states)] # List of lists to store returns for each state

    for _ in range(num_of_episodes):
        episode = generate_episode(policy, env)
        g = 0
        for state, action, reward in episode[::-1]:
            pass


    return values

def generate_episode(policy: dict, env: MouseEnv) -> list[tuple[int, int, float]]:
    state = random.randint(0, env.num_of_states)
    action = policy[state]
    reward = env.get_reward(state, action)
    if not env.is_terminal_obs(state):
        
    
    