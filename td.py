from environment import MouseEnv

def SARSA(env: MouseEnv, num_of_episodes: int, learning_rate: float, discount_rate: float):
    Q: dict[int, list[float]] = {state : [] for state in range(env.num_of_states)}

    for _ in range(num_of_episodes):
        
        episodes = env.generate_episode(policy=policy)
        
        for episode in episodes:
            state, action, reward = episode 
            next_state = env.get_next_state(state, action)
            next_action = policy[next_state]
            