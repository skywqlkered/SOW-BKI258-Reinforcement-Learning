from environment import MouseEnv

def montecarlo_prediction(policy: dict[int, int], env: MouseEnv, num_of_episodes: int, gamma: float = 0.5):
    returns: dict[int, list[float]] = {state : [] for state in range(env.num_of_states)}

    for _ in range(num_of_episodes):
        episode = generate_episode(policy, env)
        g = 0
        while episode:
            state, action, reward = episode.pop()
            g = gamma * g + reward
            if state not in episode:
                returns[state].append(g)
                break

    values: dict[int, float] = {state : sum(returns[state]) / len(returns[state]) for state in range(env.num_of_states)}
    return values

def generate_episode(policy, env: MouseEnv) -> list[tuple[int, int, float]]:
    pass