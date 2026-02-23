from environment import MouseEnv

def montecarlo_prediction(policy: dict[int, int], env: MouseEnv, num_of_episodes: int, gamma: float = 1.0):
    values: dict[float] = {state: 0.0 for state in range(env.num_of_states)} # type:ignore
    returns = [[] for _ in range(env.num_of_states)] # List of lists to store returns for each state

    for _ in range(num_of_episodes):
        episode = generate_episode(policy, env)
        g = 0
        for state, action, reward in episode[::-1]:
            pass


    return values

def generate_episode(policy, env: MouseEnv) -> list[tuple[int, int, float]]:
    pass