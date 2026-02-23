

def MC_prediction(policy, env, num_of_episodes, gamma=1.0):
    Vs: dict[float] = {state: 0.0 for state in range(env.num_of_states)} # type:ignore
    returns = [[] for _ in range(env.num_of_states)] # List of lists to store returns for each state


    return Vs