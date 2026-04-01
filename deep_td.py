import tensorflow as tf
import numpy as np
from environment import MouseEnv
from collections import deque
import random

class Deep_QNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(Deep_QNetwork, self).__init__()
        self.NNlayer1 = tf.keras.layers.Dense(units=24, activation="relu")
        self.NNlayer2 = tf.keras.layers.Dense(units=24, activation="relu")
        self.outputlayer = tf.keras.layers.Dense(num_actions, activation="linear")

    def call(self, inputs, training=None, mask=None):
        x = self.NNlayer1(inputs=inputs)
        x = self.NNlayer2(x)
        return self.outputlayer(x)

def epsilon_greedy(state: int, epsilon: float, DQN_agent) -> int:
    """Returns an action based on the epsilon value

    Args:
        state (int): number of current stateF
        epsilon (float): the epsilon value

    Returns:
        action (int) the next action to take

    """
    if np.random.random() < epsilon:
        return np.random.randint(low=0, high=MouseEnv.num_of_actions)
    else:
        state_vector = tf.one_hot([state], MouseEnv.num_of_states)
        q_values = DQN_agent(state_vector)

        return int(np.argmax(q_values[0]))

def run_deepQLearning(
    num_of_episodes: int,
    learning_rate: float,
    discount_rate: float,
    epsilon: float,
    epsilon_decay: float,
    max_steps: int,
    batch_size: int,
    track_history: bool = False,
    track_rewards: bool = False,
):
    # setup functions
    loss_function = tf.keras.losses.MeanSquaredError()
    optimizer_function = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    replay_buffer = deque(maxlen=5000)
    
    #setup training and target NN
    DQN_agent = Deep_QNetwork(MouseEnv.num_of_actions)
    target_agent = Deep_QNetwork(MouseEnv.num_of_actions)
    target_agent.set_weights(DQN_agent.get_weights())
    
    target_update_freq = 10
    all_states = tf.one_hot(list(range(MouseEnv.num_of_states)), MouseEnv.num_of_states)
    q_snapshots: list[np.ndarray] = []
    reward_list: list[float] = []
    
    # loop trough states
    for episode in range(num_of_episodes):
        state = np.random.randint(0, MouseEnv.num_of_states)
        episode_reward = 0.0
        
        # loop until terminal state or max steps is reached.
        for _ in range(max_steps):
            action = epsilon_greedy(state=state, epsilon=epsilon, DQN_agent=DQN_agent)

            reward = MouseEnv.get_reward(state=state, action=action)
            next_state = MouseEnv.get_next_state(state, action=action)
            done = MouseEnv.is_terminal_obs(next_state)
            episode_reward += float(reward)

            replay_buffer.append((state, action, reward, next_state, done))
            
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                
                states_vector = tf.one_hot(list(states), MouseEnv.num_of_states)
                next_states_vector = tf.one_hot(list(next_states), MouseEnv.num_of_states)
                rewards = tf.cast(rewards, tf.float32)
                dones = tf.cast(dones, tf.float32)
                
                #use target NN to calculate the target Q-values using bellman things
                next_q_values = target_agent(next_states_vector)
                max_next_q_value = tf.reduce_max(next_q_values, axis=1)
                targets = rewards + discount_rate * max_next_q_value * (1- dones)
                
                with tf.GradientTape() as tape:
                    
                    current_q_values = DQN_agent(states_vector)
                    action_mask = tf.one_hot(list(actions), MouseEnv.num_of_actions)
                    
                    # get q value of action
                    predicted_q = tf.reduce_sum(current_q_values * action_mask, axis=1)
                    loss = loss_function(targets, predicted_q)

                gradients = tape.gradient(loss, DQN_agent.trainable_variables)
                optimizer_function.apply_gradients(zip(gradients, DQN_agent.trainable_variables))

            # stop if done
            if done:
                break
            state = next_state

        # modified so it always explores a lil bit
        epsilon = max(0.01, epsilon * epsilon_decay)

        # update the target network according to frequency
        if episode % target_update_freq == 0:
            target_agent.set_weights(DQN_agent.get_weights())

        if track_history:
            q_snapshots.append(DQN_agent(all_states).numpy())
        if track_rewards:
            reward_list.append(episode_reward)

        if (episode + 1) % 50 == 0:
            print(f"Episode number {episode+1} out of {num_of_episodes} completed, epsilon = {epsilon:.2f}")

    if track_history or track_rewards:
        return DQN_agent, q_snapshots, reward_list

    return DQN_agent

def deepQLearning(num_of_episodes,
                  alpha,
                  discount,
                  epsilon,
                  max_steps,
                  batch_size,
                  epsilon_decay=1.0,
                  track_history: bool = False,
                  track_rewards: bool = False):
    """Runs the sarsa learning algorithm and returns a policy

    Returns:
        policy (dict): mapping of state to action
    """
    deep_result = run_deepQLearning(
        num_of_episodes,
        learning_rate=alpha,
        discount_rate=discount,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        max_steps=max_steps,
        batch_size=batch_size,
        track_history=track_history,
        track_rewards=track_rewards,
    )
    q_history: list[np.ndarray] = []
    reward_history: list[float] = []
    if track_history or track_rewards:
        trained_agent, q_history, reward_history = deep_result  # type: ignore
    else:
        trained_agent = deep_result  # type: ignore

    all_states = tf.one_hot(list(range(MouseEnv.num_of_states)), MouseEnv.num_of_states)
    q_values = trained_agent(all_states).numpy()  # type: ignore
    qlearning_policy = np.argmax(q_values, axis=1)
    policy = {}
    for i, value in enumerate(qlearning_policy):
        policy[i] = int(value)

    if track_history or track_rewards:
        return policy, q_values, q_history, reward_history

    return policy, q_values


