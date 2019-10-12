import gym
import numpy as np
import random


def double_q(num_episodes=1000):
    epsilon = 0.9
    table_q1 = np.zeros((state_size, action_size))
    table_q2 = np.zeros((state_size, action_size))
    total_rewards = []
    for episode_idx in range(1, num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action_random_probability = random.uniform(0, 1)
            if action_random_probability > epsilon:
                actions_q1_q2 = np.array([table_q1[state, :] + table_q2[state, :]])
                action = np.argmax(actions_q1_q2)
            else:
                action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward

            update_random_probability = random.uniform(0, 1)

            if update_random_probability > 0.5:
                action_q1 = np.argmax(table_q1[new_state, action])
                table_q1[state, action] = table_q1[state, action] + learning_rate * (
                    reward
                    + (
                        gamma * np.max(table_q2[new_state, action_q1])
                        - table_q1[state, action]
                    )
                )

            else:
                action_q2 = np.argmax(table_q2[new_state, action])
                table_q2[state, action] = table_q2[state, action] + learning_rate * (
                    reward
                    + (
                        gamma * np.max(table_q1[new_state, action_q2])
                        - table_q2[state, action]
                    )
                )
            state = new_state

            if done:
                total_rewards.append(episode_reward)
                print(f"Score of the {episode_idx} episode --> {episode_reward} ")
                epsilon = min_epsilon + (1 - min_epsilon) * np.exp(
                    -decay_rate * episode_idx
                )
                break
    return total_rewards


if __name__ == "__main__":

    env = gym.make("FrozenLake-v0")
    action_size = env.action_space.n
    print("Action size ", action_size)

    state_size = env.observation_space.n
    print("State size ", state_size)

    min_epsilon = 0.01
    learning_rate = 0.01  # Learning rate
    gamma = 0.99  # Discounting rate
    decay_rate = 0.003
    num_episode = 10000

    total_rewards = double_q(num_episodes=num_episode)
    print(sum(total_rewards) / len(total_rewards))

