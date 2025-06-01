import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from gym import wrappers
import sys

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


def update_sarsa_qtable(env, state, action, reward, next_state, q_table):
    "sarsa implementation"

    next_action = np.argmax(q_table[next_state, :])
    q_t = q_table[state, action]
    q_t1 = q_table[next_state, next_action]

    q_table[state, action] = q_t + learning_rate * (reward + (gamma * q_t1) - q_t)
    return q_table


def update_qlearning_qtable(env, state, action, reward, next_state, q_table):

    q_t = q_table[state, action]
    q_t1 = np.max(q_table[next_state, :])

    q_table[state, action] = q_t + learning_rate * (reward + (gamma * q_t1) - q_t)
    return q_table


def td_methods(num_eps=1000, max_steps=100, method="sarsa"):
    q_table = np.zeros((state_size, action_size))
    epsilon = 0.9
    total_rewards = []

    for episode_idx in range(1, num_eps):
        state = env.reset()
        done = False
        step = 0
        episode_scores = 0
        while not done:

            eps_greedy_action = random.uniform(0, 1)

            if eps_greedy_action > epsilon:
                # print("picking action from q table")
                action = np.argmax(q_table[state, :])

            else:
                # print("pick random action")
                action = env.action_space.sample()
                print(action)

            # env.render()

            next_state, reward, done, _ = env.step(action)
            episode_scores += reward

            if method == "sarsa":
                q_table = update_sarsa_qtable(
                    env, state, action, reward, next_state, q_table
                )
            elif method == "qlearning":
                q_table = update_qlearning_qtable(
                    env, state, action, reward, next_state, q_table
                )
            else:
                return False

            step += 1

            state = next_state

            if step > max_steps or done:
                print(f"Score of the {episode_idx} episode --> {episode_scores} ")
                epsilon = min_epsilon + (1 - min_epsilon) * np.exp(
                    -decay_rate * episode_idx
                )
                total_rewards.append(episode_scores)
                break
    return q_table, total_rewards


def test_td(qtable, num_eps=100, max_steps=100):
    total_rewards = []
    for i in range(1, num_eps):
        episode_scores = []
        done = False
        state = env.reset()
        while not done:
            # env.render()
            action = np.argmax(qtable[state, :])
            state, reward, done, _ = env.step(action)
            episode_scores.append(reward)

            if done:
                print(f"Score of the {i} episode --> {sum(episode_scores)} ")
                total_rewards.append(sum(episode_scores))

    return np.mean(total_rewards)


def analyse_and_test(
    sarsa_q_table, total_rewards_sarsa, qlearning_q_table, total_rewards_qlearning
):
    smoothed_rewards_sarsa = [
        np.mean(total_rewards_sarsa[max(0, i - 10) : i + 1])
        for i in range(len(total_rewards_sarsa))
    ]
    smoothed_rewards_qlearning = [
        np.mean(total_rewards_qlearning[max(0, i - 10) : i + 1])
        for i in range(len(total_rewards_qlearning))
    ]
    sarsa_test_mean = test_td(sarsa_q_table, num_eps=100, max_steps=100)
    qlearning_test_mean = test_td(qlearning_q_table, num_eps=100, max_steps=100)

    plt.figure(figsize=(12, 12))
    plt.plot(smoothed_rewards_sarsa, label=f"Sarsa({sarsa_test_mean})")
    plt.plot(
        smoothed_rewards_qlearning,
        label=f"Q-Learning/ Sarsa Max({qlearning_test_mean})",
    )
    plt.legend(loc="best")
    plt.title("Comparison of Sarsa and Sarsamax/Q-learning")
    plt.show()


if __name__ == "__main__":

    env = gym.make("FrozenLake-v0")
    action_size = env.action_space.n
    print("Action size ", action_size)

    state_size = env.observation_space.n
    print("State size ", state_size)

    min_epsilon = 0.01
    learning_rate = 0.1  # Learning rate
    gamma = 0.99  # Discounting rate
    decay_rate = 0.003
    num_episode = 50000
    # sarsa_q_table, total_rewards_sarsa = td_methods(
    #     num_eps=num_episode, max_steps=100, method="sarsa"
    # )
    qlearning_q_table, total_rewards_qlearning = td_methods(
        num_eps=num_episode, max_steps=100, method="qlearning"
    )
    print(sum(total_rewards_qlearning) / len(total_rewards_qlearning))

    # analyse_and_test(
    #     sarsa_q_table, total_rewards_sarsa, qlearning_q_table, total_rewards_qlearning
    # )

