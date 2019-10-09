import numpy as np

np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from gym import wrappers
import gym


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class PolicyApproximator(nn.Module):
   
    def __init__(self, input_shape, output_shape):
        super(PolicyApproximator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_shape),
            nn.Softmax(dim=1),
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # self.apply(init_weights)

    def forward(self, state):
        return self.model(state)

    def get_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        probs = self.forward(state).cpu()

        cat_probs = Categorical(probs)
        action = cat_probs.sample()
        log_prob = cat_probs.log_prob(action)
        # print(
        #     f"\n\n probs-> {probs}  cat_probs->  {cat_probs}  action -> {action}  log_prob-> {log_prob}"
        # )

        return action.item(), log_prob

    def get_loss(self, state, target):
        # _, log_prob = self.get_action(state)
        policy_loss = []
        for log_prob in state:
            policy_loss.append(-log_prob.detach() * target)
        total_policy_loss = torch.cat(policy_loss).mean()
        return total_policy_loss

    def update(self, state, target):
        policy_loss = self.get_loss(state, target)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss


class ValueApproximator(nn.Module):
    def __init__(self, input_shape):
        super(ValueApproximator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def forward(self, state):
        return self.model(state)

    def predict_reward(self, state):
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        reward = self.forward(state).cpu()
        return reward

    def update(self, state, actual_reward):

        predicted_reward = self.predict_reward(state)
        value_loss = self.criterion(actual_reward,predicted_reward)
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()
        return value_loss


def discount_rewards(rewards, gamma):
    discounted_rewards = [gamma ** i * rewards[i] for i in range(len(rewards))]
    return discounted_rewards


def normalise_rewards(rewards):
    rewards_mean = rewards.mean()
    rewards_std = rewards.std()
    rewards_norm = [
        (rewards[i] - rewards_mean) / (rewards_std + 1e-10) for i in range(len(rewards))
    ]
    return rewards_norm


def apply_reinforce(policy_approximator, value_approximator, num_episodes=100, gamma=1):

    print(policy_approximator, value_approximator)
    best_score = -1000
    scores = []
    for episode_i in range(1, num_episodes):
        # print(f"\n\n ---episode ----- {episode_i}")
        saved_log_probs, rewards, states_actions_rewards, value_loss, policy_loss = (
            [],
            [],
            [],
            [],
            [],
        )
        time_step, total_reward = 0, 0
        state = env.reset()
        done = False

        while not done:
            action, log_prob = policy_approximator.get_action(state)
            saved_log_probs.append(log_prob)
            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            time_step += 1
            states_actions_rewards.append((state, action, reward))
            state = new_state

            if done:
                total_reward = sum(rewards)
                # print(f"Done with total reward {total_reward} at time step {time_step}")
                scores.append(total_reward)

                discounted_rewards = discount_rewards(rewards, gamma)
                norm_rewards = normalise_rewards(np.asarray(discounted_rewards))

                for step_idx, step_dict in enumerate(states_actions_rewards):
                    state, action, reward = step_dict
                    discounted_R = sum(discounted_rewards[step_idx:])
                    baseline_value = value_approximator.predict_reward(state)
                    advantage = discounted_R - baseline_value
                    _value_loss = value_approximator.update(
                        state, torch.tensor(discounted_R)
                    )
                    _policy_loss = policy_approximator.update(
                        saved_log_probs, advantage
                    )

                    policy_loss.append(_policy_loss)
                    value_loss.append(_value_loss)

                # print(f"Done with total reward {total_reward} at time step {time_step}")

                if episode_i % (num_episodes / 10) == 0 or episode_i == 1:
                    print(
                        f"Episode - > {episode_i}  Score -> {total_reward} best score ->   {best_score}\n"
                        f"policy loss - > {sum(policy_loss)/len(policy_loss)} value loss - > {sum(value_loss)/len(value_loss)}  "
                    )

                if total_reward > best_score:
                    print(
                        f"\n --- saving model with score {total_reward} at Episode - > {episode_i} \n"
                    )
                    best_score = total_reward
                    torch.save(policy_approximator, model_path)
    return scores


def test_policy():
    model = torch.load(model_path)
    model.eval()

    for i in range(10):
        total_reward = 0
        done = False
        state = env.reset()
        while not done:
            action, _ = model.get_action(state)
            env.render()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        print(total_reward)

    env.close()


if __name__ == "__main__":
    env_name = "CartPole-v0"
    model_path = f"models/{env_name}_policygrad"
    env = gym.make(env_name)
    env.seed(0)
    # env = wrappers.Monitor(
    #     env,
    #     f"Saved_Videos/policy_grad/reinforce/{env_name}",
    #     resume=True,
    #     force=True,
    #     video_callable=lambda episode_id: episode_id % 100 == 0,
    # )

    device = "cpu"
    obs_shape = env.observation_space.shape
    action_shape = env.action_space
    print(obs_shape, action_shape)

    policy_approximator = PolicyApproximator(input_shape=4, output_shape=2)

    value_approximator = ValueApproximator(input_shape=4)
    rewards = apply_reinforce(
        policy_approximator, value_approximator, num_episodes=2000, gamma=0.99
    )
    smoothed_rewards = [
        np.mean(rewards[max(0, i - 10) : i + 1]) for i in range(len(rewards))
    ]

    test_policy()
    plt.figure(figsize=(12, 8))
    plt.plot(smoothed_rewards)
    plt.title(f"REINFORCE Baseline with Policy Estimation for {env_name}")
    plt.show()
