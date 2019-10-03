import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from gym import wrappers
import gym


class Policy(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)

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


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def apply_reinforce(num_episodes=100, gamma=1):

    scores = []
    best_score = -1000
    for episode_i in range(1, num_episodes):
        # print(f"\n\n ---episode ----- {episode_i}")
        saved_log_probs = []
        rewards = []
        state = env.reset()
        done = False
        time_step = 0
        total_reward = 0

        while not done:
            action, log_prob = policy.get_action(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            time_step += 11

            if done:

                total_reward = sum(rewards)
                # print(f"Done with total reward {total_reward} at time step {time_step}")
                scores.append(total_reward)

                discounted_R = sum(
                    [gamma ** i * rewards[i] for i in range(len(rewards))]
                )
                # print(rewards, R)

                policy_loss = []
                for log_prob in saved_log_probs:
                    policy_loss.append(-log_prob * discounted_R)
                # print(f"\n policy_loss {policy_loss}")
                total_policy_loss = torch.cat(policy_loss).sum()
                # print(f"\n total_policy_loss {total_policy_loss}")

                optimizer.zero_grad()
                total_policy_loss.backward()
                optimizer.step()

                if episode_i % (num_episodes / 100) == 0 or episode_i == 1:
                    print(
                        f"Episode - > {episode_i}  Score -> {total_reward} best score ->   {best_score}  "
                    )

                if total_reward > best_score:
                    print(
                        f"\n --- saving model with score {total_reward} at Episode - > {episode_i} \n"
                    )
                    best_score = total_reward
                    torch.save(policy, model_path)


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
    env_name = "CartPole-v1"
    model_path = f"models/{env_name}_policygrad"
    env = gym.make(env_name)
    env.seed(0)
    env = wrappers.Monitor(
        env,
        f"Saved_Videos/policy_grad/reinforce/{env_name}",
        resume=True,
        force=True,
        video_callable=lambda episode_id: episode_id % 100 == 0,
    )

    device = "cpu"
    obs_shape = env.observation_space.shape
    action_shape = env.action_space
    print(obs_shape, action_shape)
    policy = Policy(4, 2)
    policy.apply(init_weights)
    print(policy)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    apply_reinforce(1000, 0.99)
    test_policy()
