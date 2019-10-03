"""
Test model created  by behavioral cloning expert data by berkely cs234 
Example usage:
python test_bc.py Humanoid-v2
"""

import numpy as np
from keras.models import load_model
import gym
import argparse
from gym import wrappers


def run_bc_policy(model, roll_outs=100, render=False):

    max_steps = 10000
    returns = []
    observations = []
    actions = []
    render = False

    for time_step in range(roll_outs):
        obs = env.reset()
        done = False
        totalr = 0.0
        steps = 0
        while not done:
            action = model.predict(np.reshape(np.array(obs), (1, len(obs))))
            obs, r, done, _ = env.step(action)

            totalr += r
            steps += 1
            if render:
                env.render()
            if steps >= max_steps:
                break

        returns.append(totalr)
    return returns


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)
    args = parser.parse_args()
    env_name = args.env_name
    print(env_name)

    agent_name = env_name
    roll_outs = 100
    env = gym.make(agent_name)
    env = wrappers.Monitor(
        env,
        f"Saved_Videos/hw1/{env_name}/",
        resume=True,
        force=True,
        video_callable=lambda time_step: time_step % 10 == 0,
    )

    model = load_model(f"./models/hw1/{agent_name}.h5")
    returns = run_bc_policy(model, roll_outs=roll_outs, render=False)

    print(f"returns = {returns}")
    print(f"mean return = {np.mean(returns)}")
    print(f"std of return = {np.std(returns)}")
