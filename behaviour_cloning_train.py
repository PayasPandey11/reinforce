"""
Trains behavioral cloning model with expert data by berkely cs234 
Example usage:
python train_bc.py Humanoid-v2
"""

import pickle

import numpy as np

np.set_printoptions(suppress=True)

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.initializers import glorot_normal
from sklearn.model_selection import train_test_split
import argparse


def generate_batches(observations, actions, batch_size):
    """
    Generate batches 
    """

    num = len(observations)
    while True:
        idxs = np.random.choice(num, batch_size)
        batch_obs, batch_actions = observations[idxs], actions[idxs].astype(float)
        batch_actions = [act.flatten() for act in batch_actions]

        yield np.asarray(batch_obs), np.asarray(batch_actions)


def make_model():

    model = Sequential()
    model.add(
        Dense(
            256,
            input_shape=(input_shape,),
            activation="relu",
            kernel_initializer="glorot_normal",
        )
    )
    model.add(Dropout(0.1))
    model.add(Dense(256, activation="relu", kernel_initializer="glorot_normal"))
    model.add(Dense(out_shape))
    model.compile(
        loss="mean_squared_error", optimizer=Adam(lr=0.001), metrics=["accuracy"]
    )
    model.summary()

    return model


def run_and_save(model, epochs=30):

    batch_size = 32
    model.fit_generator(
        generate_batches(X_train, y_train, batch_size),
        validation_data=generate_batches(X_valid, y_valid, batch_size),
        epochs=epochs,
        steps_per_epoch=len(X_train) / batch_size,
        validation_steps=len(X_valid),
    )

    model.save(f"./models/hw1/{agent_name}.h5")
    print(f"Model saved to ./models/hw1/{agent_name}.h5")
    obs = observations[0]
    acs = model.predict(np.reshape(np.array(obs), (1, len(obs))))

    print(f"obesrvation {obs} \n \naction - {acs}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)
    args = parser.parse_args()
    env_name = args.env_name
    print(env_name)

    agent_name = env_name

    file = open(f"./expert_data/{agent_name}.pkl", "rb")
    data = pickle.load(file)
    print(data)

    observations, actions = data["observations"], data["actions"]
    X_train, X_valid, y_train, y_valid = train_test_split(
        observations, actions, test_size=0.1, random_state=0
    )

    print(
        f" Xtrain={X_train.shape}, ytrain = {y_train.shape},Xvalid = {X_valid.shape}"
        f", yvalid = {y_valid.shape}, lenght of X_train = {len(X_train)}, len of xvalid = {len(X_valid)}"
    )

    input_shape = X_train.shape[1]
    out_shape = y_train.shape[-1]

    print(f"input shape is {input_shape} and output shape is {out_shape}")

    model = make_model()
    run_and_save(model, epochs=30)

