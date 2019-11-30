# Deep Learning Applications - Project "Little Fighter 2"

############################# IMPORT STATEMENTS ########################################################
import argparse
import os, sys

#Import lf2gym
import random
import lf2gym
from time import sleep

#Import Keras modules
from keras.layers import Dense, Flatten, Input
from keras import Model, Sequential
import numpy as np
from keras.optimizers import Adam
from collections import deque

############################# SETUP LF2 PARAMETERS ########################################################
parser = argparse.ArgumentParser()
parser.add_argument("--player", default=lf2gym.Character.Louis, help="your character") # the AI
parser.add_argument("--opponent", default=lf2gym.Character.Freeze, help="your enemy") # the AI's enemy
parser.add_argument("--interval", default=0.9, help="your enemy")
args = parser.parse_args()

# Add import path for the lf2gym
sys.path.append(os.path.abspath('..'))


LOAD_PROGRESS_FROM_MODEL = False
EPISODES = 2
SAVE_PROGRESS_TO_MODEL = True

########################### FORCE KERAS TO USE CPU #####################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

############################# SETUP THE DEEP Q AGENT ########################################################
# Deep Q-learning Agent

# FIND INFORMATION WITH LINK: https://keon.io/deep-q-learning/
# AS WELL AS UNDER THIS LINK: https://github.com/keon/deep-q-learning/blob/master/dqn.py

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # for remembering past actions taking -> to help make better decisions
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def saveModel(self):
        self.model.save_weights("./app_model/model.h5")
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("./app_model/model.json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

    def loadModel(self):
        self.model.load_weights("./app_model/model.h5")
        print("Loaded model from disk")

############################# DEEP Q LEARNING ########################################################

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = lf2gym.make(startServer=True, driverType=lf2gym.WebDriver.Chrome,
                      background=lf2gym.Background.HK_Coliseum,
                      characters=[args.player, args.opponent], #[Me/AI, bot]
                      difficulty=lf2gym.Difficulty.Crusher,
                      versusPlayer=False) # versusPlayer= False means Agent is playing against bot instedad of user

    state_size = env.observation_space.n[0] * env.observation_space.n[1]
    print("state_size: " + str(state_size))
    action_size = int(env.action_space.n)
    print("action_size: " + str(action_size))

    agent = DQNAgent(state_size, action_size)
    if LOAD_PROGRESS_FROM_MODEL:
        agent.loadModel()

    done = False
    batch_size = 32
    # Iterate the game
    for e in range(EPISODES):
        # reset state in the beginning of each game
        state = env.reset()   # returns 243200 for our game as screen size input (160,380,4)
        state = np.reshape(state, [4, state_size])   # chose here 4, (160*380)  this equals the screen size with 4 (frames?)
        for time_t in range(500):
            # env.render()  # no need to activate render
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10

            next_state = np.reshape(next_state, [4, (160*380)])
            agent.remember(state, action, reward, next_state, done)  # Remember the previous state, action, reward, and done
            state = next_state  # make next_state the new current state for the next frame.

            if done: # # done becomes True when the game ends
                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, time_t))
                break
            # train the agent with the experience of the episode
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # save progress to model after finishing the last episode
        if e == (EPISODES - 1):
            agent.saveModel()
