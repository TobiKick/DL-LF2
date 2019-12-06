# Deep Learning Applications - Project "Little Fighter 2"

############################# IMPORT STATEMENTS ########################################################
import argparse
import os, sys

#Import lf2gym
import random
import lf2gym
from time import sleep

#Import Keras modules
from keras.layers import Dense, Flatten, Input, Conv2D, LSTM, concatenate, Concatenate, Reshape
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
EPISODES = 2  # SET TO 1000 to comparably calculate winning rate
SAVE_PROGRESS_TO_MODEL = True
HEADLESS = False

########################### FORCE KERAS TO USE CPU #####################################################
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

############################# SETUP THE DEEP Q AGENT ########################################################
# Deep Q-learning Agent

class DQNAgent:
    def __init__(self, state_size_x, state_size_y, action_size):
        self.state_size_x = state_size_x
        self.state_size_y = state_size_y
        self.action_size = action_size
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model_actor = self._build_model_actor()
        self.model_critic = self._build_model_critic()

    def shared_model(self, input, input_agent_info, input_action):
        model = Conv2D(32, kernel_size=(8,8), strides=4, padding="same", activation='relu')(input)
        model = Conv2D(64, kernel_size=(4,4), strides=2, padding="same", activation='relu')(model)
        model = Conv2D(64, kernel_size=(3,3), strides=1, padding="same", activation='relu')(model)
        model = Flatten()(model)

        merge = Concatenate()([model, input_agent_info, input_action])
        reshape = Reshape((61457, 1))(merge)
        output = LSTM(64, input_shape=(61457, 1))(reshape)
        return output

    def _build_model_actor(self):
        # Neural Net for the Actor of Actor-Critic-Model
        # defining inputs
        input = Input(shape=(state_size_x, state_size_y, 4))
        input_agent_info = Input(shape=(16,))
        input_action = Input(shape=(1,))

        # calling shared model (shared weights)
        output = self.shared_model(input, input_agent_info, input_action)

        # defining residual model + output
        output = Dense(32, activation='relu')(output)
        output = Dense(self.action_size, activation='softmax')(output)

        model = Model(inputs=[input, input_agent_info, input_action], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate))

        model.summary()
        return model

    def _build_model_critic(self):
        # Neural Net for the Critic of Actor-Critic-Model
        # defining inputs
        input = Input(shape=(state_size_x, state_size_y, 4))
        input_agent_info = Input(shape=(16,))
        input_action = Input(shape=(1,))

        # calling shared model (shared weights)
        output = self.shared_model(input, input_agent_info, input_action)

        # defining residual model + output
        output = Dense(64, activation='relu')(output)
        output = Dense(self.action_size, activation='relu')(output)

        model = Model(inputs=[input, input_agent_info, input_action], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        model.summary()
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model_actor.predict(state)
        return np.argmax(act_values[0])  # returns index of action with highges probability

    # A2C training after every action taken
    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model_critic.predict(next_state)[0])
        target_f = self.model_critic.predict(state)  # predicting probability for each action
        target_f[0][action] = target  # replacing the past action with target probability
        self.model_critic.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def saveModel(self):
        self.model_critic.save_weights("app_model/critic.h5")
        self.model_actor.save_weights("app_model/actor.h5")
        # serialize model to JSON
        model_json_critic = self.model_critic.to_json()
        with open("app_model/critic.json", "w") as json_file:
            json_file.write(model_json_critic)
        model_json_actor = self.model_actor.to_json()
        with open("app_model/actor.json", "w") as json_file:
            json_file.write(model_json_actor)
        with open("app_model/epsilon.txt", "w") as txt_file:
            txt_file.write(self.epsilon)
        print("Saved model to disk")

    def loadModel(self):
        self.model_critic.load_weights("app_model/critic.h5")
        self.model_actor.load_weights("app_model/actor.h5")
        with open("app_model/epsilon.txt", "r") as txt_file:
            self.epsilon = int(txt_file.readline())
        print("Loaded model from disk")

############################# DEEP Q LEARNING ########################################################

def getPlayerInformation(env_details):
    player1_info = env_details[0]
    player2_info = env_details[1]
    player_info = np.array((player1_info.get('hp'), player1_info.get('mp'), player1_info.get('x'), player1_info.get('y'),
                            player1_info.get('z'), player1_info.get('vx'), player1_info.get('vy'), player1_info.get('vz'),
                            player2_info.get('hp'), player2_info.get('mp'), player2_info.get('x'), player2_info.get('y'),
                            player2_info.get('z'), player2_info.get('vx'), player2_info.get('vy'), player2_info.get('vz')))
    return player_info


if __name__ == "__main__":
    # initialize gym environment and the agent
    env = lf2gym.make(startServer=True, driverType=lf2gym.WebDriver.Chrome,
                      background=lf2gym.Background.HK_Coliseum,
                      characters=[args.player, args.opponent], #[Me/AI, bot]
                      difficulty=lf2gym.Difficulty.Crusher,
                      action_options= ['Basic', 'AJD', 'Full Combos'],     # ['Basic', 'AJD', 'Full Combos']   # defines action space
                      headless=HEADLESS,
                      versusPlayer=False) # versusPlayer= False means Agent is playing against bot instedad of user

    state_size_x = env.observation_space.n[0]
    state_size_y = env.observation_space.n[1]
    print("state_size: (" + str(state_size_x) + ", " + str(state_size_y) + ")")
    action_size = int(env.action_space.n)
    print("action_size: " + str(action_size))

    agent = DQNAgent(state_size_x, state_size_y, action_size)
    if LOAD_PROGRESS_FROM_MODEL:
        agent.loadModel()

    done = False
    batch_size = 32
    wins = 0
    action_last_episode = 0
    # Iterate the game
    for e in range(EPISODES):
        # reset state in the beginning of each game
        state = env.reset()   # returns 243200 for our game as screen size input (160,380,4)

        player_info = getPlayerInformation(env.get_detail())
        _state = np.reshape(state, (1, state_size_x, state_size_y, 4))
        _player_info = np.reshape(player_info, (1, 16))
        _action_last_episode = np.reshape(action_last_episode, (1, 1))
        state = [_state, _player_info, _action_last_episode]
        for time_t in range(500):
            # env.render()  # no need to activate render
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10  # punishes the agent if the game takes too long

            if(env.get_detail() != None):
                player_info = getPlayerInformation(env.get_detail())
            _next_state = np.reshape(next_state, (1, state_size_x, state_size_y, 4))
            _player_info = np.reshape(player_info, (1, 16))
            _action = np.reshape(action, (1, 1))
            next_state = [_next_state, _player_info, _action]

            agent.train(state, action, reward, next_state, done)  # Remember the previous state, action, reward, and done
            state = next_state  # make next_state the new current state for the next frame.

            if done: # # done becomes True when the game ends
                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, time_t))
                break

        if env.get_detail()[0].get('hp') == 0:    # [0] is player (=agent), [1] is opponent (=bot)
            wins += 1
        # save progress to model after finishing the last episode
        if e == (EPISODES - 1):
            winningRate = wins/(e+1)
            print("# Wins: " + str(wins) + ", # Episodes: " + str(e+1) + ", Winning Rate: " + str(winningRate))
            print("# Current epsilon: " + str(agent.epsilon))
            agent.saveModel()
