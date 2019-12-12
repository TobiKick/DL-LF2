# Deep Learning Applications - Project "Little Fighter 2"

############################# IMPORT STATEMENTS ########################################################
import argparse
import os, sys

#Import lf2gym
import random
import lf2gym

#Import Keras modules
from keras.layers import Dense, Flatten, Input, Conv2D, LSTM, Concatenate, Reshape, MaxPool2D
from keras import Model
import numpy as np
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import to_categorical

############################# SETUP LF2 PARAMETERS ########################################################
parser = argparse.ArgumentParser()
parser.add_argument("--player", default=lf2gym.Character.Louis, help="your character") # the AI
parser.add_argument("--opponent", default=lf2gym.Character.Freeze, help="your enemy") # the AI's enemy
parser.add_argument("--interval", default=0.9, help="your enemy")
args = parser.parse_args()

# Add import path for the lf2gym
sys.path.append(os.path.abspath('..'))


EPISODES = 50
TIME_MAX = 500
LOAD_PROGRESS_FROM_MODEL = True
SAVE_PROGRESS_TO_MODEL = True
HEADLESS = True
TRAINING = True

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
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.totalEpisodes = 0
        self.model = self._build_model()

    def logloss(self, y_true, y_pred):     #policy loss
        return -K.sum( K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + 1e-5), axis=-1)

    def _build_model(self):
        input = Input(shape=(state_size_x, state_size_y, 4))

        model = Conv2D(32, kernel_size=(8,8), strides=4, activation='relu')(input)
        model = MaxPool2D((2,2), padding='same')(model)
        model = Conv2D(64, kernel_size=(4,4), strides=2, activation='relu')(model)
        model = MaxPool2D((2,2), padding='same')(model)
        model = Conv2D(64, kernel_size=(3,3), strides=1, activation='relu')(model)
        model = MaxPool2D((2,2), padding='same')(model)
        model = Flatten()(model)


        input_agent_info = Input(shape=(16,))
        input_action = Input(shape=(1,))
        merge = Concatenate()([model, input_agent_info, input_action])
        reshape = Reshape((657, 1))(merge)
        output = LSTM(64, input_shape=(657, 1))(reshape)

        #output = Dense(32, activation='relu')(output)
        actions_out = Dense(self.action_size,  name = 'o_Policy', activation='softmax')(output)
        value_out = Dense(1,  name = 'o_Value', activation='linear')(output)

        model = Model(inputs=[input, input_agent_info, input_action], outputs=[actions_out, value_out])
        model.compile(loss = {'o_Policy': self.logloss, 'o_Value': 'mse'},  loss_weights = {'o_Policy': 1., 'o_Value' : 0.5}, optimizer=Adam(lr=self.learning_rate))

        model.summary()
        return model


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)[0]
        return np.argmax(act_values[0])  # returns index of action with highest probability

    def discount(self, r):
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    # not yet using oneHot encoded actions ( -> because of implications on the act() function
    def train(self, states, agent_info, actions, actions_oneHot, rewards):
        discounted_rewards = self.discount(rewards)
        state_values = self.model.predict([states, agent_info, actions])
        state_values = state_values[1]
        advantages = discounted_rewards - np.reshape(state_values[0], len(state_values[0]))
        weights = {'o_Policy': advantages, 'o_Value': np.ones(len(advantages))}
        self.model.fit([states, agent_info, actions], [actions, discounted_rewards], epochs=1, sample_weight=weights, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def saveModel(self, wins, e):
        self.model.save_weights("app_model/model_50.h5")
        with open("app_model/stats_50.txt", "w", newline="\n", encoding="utf-8") as txt_file:
            txt_file.writelines([str(self.epsilon), "\n" + str(self.totalEpisodes) + "\n" + str(wins), "\n" + str(e)])
        print("Saved model to disk")

    def loadModel(self):
        self.model.load_weights("app_model/model_50.h5")
        with open("app_model/stats_50.txt", "r") as txt_file:
            self.epsilon = float(txt_file.readline())
            self.totalEpisodes = int(txt_file.readline())
        print("Epsilon: " + str(self.epsilon))
        print("TotalEpisodes: " + str(self.totalEpisodes))
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
                      rewardList= ['hp'],
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

        state_buffer, agent_info_buffer, action_buffer, action_oneHot_buffer, reward_buffer = [], [], [], [], []

        player_info = getPlayerInformation(env.get_detail())
        _state = np.reshape(state, (1, state_size_x, state_size_y, 4))
        _player_info = np.reshape(player_info, (1, 16))
        _action_last_episode = np.reshape(action_last_episode, (1, 1))
        state = [_state, _player_info, _action_last_episode]

        for time_t in range(TIME_MAX):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            if time_t == (TIME_MAX -1):
                reward = -10
            if(env.get_detail() != None):
                player_info = getPlayerInformation(env.get_detail())
                if done == False and env.get_detail()[0].get('hp') == 0:
                    reward = -10
            _next_state = np.reshape(next_state, (1, state_size_x, state_size_y, 4))
            _player_info = np.reshape(player_info, (1, 16))
            _action = np.reshape(action, (1, 1))
            __next_state = [_next_state, _player_info, _action]

            state_buffer.append(np.reshape(next_state, (state_size_x, state_size_y, 4)))
            agent_info_buffer.append(player_info)
            action_oneHot_buffer.append(to_categorical(action, agent.action_size)) #one hot encoding the actions
            action_buffer.append(action)
            reward_buffer.append(reward)
            state = __next_state  # make next_state the new current state for the next frame.

            if done: # # done becomes True when the game ends
                print("episode: {}/{}, score: {}"
                      .format(e+1, EPISODES, time_t))
                break

        if TRAINING == True:
            agent.train(state_buffer, agent_info_buffer, action_buffer, action_oneHot_buffer, reward_buffer)

        print("Epsiode: " + str(e+1))
        if env.get_detail() != None:
            if env.get_detail()[0].get('hp') == 0:    # [0] is player (=agent), [1] is opponent (=bot)
                wins += 1
                print("Agent won!")

        # save progress to model after finishing the last episode
        if e == (EPISODES - 1):
            agent.totalEpisodes += (e+1)
            winningRate = wins/(e+1)
            print("# Wins: " + str(wins) + ", # Episodes: " + str(e+1) + ", Winning Rate: " + str(winningRate))
            print("# Current epsilon: " + str(agent.epsilon))
            agent.saveModel(wins, e+1)
