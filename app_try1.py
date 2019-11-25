# Deep Learning Applications - Project "Little Fighter 2"

############################# IMPORT STATEMENTS ########################################################
import argparse
import os, sys

#Import lf2gym
import lf2gym
from time import sleep

#Import Keras modules
from keras.layers import Dense, Flatten, Input
from keras import Model
from rl.agents.dqn import DQNAgent
from keras.optimizers import Adam
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

############################# SETUP LF2 ENVIRONMENT ########################################################
parser = argparse.ArgumentParser()
parser.add_argument("--player", default="Bandit", help="your character")
parser.add_argument("--opponent", default="Bandit", help="your enemy")
parser.add_argument("--interval", default=0.2, help="your enemy")
args = parser.parse_args()

# Add import path and import the lf2gym
sys.path.append(os.path.abspath('..'))

# Make an env
env = lf2gym.make(startServer=True, driverType=lf2gym.WebDriver.Chrome, 
    characters=[lf2gym.Character[args.opponent], lf2gym.Character[args.player]], versusPlayer=True)
# Reset the env
env.reset()

########################### FORCE KERAS TO USE CPU #####################################################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

############################# SETUP KERAS MODEL ########################################################
def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    return model

model = build_model(21, 21)   # number of actions space is 0 to 21 ; number state_size is  ????
memory = SequentialMemory(limit=50000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)

dqn = DQNAgent(model=model, nb_actions=1000, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
# what should nb_actions = have as input ???

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

def build_callbacks(env_name):
    checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks
    
callbacks = build_callbacks("ENV_NAME")

dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, callbacks=callbacks)


############################# LET THE GAME BEGIN ########################################################
# Game starts

##done = False
##while not done:
    # Sample a random action
##    action = env.action_space.sample()
    # Take the chosen action
##    _, reward, done, _ = env.step(action)
##    print('Enemy took action %d.' % action)
##    sleep(args.interval)
# Game ends
##print('Gameover!')