import numpy as np
import random
import keras as k
import keras.models as km
import keras.layers as kl

import GeometryDashEnvironment as GDenv

class NeuralNetwork:
    
        #builds neural newtorks and intializes the neccesary paramters with the confusion on how we implement q-learning 
        # iwasnt fully sure on how to implement certain neccesary paramters for that btu they ar included here for conveniance in case
        # the code needs to be updated 
    def __init__(self, alpha=0.98, num_of_runs=1000, exp=0.3, gamma=0.99, env=GDenv()):
        self.GD = env
        self.alpha = alpha
        self.num_of_runs = num_of_runs
        self.exp = exp
        self.gamma = gamma 
        self.neural = km.Sequential()
        self.neural.add(kl.Input(shape=(1,self.GD.observation_space.n)))
        self.neural.add(kl.Conv2D(2, 512, strides=1, activation="relu"))
        self.neural.add(kl.Conv2D(512, 128, strides=1, activation="relu"))
        self.neural.add(kl.Conv2D(128, 64, strides=1, activation="relu"))
        self.neural.add(kl.Conv2D(64, 32, strides=1, activation="relu"))
        self.neural.add(kl.Conv2D(32, 4, strides=1, activation="relu"))
        self.neural.flatten()
        self.neural.add(kl.Dense(64, activation="relu"))
        self.neural.add(kl.Dense(self.GD.action_space.n, activation='linear'))  # Output layer for Q-values
        self.neural.compile(optimizer='adam', loss='mse')  # Using Mean Squared Error for Q-learning




    #this is for a possible model based on q-learning as possibly another way to predict the action
    # also seen in the keras documentation in this case they don't compile the net and it uses q learning and a q table
    # essentially making it a glorified q table because no predicitons are made this code is just here for possible testing
    #def predict_action(self, observation):
     #   action_choice = [True, False]
      #  choice = random.choices(action_choice, weights = [self.exp, 1 - self.exp], k = 1)
      #  if (choice[0] == True):
       #     return random.choice([0,1])
       # else:
       #     ten_obs = k.ops.convert_to_tensor(observation)
        #    return k.ops.max(self.neural(ten_obs,Training=False)[0])

    '''   
    #predicitng the action alternates between random and prediction 
    def predict_action(self, observation):
        action_choice = [True, False]
        choice = random.choices(action_choice, weights = [self.exp, 1 - self.exp], k = 1)
        if (choice[0] == True):
            return random.choice([0,1])
        else:
            #Also unsure of what I need tp use in predict some things in the keras documentation 
            return k.ops.max(self.neural.predict(observation))

        
    #def update_network(self, new_obs, reward, terminated):
    #    ten_obs = k.ops.convert_to_tensor(new_obs)
     #   x = k.ops.max(self.neural(ten_obs,Training=False)[0])
     #   Goal = reward+self.gamma*possible_qtablefunction()
    # self.neural.set_weights()
      #  self.neural.fit()
        '''
    
        # Predicting the action alternates between random and prediction based on exploration rate
    def predict_action(self, observation):
        if random.random() < self.exp:
            return random.choice([0, 1])  # Choosing a random action
        else:
            return np.argmax(self.neural.predict(observation))  # Choosing the best action based on Q-values

    # Updates the fit of the network based upon the new observation
    def update_network(self, new_obs, reward, observation, terminated):
        current_q = self.neural.predict(observation)
        new_q = self.neural.predict(new_obs)
        max_new_q = np.max(new_q)
        
        # Q-learning update rule
        if terminated:
            current_q[0, np.argmax(current_q)] = reward
        else:
            current_q[0, np.argmax(current_q)] = reward + self.gamma * max_new_q
        
        # Fit the model
        self.neural.fit(observation, current_q, verbose=0)
    



