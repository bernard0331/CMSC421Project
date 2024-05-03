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
        self.neural.add(kl.Dense(512), activation="relu")
        self.neural.add(kl.Dense(2))
        self.neural.compile(loss='mse')



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
        
    #predicitng the action alternates between random and prediction 
    def predict_action(self, observation):
        action_choice = [True, False]
        choice = random.choices(action_choice, weights = [self.exp, 1 - self.exp], k = 1)
        if (choice[0] == True):
            return random.choice([0,1])
        else:
            #Also unsure of what I need tp use in predict some things in the keras documentation i've found suggest 
            # using np.identitty like I try to do in the update comment but I've found conflicting info on that which is why it's commented out
            # as well as in the comment below below 
            # k.ops.max(self.neural.predict(np.identity(self.GD.observation_space.n)[new_obs:new_obs + 1]))
            return k.ops.max(self.neural.predict(observation))
        
    #def update_network(self, new_obs, reward, terminated):
    #    ten_obs = k.ops.convert_to_tensor(new_obs)
     #   x = k.ops.max(self.neural(ten_obs,Training=False)[0])
     #   Goal = reward+self.gamma*possible_qtablefunction()
    # self.neural.set_weights()
      #  self.neural.fit()

    #updates the fit of the nwtork based upon the new observation
    def update_network(self, new_obs, reward, observation, terminated):
        ten_obs = k.ops.convert_to_tensor(new_obs)
        x = k.ops.max(self.neural(ten_obs,Training=False)[0])
        Goal = reward+self.gamma*k.ops.max(self.neural.predict(new_obs))
        #self.neural.fit(np.identity(self.GD.observation_space.n)[new_obs:new_obs + 1], [Goal].reshape(-1, self.GD.action_space.n))
        self.neural.fit(new_obs, [Goal].reshape(-1, self.GD.action_space.n))

    
    


