import numpy as np
import random
import time
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras as k
from keras import models as km
from keras import layers as kl

from GeometryDashEnvironment import GDashEnv

class NeuralNetwork:
    
        #builds neural newtorks and intializes the neccesary paramters with the confusion on how we implement q-learning 
        # iwasnt fully sure on how to implement certain neccesary paramters for that btu they ar included here for conveniance in case
        # the code needs to be updated 
    def __init__(self, alpha=0.98, max_experience=10000, batch_size = 4, num_of_runs=1000,
                eps=0.5, eps_decay=0.999, gamma=0.98, lr=0.001, n=20, env=GDashEnv(),
                target_frames=500, eps_min = 0.1, rand_batch=False, double_q=False):
        self.GD = env
        self.action_space = self.GD.action_space
        self.num_actions = self.GD.action_space.n
        self.alpha = alpha
        self.frame_count = 0
        self.input_shape = (self.GD.observation_space.shape[1], self.GD.observation_space.shape[0], 1)
        self.experience_count = 0
        self.max_experience = max_experience
        self.double_q = double_q
        self.rand_batch = rand_batch
        if rand_batch:
            self.observation_exp = np.zeros((max_experience, self.input_shape[0],self.input_shape[1],self.input_shape[2]))
            self.new_obs_exp = np.zeros((max_experience, self.input_shape[0],self.input_shape[1],self.input_shape[2]))
            self.action_exp = np.zeros((max_experience, self.num_actions),dtype=np.int8)
            self.reward_exp = np.zeros(max_experience)
            self.terminated_exp = np.zeros(max_experience)
        else:
            self.observation_exp = np.zeros((n, self.input_shape[0],self.input_shape[1],self.input_shape[2]))
            self.new_obs_exp = np.zeros((n, self.input_shape[0],self.input_shape[1],self.input_shape[2]))
            self.action_exp = np.zeros((n, self.num_actions),dtype=np.int8)
            self.reward_exp = np.zeros(n)
            self.terminated_exp = np.zeros(n)
        self.batch_size = batch_size
        self.num_of_runs = num_of_runs
        self.eps = eps
        self.exp_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.lr = lr 
        self.target_frames = target_frames
        self.n = n
        self.neural = self.build_model()
        if double_q:
            self.neural_target = self.build_model()

    def build_model(self):
        input_shape = self.input_shape
        model = km.Sequential([
            kl.Input(shape=input_shape),
            kl.Conv2D(32, 8, strides=4, activation="relu"),
            # self.neural.add(kl.MaxPooling2D((2, 2), padding='same'))
            kl.Conv2D(64, 4, strides=2, activation="relu"),
            # self.neural.add(kl.MaxPooling2D((2, 2), padding='same'))
            kl.Conv2D(64, 3, strides=1, activation="relu"),
            kl.Flatten(),
            kl.Dense(512, activation="relu"),
            #model.add(kl.Dense(256, input_shape=input_shape, activation="relu"))
            #model.add(kl.Dense(256, activation="relu"))
            kl.Dense(self.num_actions, activation='linear')  # Assuming action_space.n is the number of possible actions
        ])
        model.compile(optimizer=k.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0), loss='mse')
        return model
    
    def save_experience(self, observation, action, reward, new_obs, terminated):
        indx = self.experience_count % self.max_experience if self.rand_batch else self.experience_count % self.n
        self.observation_exp[indx] = observation
        actions = np.zeros(self.action_exp.shape[1])
        actions[action] = 1.0
        self.action_exp[indx] = actions
        self.reward_exp[indx] = reward
        self.new_obs_exp[indx] = new_obs
        self.terminated_exp[indx] = 1 - int(terminated)
        self.experience_count += 1
    
    def sample_experiences(self):
        max_exp = min(self.experience_count, self.max_experience)
        indxs = np.random.choice(max_exp, self.batch_size)
        
        observations = self.observation_exp[indxs]
        actions = self.action_exp[indxs]
        rewards = self.reward_exp[indxs]
        new_obs = self.new_obs_exp[indxs]
        terminated = self.terminated_exp[indxs]
        
        return observations, actions, rewards, new_obs, terminated
    
    def sample_n_step(self):
        observations = self.observation_exp
        actions = self.action_exp
        rewards = self.reward_exp
        new_obs = self.new_obs_exp
        terminated = self.terminated_exp
        
        return observations, actions, rewards, new_obs, terminated

    # Predicting the action alternates between random and prediction based on exploration rate
    def predict_action(self, observation):
        if random.random() < self.eps:
            return self.GD.action_space.sample()  # Choosing a random action
        else:
            obs = np.expand_dims(observation, axis=0)
            return np.argmax(self.neural.predict(obs, verbose=0))  # Choosing the best action based on Q-values

    # Updates the fit of the network based upon the new observation
    def update_network(self):
        if self.rand_batch: # Random batch implementation
            if self.experience_count < self.batch_size:
                return
            observations, actions, rewards, new_obs_exp, terminated_exp = self.sample_experiences()
        
            action_opts = np.array(np.array([0,1]), dtype=np.int8)
            action_indxs = np.dot(actions, action_opts)
            
            q_table = self.neural.predict(observations, verbose=0)
            
            new_q_target = self.neural_target.predict(new_obs_exp,verbose=0) if self.double_q else None
           
            new_q = self.neural.predict(new_obs_exp, verbose=0)
            
            max_actions = np.argmax(q_table, axis=1) if self.double_q else None
            
            q_target = q_table if self.double_q else q_table.copy()
            
            indx = np.arange(self.batch_size, dtype=np.int32)
            if self.double_q:
                q_target[indx, action_indxs] = rewards + self.gamma * new_q_target[indx, max_actions.astype(int)] * terminated_exp
            else:
                q_target[indx, action_indxs] = rewards + (self.gamma * np.max(new_q, axis=1)) * terminated_exp
            
            self.neural.fit(observations, q_target, verbose=0)
        else: # N-step implementation
            if self.experience_count < self.n:
                return
            
            observations, actions, rewards, new_obs_exp, terminated_exp = self.sample_n_step()
            
            indx = self.experience_count % self.n + 1
            total_reward = 0
            discount = 1
            for i in range(self.n):
                if indx >= self.n:
                    indx = 0
                total_reward += rewards[indx] * discount
                discount *= self.gamma
                if terminated_exp[indx] == 0:
                    break
                indx += 1
            
            indx = self.experience_count % self.n + 1
            if indx >= self.n:
                    indx = 0
            
            if terminated_exp[indx] == 1:
                if self.double_q:
                    new_obs = np.expand_dims(new_obs_exp[indx], axis=0)
                    total_reward += discount * np.max(self.neural_target.predict(new_obs,verbose=0)[0])
                else:
                    new_obs = np.expand_dims(new_obs_exp[indx], axis=0)
                    total_reward += discount * np.max(self.neural.predict(new_obs,verbose=0)[0])
            
            action_indxs = np.argmax(actions, axis=1)
            
            obs = np.expand_dims(observations[indx], axis=0)
            q_value = self.neural.predict(obs, verbose=0)
            
            q_value[0,action_indxs[indx]] = total_reward
            
            self.neural.fit(obs, q_value, verbose=0)
                
                
                
                
            

    # Calculates FPS
    def train_model_FPS(self, num_iterations=50, model_file_path=".venv\\Models\\",
                    model_name="FPS.keras", target_name="FPSTarget.keras", render=False,
                    log_file_path=".venv\\TrainingLogs\\FPS.txt", explore_frames=100):
        training_results = np.empty((num_iterations,5)) 
        absolute_max_progress = 0.0
        game = self.GD
        frame_processor = game.frame_processor
        fps = 0
        time_start = time.time()
        i = 0
        while time.time() - time_start < 30:
            if self.double_q and self.frame_count % self.target_frames == 0 and i > 0:
                    self.update_target()
            log = open(log_file_path,"a")
            observation = self.GD.reset()
            observation = frame_processor.prepare_for_nn(observation)  # Use centralized preprocessing
            terminated = False
            max_progress = 0.0
            reward = 0.0
            start_frame = self.frame_count
            while not terminated and time.time() - time_start < 30:
                action = self.predict_action(observation)
                new_obs, reward, terminated, progress = game.step(action)
                
                # Used to get how much progress was achieved in that iteration
                if progress > 0.0:
                    max_progress = progress
               
                new_obs = frame_processor.prepare_for_nn(new_obs)  # Ensure observation is processed for the next cycle
                self.save_experience(observation, action,  reward, new_obs ,terminated)
                self.update_network()
                
                observation = new_obs
                
                self.frame_count += 1
                fps += 1
                if render:
                    game.render()
                
            end_frame = self.frame_count
            
            # Decaying Epsilon
            if self.eps > self.eps_min and self.frame_count > explore_frames:
                self.eps *= self.exp_decay
            
            # Tracking Progress
            training_results[i,0] = max_progress
            training_results[i,1] = reward
            training_results[i,2] = game.cur_jumps
            training_results[i,3] = self.eps
            training_results[i,4] = end_frame - start_frame
            
            if max_progress > absolute_max_progress:
                absolute_max_progress = max_progress
            
            print("Run #",i+1," Reward: ",reward," Progress: %",max_progress,
                " Frame count: ",self.frame_count)
            print("Run #",i+1," Reward: ",reward," Progress: %",max_progress,
                " Frame count: ",self.frame_count,"\n",file=log)
            
            # Logging Averaged every 50 runs
            if (i+1) % 50 == 0:
                run_start = i-48
                run_end = i+1
                avg_prog = np.mean(training_results[(i-49):i,0])
                avg_reward = np.mean(training_results[(i-49):i,1])
                avg_jumps = np.mean(training_results[(i-49):i,2])
                avg_jumps_per_prog = np.round(avg_jumps/avg_prog,0)
                avg_epsilon = np.mean(training_results[(i-49):i,3])
                avg_frames = np.mean(training_results[(i-49):i,4])
                
                print("\nRun:",run_start,"-",run_end," Average Progress:",avg_prog,
                      " Average Reward:",avg_reward," Average Total Jumps:",
                      avg_jumps," Average Jumps Per Progress %:",avg_jumps_per_prog," Average Epsilon:",
                      avg_epsilon, " Average Number of Frames:",avg_frames,"\n",file=log)
                print("\nRun:",run_start,"-",run_end," Average Progress:",avg_prog,
                      " Average Reward:",avg_reward," Average Total Jumps:",
                      avg_jumps," Average Jumps Per Progress %:",avg_jumps_per_prog," Average Epsilon:",
                      avg_epsilon, " Average Number of Frames:",avg_frames,"\n")
            
            # Saving Weights every 500 runs
            if i % 500 == 0 and i > 0: 
                self.neural.save(model_file_path+model_name)
                self.neural_target.save(model_file_path+target_name)
            log.close()
            i += 1
        self.frame_count = 0
        print("\nFPS: ",np.round((fps/30),0))
    
    # Trains Model
    def train_model(self, num_iterations, load=False, model_file_path=".venv\\Models\\",
                    model_name="Model.keras", target_name="Target.keras", log_file_path=None,
                    notes="Notes", explore_frames=5000):
        self.neural = self.build_model()
        if self.double_q:
            self.neural_target = self.build_model()
        if load:
            self.neural = km.load_model(model_file_path+model_name)
            if self.double_q:
                self.neural_target = km.load_model(model_file_path+target_name)
        self.neural.save(model_file_path+model_name)
        if self.double_q:
            self.neural_target.save(model_file_path+target_name)
        log = open(log_file_path,"a")
        print("\nModel: ",model_name,"\n",file=log)
        print("\n",notes,"\n",file=log)
        print("Iterations: ",num_iterations," Gamma: ",self.gamma," Epsilon: ",self.eps,
              "\nEpsilon Decay: ",self.exp_decay," Learning Rate: ",self.lr,
              " Batch Size: ",self.batch_size,"\n",file=log)
        training_results = np.empty((num_iterations,6)) 
        absolute_max_progress = 0.0
        game = self.GD
        print("Living Factor: ",game.survival_reward," Death Penalty: ",game.death_penalty,
              " Jump Penalty: ",game.jump_penalty,"\n",file=log)
        print("Input Shape: ",self.input_shape,"\n",file=log)
        print("\n---------- Beginning Training ----------\n",file=log)
        log.close()
        frame_processor = game.frame_processor
        for i in range(num_iterations):
            log = open(log_file_path,"a")
            observation = self.GD.reset()
            observation = frame_processor.prepare_for_nn(observation)  # Use centralized preprocessing
            terminated = False
            max_progress = 0.0
            reward = 0.0
            start_frame = self.frame_count
            time_start = time.time()
            while not terminated:
                if self.double_q and self.frame_count % self.target_frames == 0 and i > 0:
                    self.update_target()
                action = self.predict_action(observation)
                new_obs, reward, terminated, progress = game.step(action)
                
                # Used to get how much progress was achieved in that iteration
                if progress > 0.0:
                    max_progress = progress
               
                new_obs = frame_processor.prepare_for_nn(new_obs)  # Ensure observation is processed for the next cycle
                self.save_experience(observation, action, reward, new_obs ,terminated)
                self.update_network()
                
                observation = new_obs
                
                self.frame_count += 1
                # game.render()
            
            run_time = time.time() - time_start
            end_frame = self.frame_count
            
            # Decaying Epsilon
            if self.eps > self.eps_min and self.frame_count > explore_frames:
                self.eps *= self.exp_decay
            
            # Tracking Progress
            training_results[i,0] = max_progress
            training_results[i,1] = game.total_reward
            training_results[i,2] = game.cur_jumps
            training_results[i,3] = self.eps
            training_results[i,4] = end_frame - start_frame
            training_results[i,5] = run_time
            
            if max_progress > absolute_max_progress:
                absolute_max_progress = max_progress
            
            print("Run #",i+1," Reward: ",game.total_reward," Progress: %",max_progress,
                " Frame count: ",self.frame_count)
            print("Run #",i+1," Reward: ",game.total_reward," Progress: %",max_progress,
                " Frame count: ",self.frame_count,"\n",file=log) 
            
            # Logging Averaged every 50 runs
            if (i+1) % 50 == 0:
                run_start = i-48
                run_end = i+1
                avg_prog = np.mean(training_results[(i-49):i,0])
                avg_reward = np.mean(training_results[(i-49):i,1])
                avg_jumps = np.mean(training_results[(i-49):i,2])
                avg_jumps_per_prog = np.round(avg_jumps/avg_prog,0)
                avg_epsilon = np.mean(training_results[(i-49):i,3])
                avg_frames = np.mean(training_results[(i-49):i,4])
                avg_run_time = np.mean(training_results[(i-49):i,5])
                avg_fps = avg_frames / avg_run_time
                
                print("\nRun:",run_start,"-",run_end," Average Progress:",avg_prog,
                      " Average Total Reward:",avg_reward,"\nAverage Total Jumps:",
                      avg_jumps," Average Jumps Per Progress %:",avg_jumps_per_prog," Average Epsilon:",
                      avg_epsilon,"\nAverage Number of Frames:",avg_frames," Average Run Time:",avg_run_time,
                      "Average FPS:",avg_fps,"\n",file=log)
                print("\nRun:",run_start,"-",run_end," Average Progress:",avg_prog,
                      " Average Total Reward:",avg_reward,"\nAverage Total Jumps:",
                      avg_jumps," Average Jumps Per Progress %:",avg_jumps_per_prog," Average Epsilon:",
                      avg_epsilon,"\nAverage Number of Frames:",avg_frames," Average Run Time:",avg_run_time,
                      "Average FPS:",avg_fps,"\n")
            
            # Saving Weights every 500 runs
            if i % 50 == 0 and i > 0: 
                self.neural.save(model_file_path+model_name)
                if self.double_q:
                    self.neural_target.save(model_file_path+target_name)
            log.close()
            
        log = open(log_file_path,"a")
        print("\n---------- End Training ----------\n\n",file=log)
        
        self.neural.save(model_file_path+model_name)
        if self.double_q:
            self.neural_target.save(model_file_path+target_name)
        
        for i in range(num_iterations):
            if (i+1) % 50 == 0:
                run_start = i-48
                run_end = i+1
                avg_prog = np.mean(training_results[(i-49):i,0])
                avg_reward = np.mean(training_results[(i-49):i,1])
                avg_jumps = np.mean(training_results[(i-49):i,2])
                avg_jumps_per_prog = np.round(avg_jumps/avg_prog,0)
                avg_epsilon = np.mean(training_results[(i-49):i,3])
                avg_frames = np.mean(training_results[(i-49):i,4])
                avg_run_time = np.mean(training_results[(i-49):i,5])
                avg_fps = avg_frames / avg_run_time
                
                print("\nRun:",run_start,"-",run_end," Average Progress:",avg_prog,
                      " Average Total Reward:",avg_reward,"\nAverage Total Jumps:",
                      avg_jumps," Average Jumps Per Progress %:",avg_jumps_per_prog," Average Epsilon:",
                      avg_epsilon,"\nAverage Number of Frames:",avg_frames," Average Run Time:",avg_run_time,
                      "Average FPS:",avg_fps,file=log)
                print("\nRun:",run_start,"-",run_end," Average Progress:",avg_prog,
                      " Average Total Reward:",avg_reward,"\nAverage Total Jumps:",
                      avg_jumps," Average Jumps Per Progress %:",avg_jumps_per_prog," Average Epsilon:",
                      avg_epsilon,"\nAverage Number of Frames:",avg_frames," Average Run Time:",avg_run_time,
                      "Average FPS:",avg_fps)
            
        print("\nAverage first 10 iterations: ", np.mean(training_results[:10,0]),"\n",
              "Average first 100 iterations: ", np.mean(training_results[:10,0]),"\n\n",
              "Average last 10 iterations: ", np.mean(training_results[-10:,0]),"\n",
              "Average last 100 iterations: ", np.mean(training_results[-100:,0]),"\n\n",
              "Absolute max progress: ", absolute_max_progress,"\n",
              "Current Exploration Rate: ", self.eps,"\n",file=log)
        print("\nAverage first 10 iterations: ", np.mean(training_results[:10,0]),"\n",
              "Average first 100 iterations: ", np.mean(training_results[:100,0]),"\n\n",
              "Average last 10 iterations: ", np.mean(training_results[-10:,0]),"\n",
              "Average last 100 iterations: ", np.mean(training_results[-100:,0]),"\n\n",
              "Absolute max progress: ", absolute_max_progress,"\n",
              "Current Exploration Rate: ", self.eps,"\n")
        log.close()
    
    # Updates neural_target weights from neural weights
    def update_target(self):
        self.neural_target.set_weights(self.neural.get_weights())
        




