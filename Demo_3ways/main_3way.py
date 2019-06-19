# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

# import routes_generation_training

import os
import sys
import random
import numpy as np
import math
import time
import pickle
import constants


import timeit
import matplotlib.pyplot as plt
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to kill warning about tensorflow

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

PHASE_RIGHT_GREEN = 0 
PHASE_RIGHT_YELLOW = 1
PHASE_TOPBOT_GREEN = 2 
PHASE_TOPBOT_YELLOW = 3

from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Average, Add, Dot, Subtract, Multiply
from keras.models import Model
from keras.optimizers import Adam
import genorator

class DQNAgent:
	# initialize agent
	def __init__(self, num_actions, batch_size):
		
		# DA implement:
		self.Beta = 0.01        			# Leaky ReLU
		self.learning_rate = 0.0001 		# learning rate
		
		# # best model ========================================================================
		self._num_actions = num_actions
		self._batch_size = batch_size
		self._num_states = 80						# hard code

		# now setup the model
		self.model = self._define_model()

	# define architect of model
	def _define_model(self):
		input_1 = Input(shape=(60, 60, 1))
		input_2 = Input(shape=(self._num_actions, self._num_actions))

		x1 = Conv2D(32, (4, 4), strides=(2, 2),padding='Same', activation=LeakyReLU(alpha=self.Beta))(input_1)
		x1 = Conv2D(64, (2, 2), strides=(2, 2),padding='Same', activation=LeakyReLU(alpha=self.Beta))(x1)
		x1 = Conv2D(128, (2, 2), strides=(1, 1),padding='Same', activation=LeakyReLU(alpha=self.Beta))(x1)
		x1 = Flatten()(x1)
		x1 = Dense(128, activation=LeakyReLU(alpha=self.Beta))(x1)				# fully connected.
		x1_value = Dense(64, activation=LeakyReLU(alpha=self.Beta))(x1)			# 64
		value = Dense(1, activation=LeakyReLU(alpha=self.Beta))(x1_value)

		x1_advantage = Dense(64, activation=LeakyReLU(alpha=self.Beta))(x1)
		advantage = Dense(self._num_actions, activation=LeakyReLU(alpha=self.Beta))(x1_advantage)

		A = Dot(axes=1)([input_2, advantage])
		A_subtract = Subtract()([advantage, A])
		Q_value = Add()([value, A_subtract])

		model = Model(inputs=[input_1, input_2], outputs=[Q_value])
		model.compile(optimizer= Adam(lr=self.learning_rate), loss='mse')
		return model

	def save(self, name):
		self.model.save_weights(name)
    
	def load(self, name):
		self.model.load_weights(name)


	# just test
	@property
	def num_states(self):
		return self._num_states

	@property
	def num_actions(self):
		return self._num_actions

	@property
	def batch_size(self):
		return self._batch_size

# HANDLE THE SIMULATION OF THE AGENT
class SumoSimulation:
    def __init__(self, agent, target_agent, memory, gamma, max_steps, sumoCmd, green_duration, yellow_duration):
        
        self.I = np.full((agent.num_actions, agent.num_actions), 0.5).reshape(1, agent.num_actions, agent.num_actions)
        self._green_duration = green_duration					# hard code.
        self._yellow_duration = yellow_duration
        self._alpha_update_target = 0.0001
        # ---------------------------------------------------------------------------------------
        self._agent = agent
        self._target_agent = target_agent
        self._memory = memory
        self._epsilon = 0 							# controls the explorative/exploitative payoff
        self._gamma = gamma
        self._max_steps = max_steps
        self._sumoCmd = sumoCmd

        self._reward_store_LOW = []
        self._cumulative_wait_store_LOW = []
        self._avg_intersection_queue_store_LOW = []

        self._reward_store_HIGH = []
        self._cumulative_wait_store_HIGH = []
        self._avg_intersection_queue_store_HIGH = []

        self._reward_store_NS = []
        self._cumulative_wait_store_NS = []
        self._avg_intersection_queue_store_NS = []

        self._reward_store_EW = []
        self._cumulative_wait_store_EW = []
        self._avg_intersection_queue_store_EW = []

    def get_simu_type_str(self,simu_type):
        if simu_type == 0:
            return 'LOW   '
        elif simu_type == 1:
            return 'HIGH  ' 
        elif simu_type == 2:
            return 'NS    ' 
        elif simu_type == 3:
            return 'EW    ' 

    # Run 1 simulation:
    def run_one_episode(self, episode, total_episodes):
        
        simu_type = genorator.gen_route(episode)
        print('Mode: ', self.get_simu_type_str(simu_type))
        self._epsilon = 1.0 - (episode / total_episodes)								# setup epsilon
        traci.start(self._sumoCmd)														# then, start sumo
        
        # INIT some vars:
        self._steps = 0			
        self._sum_intersection_queue = 0		# increases every step/seconds
        tot_neg_reward = 0						# total negative reward
        pre_wait_time = 0						# 

        # INIT my vars:
        action = 0						# initial action
        old_action = 0
        state = self._get_state(self.I)
        action_count = [0,0]        # cal percent of actions
        good_bad_count = [0,0]      #   count good bad actions

        # run 1 simulation (maxsteps)
        while self._steps < self._max_steps:
            # reset current_wait_time:
            current_wait_time = 0

            # select action (select index of action, then edit action_time)
            action = self._choose_action(state)
            # just count numb of taken actions.
            action_count[action] += 1

            #  ================================ Take action ====================================================================
            if self._steps != 0 and old_action != action:
                # just set traffic_light in sumo
                self._set_yellow_phase(old_action)
                current_wait_time = self._simulate(self._yellow_duration)           # what for?
            self._set_green_phase(action)
            current_wait_time = self._simulate(self._green_duration)
            #  =================================================================================================================

            # get next_state and reward
            next_state = self._get_state(self.I)

            reward = pre_wait_time - current_wait_time
            if reward < 0:
                tot_neg_reward += reward
                good_bad_count[1] += 1
            else:
                good_bad_count[0] += 1
            
            # save tuple:			
            self._memory.add_sample((state, action, reward, next_state))
            
            # training:
            self._replay()		

            # reassign:
            pre_wait_time = current_wait_time
            state = next_state
            old_action = action

            # print
            eval_this_action = 'Good action' if (reward>=0) else 'Bad Action'
            print('step: ', self._steps, '/',self._max_steps,' || action: ', self.get_action_name(action), ': ',eval_this_action,' || negative reward: ', tot_neg_reward)

        print('percent of actions: ', np.array(action_count)/sum(action_count))
        print('good actions: ', good_bad_count[0])  
        print('bad actions:  ', good_bad_count[1])
        print("Total negative reward: {}, Eps: {}".format(tot_neg_reward, self._epsilon))
        self._save_stats(simu_type,tot_neg_reward)		

        # close gui.
        traci.close(wait = False)

    #  FOR TEST
    def get_action_name(self,action):
        if action == 0:
            return 'EW     '
        elif action == 1:
            return 'NS     '

    # HANDLE THE CORRECT NUMBER OF STEPS TO SIMULATE
    def _simulate(self, steps_todo):
        intersection_queue, summed_wait = self._get_stats() # init the summed_wait, in order to avoid a null return
        if (self._steps + steps_todo) >= self._max_steps: # do not do more steps than the maximum number of steps
            steps_todo = self._max_steps - self._steps
        while steps_todo > 0:
            traci.simulationStep() # simulate 1 step in sumo
            self._steps = self._steps + 1
            steps_todo -= 1
            intersection_queue, summed_wait = self._get_stats()     # why just get final step instead get every step.
            # intersection_queue: queue of this step
            self._sum_intersection_queue += intersection_queue      # sum_queue is increased every step.
        return summed_wait

    # RETRIEVE THE STATS OF THE SIMULATION FOR ONE SINGLE STEP
    def _get_stats(self):
        halt_right = traci.edge.getLastStepHaltingNumber("right_in") # number of cars in halt in a road
        halt_top = traci.edge.getLastStepHaltingNumber("top_in")
        halt_bot = traci.edge.getLastStepHaltingNumber("bottom_in")
        intersection_queue = halt_right + halt_top + halt_bot
        
        wait_right = traci.edge.getWaitingTime("right_in") # total waiting times of cars in a road
        wait_top = traci.edge.getWaitingTime("top_in")
        wait_bottom = traci.edge.getWaitingTime("bottom_in")
        summed_wait = wait_right + wait_top + wait_bottom                # total waiting time of all vehicles (at time step t)

        return intersection_queue, summed_wait

    # SET IN SUMO A YELLOW PHASE
    def _set_yellow_phase(self, old_action):
        yellow_phase = old_action * 2 + 1 # obtain the correct yellow_phase_number based on the old action
        traci.trafficlight.setPhase(constants.light_id, yellow_phase)

    # SET IN SUMO A GREEN PHASE
    def _set_green_phase(self, phase_number):
        if phase_number == 0:
            traci.trafficlight.setPhase(constants.light_id, 0)
        elif phase_number == 1:
            traci.trafficlight.setPhase(constants.light_id, 2)

    # SAVE THE STATS OF THE EPISODE TO PLOT THE GRAPHS AT THE END OF THE SESSION
    def _save_stats(self, traffic_code, tot_neg_reward): # save the stats for this episode
        if traffic_code == 0: # data low
            self._reward_store_LOW.append(tot_neg_reward) # how much negative reward in this episode
            self._cumulative_wait_store_LOW.append(self._sum_intersection_queue) # total number of seconds waited by cars (1 step = 1 second -> 1 car in queue/step = 1 second in queue/step=
            self._avg_intersection_queue_store_LOW.append(self._sum_intersection_queue / self._max_steps) # average number of queued cars per step, in this episode

        if traffic_code == 1: # data high
            self._reward_store_HIGH.append(tot_neg_reward)
            self._cumulative_wait_store_HIGH.append(self._sum_intersection_queue)
            self._avg_intersection_queue_store_HIGH.append(self._sum_intersection_queue / self._max_steps)

        if traffic_code == 2: # data ns
            self._reward_store_NS.append(tot_neg_reward)
            self._cumulative_wait_store_NS.append(self._sum_intersection_queue)
            self._avg_intersection_queue_store_NS.append(self._sum_intersection_queue / self._max_steps)

        if traffic_code == 3: # data ew
            self._reward_store_EW.append(tot_neg_reward)
            self._cumulative_wait_store_EW.append(self._sum_intersection_queue)
            self._avg_intersection_queue_store_EW.append(self._sum_intersection_queue / self._max_steps)

    def showMatrix(self, matrix):
        for i in range(60):
            for j in range(60):
                print (int(matrix[i][j]), end = "")
            print('')
        print('')

    # select action:
    def _choose_action(self, state):
        if random.random() <= self._epsilon:
            return random.randint(0, self._agent.num_actions - 1) # random action
        else:
            return np.argmax(self._agent.model.predict(state))

    # get minibatch and train
    def _replay(self):
        minibatch = self._memory.get_samples(self._agent.batch_size) # retrieve a group of samples
        if len(minibatch) > 0: # if there is at least 1 sample in the memory
            for state, action, reward, next_state in minibatch:
                q_values_s = self._agent.model.predict(state)				# q_values of s
                



                # Úsing target network:
                q_values_next_s = self._target_agent.model.predict(next_state)

                # ver 1: non double
                # q_target = reward + self._gamma * np.amax(q_values_next_s)		# q_target (number)
                
                # ver 2: double
                index_action = np.argmax(q_values_s) 								# index of action causing max of q_values_s
                q_target = reward + self._gamma * q_values_next_s[0][index_action]		# q_target (number)

                q_values_s[0][action] = q_target									# q_target (array of q_values)

                self._agent.model.fit(state, q_values_s, epochs=1, verbose=0)
        
            # update target network:
            self.update_target_weights(self._agent.model.get_weights())

    def update_target_weights(self, primary_network_weights):
        # v1: update through tau/_alpha_update_target:
        target_network_weights = self._target_agent.model.get_weights()
        for i in range(len(target_network_weights)):
            target_network_weights[i] = self._alpha_update_target*target_network_weights[i] + (1-self._alpha_update_target)*primary_network_weights[i]
            # update target weights
        self._target_agent.model.set_weights(target_network_weights)

        # v2: update directly:
        # self._target_agent.model.set_weights(primary_network_weights)

        # Duy Do State

    def get_lane_cell_lane0(self, lane_len, position, avg_cell):
        position = lane_len - position          # position with the end_of_lane (the traffic light)

        if position < avg_cell:
            return 0
        elif position < avg_cell*2:
            return 1
        elif position < avg_cell*3:
            return 2
        elif position < avg_cell*4:
            return 3
        elif position < avg_cell*5:
            return 4
        elif position < avg_cell*6:
            return 5
        elif position < avg_cell*7:
            return 6
        elif position < avg_cell*8:
            return 7
        elif position < avg_cell*9:
            return 8
        elif position < avg_cell*10:
            return 9
        if position < avg_cell*11:
            return 10
        elif position < avg_cell*12:
            return 11
        elif position < avg_cell*13:
            return 12
        elif position < avg_cell*14:
            return 13
        elif position < avg_cell*15:
            return 14
        elif position < avg_cell*15 + 10:
            return 15
        elif position < avg_cell*15 + 20:
            return 16
        elif position < avg_cell*15 + 30:
            return 17
        elif position < avg_cell*15 + 40:
            return 18
        elif position < avg_cell*15 + 50:
            return 19
        elif position < avg_cell*15 + 60:
            return 20
        elif position < avg_cell*15 + 70:
            return 21
        elif position < avg_cell*15 + 80:
            return 22
        elif position < avg_cell*15 + 90:
            return 23
        elif position < avg_cell*15 + 100:
            return 24
        elif position < avg_cell*15 + 150:
            return 25
        elif position < avg_cell*15 + 200:
            return 26
    
    def get_lane_cell_lane1(self, lane_len, position, avg_cell):
        position = lane_len - position          # position with the end_of_lane (the traffic light)
        if position < avg_cell:
            return 0
        elif position < avg_cell*2:
            return 1
        elif position < avg_cell*3:
            return 2
        elif position < avg_cell*4:
            return 3
        elif position < avg_cell*5:
            return 4
        elif position < avg_cell*6:
            return 5
        elif position < avg_cell*7:
            return 6
        elif position < avg_cell*8:
            return 7
        elif position < avg_cell*9:
            return 8
        elif position < avg_cell*10:
            return 9
        if position < avg_cell*11:
            return 10
        elif position < avg_cell*12:
            return 11
        elif position < avg_cell*13:
            return 12
        elif position < avg_cell*14:
            return 13
        elif position < avg_cell*15:
            return 14
        elif position < avg_cell*15 + 10:
            return 15
        elif position < avg_cell*15 + 20:
            return 16
        elif position < avg_cell*15 + 30:
            return 17
        elif position < avg_cell*15 + 40:
            return 18
        elif position < avg_cell*15 + 50:
            return 19
        elif position < avg_cell*15 + 60:
            return 20
        elif position < avg_cell*15 + 70:
            return 21
        elif position < avg_cell*15 + 80:
            return 22
        elif position < avg_cell*15 + 90:
            return 23
        elif position < avg_cell*15 + 100:
            return 24
        elif position < avg_cell*15 + 110:
            return 25
        elif position < avg_cell*15 + 120:
            return 26

    def _get_state(self, I):
            position_matrix = np.zeros([60,60])
            # speed_matrix = np.zeros([60,60])

            # for every vehicle in current-time-step:
            for vehicle_id in traci.vehicle.getIDList():
                lane_position = traci.vehicle.getLanePosition(vehicle_id)       # position with the beginning_of_lane
                lane_id = traci.vehicle.getLaneID(vehicle_id)
                # speed = traci.vehicle.getSpeed(vehicle_id)
                if (lane_id == 'bottom_in_0') or (lane_id == 'top_in_0') or (lane_id == 'right_in_0'):
                    lane_cell = self.get_lane_cell_lane0(200,lane_position, 2.7)
                elif (lane_id == 'bottom_in_1') or (lane_id == 'top_in_1') or (lane_id == 'right_in_1'):
                    lane_cell = self.get_lane_cell_lane1(210,lane_position, 6.5)
                elif (lane_id == 'bottom_in_2') or (lane_id == 'top_in_2') or (lane_id == 'right_in_2'):
                    lane_cell = self.get_lane_cell_lane1(210,lane_position,7)
                    
                if lane_id == 'bottom_in_0':                               # bot top
                    position_matrix[33+lane_cell][5] = 1
                elif lane_id == 'bottom_in_1':
                    position_matrix[33+lane_cell][4] = 1
                elif lane_id == 'bottom_in_2':
                    position_matrix[33+lane_cell][3] = 1

                elif lane_id == 'top_in_0':                             # top bot
                        position_matrix[26-lane_cell][0] = 1
                elif lane_id == 'top_in_1':
                    position_matrix[26-lane_cell][1] = 1
                elif lane_id == 'top_in_2':
                    position_matrix[26-lane_cell][2] = 1

                elif lane_id == 'right_in_0':                             # left right
                    position_matrix[27][6+lane_cell] = 1
                elif lane_id == 'right_in_1':
                    position_matrix[28][6+lane_cell] = 1
                elif lane_id == 'right_in_2':
                    position_matrix[29][6+lane_cell] = 1
            
            # position and speed:
            # outputMatrix = [position_matrix, speed_matrix]
            # output = np.transpose(outputMatrix) # np.array(outputMatrix)
            # output = output.reshape(1,60,60,2)
            
            # just position:
            outputMatrix = [position_matrix]
            output = np.transpose(outputMatrix) # np.array(outputMatrix)
            output = output.reshape(1,60,60,1)
            # self.showMatrix(position_matrix)

            return [output, I]
    
    @property
    def reward_store_LOW(self):
        return self._reward_store_LOW

    @property
    def cumulative_wait_store_LOW(self):
        return self._cumulative_wait_store_LOW

    @property
    def avg_intersection_queue_store_LOW(self):
        return self._avg_intersection_queue_store_LOW

    @property
    def reward_store_HIGH(self):
        return self._reward_store_HIGH

    @property
    def cumulative_wait_store_HIGH(self):
        return self._cumulative_wait_store_HIGH

    @property
    def avg_intersection_queue_store_HIGH(self):
        return self._avg_intersection_queue_store_HIGH

    @property
    def reward_store_NS(self):
        return self._reward_store_NS

    @property
    def cumulative_wait_store_NS(self):
        return self._cumulative_wait_store_NS

    @property
    def avg_intersection_queue_store_NS(self):
        return self._avg_intersection_queue_store_NS

    @property
    def reward_store_EW(self):
        return self._reward_store_EW

    @property
    def cumulative_wait_store_EW(self):
        return self._cumulative_wait_store_EW

    @property
    def avg_intersection_queue_store_EW(self):
        return self._avg_intersection_queue_store_EW

# HANDLE THE MEMORY
class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory # size of memory
        self._samples = []

    # ADD A SAMPLE INTO THE MEMORY
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0) # if the length is greater than the size of memory, remove the oldest element

    # GET n_samples SAMPLES RANDOMLY FROM THE MEMORY
    def get_samples(self, n_samples):
        if n_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples)) # get all the samples
        else:
            return random.sample(self._samples, n_samples) # get "batch size" number of samples

# PLOT AND SAVE THE STATS ABOUT THE SESSION
def save_graphs(sumo_simulation, total_episodes, mode, plot_path):
    plt.rcParams.update({'font.size': 18})
    # x = np.linspace(0, total_episodes, math.ceil(total_episodes/4))
    # reward
    data = sumo_simulation._reward_store             # neg-reward
    plt.plot(data)
    plt.ylabel("Cumulative negative reward = Total negative reward")
    plt.xlabel("Epoch")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val + 0.05 * min_val, max_val - 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'reward_' + mode + '.png', dpi=96)
    plt.close("all")

    # cumulative wait
    data = sumo_simulation._cumulative_wait_store            # total length queue ~ _sum_intersection_queue
    plt.plot(data)
    plt.ylabel("Cumulative delay (s) = Total queue length")
    plt.xlabel("Epoch")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'delay_' + mode + '.png', dpi=96)
    plt.close("all")

    # average number of cars in queue
    data = sumo_simulation._avg_intersection_queue_store
    plt.plot(data)
    plt.ylabel("Average queue length (vehicles) = Total queue lenght / maxs teps")
    plt.xlabel("Epoch")
    plt.margins(0)
    min_val = min(data)
    max_val = max(data)
    plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'queue_' + mode + '.png', dpi=96)
    plt.close("all")
def save_graphs_for_one_mode(sumo_simulation, total_episodes, mode, plot_path):
    plt.rcParams.update({'font.size': 18})
    # reward
    if mode == "L":
        data = sumo_simulation.reward_store_LOW             # neg-reward
    if mode == "H":
        data = sumo_simulation.reward_store_HIGH
    if mode == "NS":
        data = sumo_simulation.reward_store_NS
    if mode == "EW":
        data = sumo_simulation.reward_store_EW
    
    plt.plot(data)
    plt.ylabel("Cumulative negative reward = Total negative reward")
    plt.xlabel("Epoch")
    plt.margins(0)
    
    # min_val = min(data)
    # max_val = max(data)

    # plt.ylim(min_val + 0.05 * min_val, max_val - 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'reward_' + mode + '.png', dpi=96)
    plt.close("all")

    # cumulative wait
    if mode == "L":
        data = sumo_simulation.cumulative_wait_store_LOW            # total length queue ~ _sum_intersection_queue
    if mode == "H":
        data = sumo_simulation.cumulative_wait_store_HIGH
    if mode == "NS":
        data = sumo_simulation.cumulative_wait_store_NS
    if mode == "EW":
        data = sumo_simulation.cumulative_wait_store_EW
    plt.plot(data)
    plt.ylabel("Cumulative delay (s) = Total queue length")
    plt.xlabel("Epoch")
    plt.margins(0)
    # min_val = min(data)
    # max_val = max(data)
    # plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'delay_' + mode + '.png', dpi=96)
    plt.close("all")

    # average number of cars in queue
    if mode == "L":
        data = sumo_simulation.avg_intersection_queue_store_LOW
    if mode == "H":
        data = sumo_simulation.avg_intersection_queue_store_HIGH
    if mode == "NS":
        data = sumo_simulation.avg_intersection_queue_store_NS
    if mode == "EW":
        data = sumo_simulation.avg_intersection_queue_store_EW
    plt.plot(data)
    plt.ylabel("Average queue length (vehicles) = Total queue lenght / maxs teps")
    plt.xlabel("Epoch")
    plt.margins(0)
    # min_val = min(data)
    # max_val = max(data)
    # plt.ylim(min_val - 0.05 * min_val, max_val + 0.05 * max_val)
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    fig.savefig(plot_path + 'queue_' + mode + '.png', dpi=96)
    plt.close("all")

def main():
    # ---------------------------- CONFIGURATION for TRAINING in constants.py -----------------------------------
    total_episodes = constants.total_episodes
    batch_size = constants.batch_size
    gamma = constants.gamma			
    path = constants.path
    memory_size = constants.memory_size
    num_actions = constants.num_actions
    max_steps = constants.max_steps            
    sumoCmd = constants.sumoCmd
    green_duration = constants.green_duration
    yellow_duration = constants.yellow_duration
    # create folder:
    if not os.path.exists(path):
        os.makedirs(path)
    # ------------------------------------------------------------------------------------------------------------

    # Primary objects:
    agent = DQNAgent(num_actions, batch_size)               # primary agent
    target_agent = DQNAgent(num_actions, batch_size)        # target agent
    memory = Memory(memory_size)                            # memory
    sumo_simulation = SumoSimulation(agent, target_agent, memory, gamma, max_steps, sumoCmd, green_duration, yellow_duration) # sumo

    # Run every simulation:
    current_episode = 0     
    while current_episode < total_episodes:
        print('----- Simulation {} of {}'.format(current_episode+1, total_episodes))
        start = timeit.default_timer()

        # START: RUN 1 SIMULATION ------------------------------------------------------------------
        sumo_simulation.run_one_episode(current_episode, total_episodes)
        current_episode += 1
        # -----------------------------------------------------------------------------------------
        
        # END: after running 1 simulation:
        stop = timeit.default_timer()
        total_time = round(stop - start, 1)
        remaining_time = (total_episodes - current_episode - 1)*total_time/3600
        hours = int(remaining_time)
        mins = (remaining_time - hours)*60
        print('Time for this episode: ', total_time)
        print('Remaing time for training: ', hours, 'hours ', mins, 'mins')

        # ------------------------------------ JUST SAVE AGENT EVERY SIMULATION ---------------------
        if current_episode % 5 == 0:
            agent.save(path+'primary_model_weights_'+ str(current_episode)+'.h5')
            target_agent.save(path+'target_model_weights_'+ str(current_episode)+'.h5')

        # --------------------------------------- LOG CHARTS -----------------------------------------
        save_graphs_for_one_mode(sumo_simulation, total_episodes, "L", path)
        save_graphs_for_one_mode(sumo_simulation, total_episodes, "H", path)
        save_graphs_for_one_mode(sumo_simulation, total_episodes, "NS", path)
        save_graphs_for_one_mode(sumo_simulation, total_episodes, "EW", path)    

        # ----------- lOG ARRAY REARDS --------------------------------------------------------------
        log2 = open(path+'/rewards.txt','w')
        text = 'L: ' + str(sumo_simulation._reward_store_LOW) + '\n'
        text += 'H: ' + str(sumo_simulation._reward_store_HIGH) + '\n'
        text += 'NS: ' + str(sumo_simulation._reward_store_NS) + '\n'
        text += 'EW: ' + str(sumo_simulation._reward_store_EW) + '\n'
        log2.write(text)
        print('\n\n')

    # AFTER FISHING ALL EPS: 
    print("----- End time:", datetime.datetime.now())
    agent.save(path+'last_model.h5')
    save_graphs(sumo_simulation, total_episodes, "L", path)
    save_graphs(sumo_simulation, total_episodes, "H", path)
    save_graphs(sumo_simulation, total_episodes, "NS", path)
    save_graphs(sumo_simulation, total_episodes, "EW", path)

if __name__ == "__main__":
	main()
