from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import random
import numpy as np
import math
import time
import constants
import timeit
import matplotlib.pyplot as plt
import datetime
from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Average, Add, Dot, Subtract, Multiply
from keras.models import Model
from keras.optimizers import Adam
import genorator

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to kill warning about tensorflow


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

# phase codes based on xai_tlcs.net.xml
PHASE_NS_GREEN = 0 # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2 # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4 # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6 # action 3 code 11
PHASE_EWL_YELLOW = 7

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
		self._green_duration = green_duration					
		self._yellow_duration = yellow_duration
		self._alpha_update_target = 0.0001

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
			return 'LOW'
		elif simu_type == 1:
			return 'HIGH' 
		elif simu_type == 2:
			return 'NS' 
		elif simu_type == 3:
			return 'EW' 

	def reset(self):
		self._neg_pos_reward_store = []                             # list reward/ action-step
		self._summed_wait_store = []                       			# list sum-wating-time
		self._carID_that_waited = []
		self._carID_aheaddx = []
		self._carID_sx = []
		self._car_gen_step = []
		self._n_cars_generated = 0

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

	# Run 1 simulation:
	def run_one_episode(self, simu_type, total_episodes):
		
		traffic_code_mode = genorator.gen_route(simu_type)						# gen route file.       
		print('Mode: ', self.get_simu_type_str(simu_type))
		traci.start(self._sumoCmd)														# then, start sumo

		self.reset()

		# INIT some vars:
		self._steps = 0			
		self._sum_intersection_queue = 0		# increases every step/seconds
		tot_neg_reward = 0						# total negative reward
		pre_wait_time = 0						# 

		# INIT my vars:
		new_action = 0						# initial action
		state = self._get_state(self.I)

		# test
		actions_count = [0,0]				# count numb_times of picked actions.
		# run 1 simulation (1h30m)
		while self._steps < self._max_steps:
			# reset current_wait_time:
			current_wait_time = 0

			# select action (select index of action, then edit action_time)
			new_action = self._choose_action(state)
			
			# for writing log:
			actions_count[new_action] += 1
			action_name = self.get_action_name(new_action)			

            #  ================================ Take new_action ====================================================================
			if self._steps != 0 and old_action != new_action:
				self._set_yellow_phase(old_action)
				current_wait_time = self._simulate(self._yellow_duration)           
			# take action:
			self._set_green_phase(new_action)
			current_wait_time = self._simulate(self._green_duration)
			#  ======================================================================================================================

			# get next_state and reward
			next_state = self._get_state(self.I)
			reward = pre_wait_time - current_wait_time
			self._neg_pos_reward_store.append(reward)

			if reward < 0:
				tot_neg_reward += reward
			
		
			# reassign:
			pre_wait_time = current_wait_time
			state = next_state
			old_action = new_action

			# print
			print('step: ', self._steps, ' || action: ', action_name, ' || negative reward: ', tot_neg_reward)


		# AFTER run 1 simulation:
		#  print percent of taken action:
		p_EW = round(100*actions_count[0]/sum(actions_count),2)
		p_NS = round(100*actions_count[1]/sum(actions_count),2)
		print('EW: ', p_EW)
		print('NS: ', p_NS)

		self._save_stats(traffic_code_mode, tot_neg_reward)		# mode LOW + total neg-REWARD
		print("Total negative reward: {}, Eps: {}".format(tot_neg_reward, self._epsilon))
		
		mode = self.get_simu_type_str(traffic_code_mode)
		log_path = constants.plot_path_trained_model + mode + '/'
		os.makedirs(os.path.dirname(log_path), exist_ok=True)
		log = open(log_path +'tog_neg_reward.txt', 'a')  # open file text.
		text = mode + ' reward: ' + str(tot_neg_reward) + ' _sum_intersection_queue: ' + str(self._sum_intersection_queue)
		log.write(text)
		log.close()

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
		intersection_queue, summed_wait, arrived_now = self._get_stats()
		if (self._steps + steps_todo) >= self._max_steps: # do not do more steps than the maximum number of steps
			steps_todo = self._max_steps - self._steps
		while steps_todo > 0:
			traci.simulationStep() # simulate 1 step in sumo
			self._steps = self._steps + 1
			steps_todo -= 1
			intersection_queue, summed_wait, arrived_now = self._get_stats()
			self._summed_wait_store.append(summed_wait)
			self._sum_intersection_queue += intersection_queue
		return summed_wait


	# RETRIEVE THE STATS OF THE SIMULATION FOR ONE SINGLE STEP
	def _get_stats(self):
		route_turnleft = {"W_N", "N_E", "E_S", "S_W"}
		intersection_queue = 0
		summed_wait = 0
		for veh_id in traci.vehicle.getIDList():
			wait_time_car = traci.vehicle.getWaitingTime(veh_id)
			if wait_time_car > 0.5:
				intersection_queue += 1
				self._carID_that_waited.append(veh_id)
				route_ID = traci.vehicle.getRouteID(veh_id)
				if route_ID in route_turnleft:
					self._carID_sx.append(veh_id)
				else:
					self._carID_aheaddx.append(veh_id)
			summed_wait += wait_time_car
		arrived_now = traci.simulation.getArrivedNumber()
		return intersection_queue, summed_wait, arrived_now

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
	def _save_stats(self, traffic_code_mode, tot_neg_reward): # save the stats for this episode
		if traffic_code_mode == 1: # data low
			self._reward_store_LOW.append(tot_neg_reward) # how much negative reward in this episode
			self._cumulative_wait_store_LOW.append(self._sum_intersection_queue) # total number of seconds waited by cars (1 step = 1 second -> 1 car in queue/step = 1 second in queue/step=
			self._avg_intersection_queue_store_LOW.append(self._sum_intersection_queue / self._max_steps) # average number of queued cars per step, in this episode

		if traffic_code_mode == 2: # data high
			self._reward_store_HIGH.append(tot_neg_reward)
			self._cumulative_wait_store_HIGH.append(self._sum_intersection_queue)
			self._avg_intersection_queue_store_HIGH.append(self._sum_intersection_queue / self._max_steps)

		if traffic_code_mode == 3: # data ns
			self._reward_store_NS.append(tot_neg_reward)
			self._cumulative_wait_store_NS.append(self._sum_intersection_queue)
			self._avg_intersection_queue_store_NS.append(self._sum_intersection_queue / self._max_steps)

		if traffic_code_mode == 4: # data ew
			self._reward_store_EW.append(tot_neg_reward)
			self._cumulative_wait_store_EW.append(self._sum_intersection_queue)
			self._avg_intersection_queue_store_EW.append(self._sum_intersection_queue / self._max_steps)

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

	# select action:
	def _choose_action(self, state):
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

def get_simu_type(type):
	if type == 0:
		return 'LOW'
	elif type == 1:
		return 'HIGH'
	elif type == 2:
		return 'NS'
	elif type == 3:
		return 'EW'


def save_charts(sumo_simu, plot_path, simu_type, neg_pos_rewards, summed_waiting_times):
	plot_path += simu_type + '/'
	os.makedirs(os.path.dirname(plot_path), exist_ok=True)
	
	# save reward chard:
	plt.rcParams.update({'font.size': 18})
	data = neg_pos_rewards
	plt.plot(data)
	plt.ylabel("Neg Pos reward")
	plt.xlabel("n-th action")
	plt.margins(0)
	min_val = min(data)
	max_val = max(data)
	plt.ylim(min_val + 0.05 * min_val, max_val + 0.05 * max_val)
	plt.xlim(0, len(data))
	fig = plt.gcf()
	fig.set_size_inches(20, 11.25)
	fig.savefig(plot_path + '/reward_model.png', dpi=96)
	plt.close("all")
	# save np array:
	np.save(plot_path+'reward_array_model',np.array(data))

	# save waiting time chard:
	data = summed_waiting_times
	plt.plot(data)
	plt.ylabel("Waiting Time / Second(s)")
	plt.xlabel("Timeline")
	plt.margins(0)
	min_val = min(data)
	max_val = max(data)
	plt.ylim(min_val + 0.05 * min_val, max_val + 0.05 * max_val)
	plt.xlim(0, len(data))
	fig = plt.gcf()
	fig.set_size_inches(20, 11.25)
	fig.savefig(plot_path + 'delay_model.png', dpi=96)
	plt.close("all")	
	# save np array:
	np.save(plot_path+'delay_array_model',np.array(data))

def save_chart(low_fix_reward, low_model_reward, path, metric_type):

    plt.rcParams.update({'font.size': 18})
    data = low_fix_reward
    plt.plot(data, label='Fix time')  
    plt.ylabel("Cumulative Delay (s)")
    plt.xlabel("n-th step (s)")
    plt.margins(0)
    plt.plot(low_model_reward, label = 'Model - 200 simulations')
    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    plt.legend(loc='upper left')
    fig.savefig(path + '/compare_'+ metric_type +'.png', dpi=96)
    plt.close("all")

def save_comparation():
	# fixed path:
    fixed_path = constants.plot_path_fixed_sys

    # model path:
    model_path = constants.plot_path_trained_model

    # save comparation:
    comparation_path = constants.comparation_path
    os.makedirs(os.path.dirname(comparation_path), exist_ok=True)


    low_fix_reward = np.load(fixed_path+'LOW/reward_array_fix.npy')
    high_fix_reward = np.load(fixed_path+'HIGH/reward_array_fix.npy')
    ns_fix_reward = np.load(fixed_path+'NS/reward_array_fix.npy')
    ew_fix_reward = np.load(fixed_path+'EW/reward_array_fix.npy')

    low_fix_delay = np.load(fixed_path+'LOW/delay_array_fix.npy')
    high_fix_delay = np.load(fixed_path+'HIGH/delay_array_fix.npy')
    ns_fix_delay = np.load(fixed_path+'NS/delay_array_fix.npy')
    ew_fix_delay = np.load(fixed_path+'EW/delay_array_fix.npy')
        
    # load model array:
    low_model_reward = np.load(model_path+'LOW/reward_array_model.npy')
    high_model_reward = np.load(model_path+'HIGH/reward_array_model.npy')
    ns_model_reward = np.load(model_path+'NS/reward_array_model.npy')
    ew_model_reward = np.load(model_path+'EW/reward_array_model.npy')

    low_model_delay = np.load(model_path+'LOW/delay_array_model.npy')
    high_model_delay = np.load(model_path+'HIGH/delay_array_model.npy')
    ns_model_delay = np.load(model_path+'NS/delay_array_model.npy')
    ew_model_delay = np.load(model_path+'EW/delay_array_model.npy')

    # 
    save_chart(low_fix_reward, low_model_reward, comparation_path, 'REWARD_LOW')        
    save_chart(high_fix_reward, high_model_reward, comparation_path, 'REWARD_HIGH')        
    save_chart(ns_fix_reward, ns_model_reward, comparation_path, 'REWARD_NS')        
    save_chart(ew_fix_reward, ew_model_reward, comparation_path, 'REWARD_EW')        
      
    save_chart(low_fix_delay, low_model_delay, comparation_path, 'DELAY_LOW')        
    save_chart(high_fix_delay, high_model_delay, comparation_path, 'DELAY_HIGH')        
    save_chart(ns_fix_delay, ns_model_delay, comparation_path, 'DELAY_NS')        
    save_chart(ew_fix_delay, ew_model_delay, comparation_path, 'DELAY_EW')        


def main():

	# ---------------------------- CONFIGURATION for EVALUATION -----------------------------------
	model_name = constants.loaded_model_name
	simu_types = constants.simu_types 
	plot_path = constants.plot_path_trained_model
	os.makedirs(os.path.dirname(plot_path), exist_ok=True)
	green_duration = constants.green_duration_eval
	yellow_duration = constants.yellow_duration_eval
	# ---------------------------- CONFIGURATION for EVALUATION -----------------------------------

	# --------- NO EDIT -----------------------------------------------------------------------------------------------------------
	total_episodes = 4
	batch_size = constants.batch_size
	gamma = constants.gamma			
	memory_size = constants.memory_size
	num_actions = constants.num_actions
	max_steps = constants.max_steps            
	sumoCmd = constants.sumoCmd

	# load agent:
	agent = DQNAgent(num_actions, batch_size)              	# primary agent
	try:
		agent.load('trained_models/' + model_name)
		print('load model successfull! waiting for 5s...')
		time.sleep(5)
	except:
		print('No models found. Please check again! \n')
		time.sleep(100)

	target_agent = DQNAgent(num_actions, batch_size)       	# target agent
	memory = Memory(memory_size)                            # memory
	sumo_simulation = SumoSimulation(agent, target_agent, memory, gamma, max_steps, sumoCmd, green_duration, yellow_duration) # sumo
	# --------------------------------------------------------------------------------------------------------------------------------------

	# for simu_type in simu_types:
	# 	# START: RUN 1 simulation ------------------------------------------------------------------
	# 	sumo_simulation.run_one_episode(simu_type, total_episodes)
		
	# 	simu_type = get_simu_type(simu_type)
	# 	print('End simulation: ', simu_type)
		
	# 	# get data:
	# 	neg_pos_rewards = sumo_simulation._neg_pos_reward_store			# reward/action-step
	# 	summed_waiting_times = sumo_simulation._summed_wait_store		# waiting_time/1second
	# 	# log:

	# 	# plot charts:
	# 	save_charts(sumo_simulation, plot_path, simu_type, neg_pos_rewards, summed_waiting_times)
		
	
	
	# plot compare_charts:
	save_comparation()


if __name__ == "__main__":
	main()

