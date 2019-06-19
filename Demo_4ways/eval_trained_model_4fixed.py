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
import count_veh

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
		
		# # best model =======eval_trained_model=================================================================
		self._num_actions = num_actions
		self._batch_size = batch_size
		self._num_states = 80						# hard code

		# now setup the model
		self.model = self._define_model()

	# define architect of model
	def _define_model(self):
		input_1 = Input(shape=(60, 60, 2))
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
		self.list_lane_ids = ['gneE86_1','gneE86_2','gneE86_3','gneE85_1','gneE85_2','gneE85_3','gneE21_1','gneE21_2','gneE21_3','gneE89_1','gneE89_2','gneE89_3']

		self._agent = agent
		self._target_agent = target_agent
		self._memory = memory
		self._epsilon = 0 							# controls the explorative/exploitative payoff
		self._gamma = gamma
		self._max_steps = max_steps
		self._sumoCmd = sumoCmd


		self._reward_store_LOW = []
		self._cumulative_wait_store_LOW = []
		self._avg_intersection_queue_store_LOW = []             # mỗi giây có bao nhiêu thằng đợi?
		self._avg_waiting_time_per_veh_LOW = []                 # mỗi veh đợi trung bình bao nhiêu giây?

		self._reward_store_HIGH = []
		self._cumulative_wait_store_HIGH = []
		self._avg_intersection_queue_store_HIGH = []
		self._avg_waiting_time_per_veh_HIGH = []                 # mỗi veh đợi trung bình bao nhiêu giây?


		self._reward_store_NS = []
		self._cumulative_wait_store_NS = []
		self._avg_intersection_queue_store_NS = []
		self._avg_waiting_time_per_veh_NS = []                 # mỗi veh đợi trung bình bao nhiêu giây?


		self._reward_store_EW = []
		self._cumulative_wait_store_EW = []
		self._avg_intersection_queue_store_EW = []
		self._avg_waiting_time_per_veh_EW = []                 # mỗi veh đợi trung bình bao nhiêu giây?


		self._avg_waiting_time_per_veh_RANDOM = []                 # mỗi veh đợi trung bình bao nhiêu giây?

	def get_simu_type_str(self,simu_type):
		if simu_type == 0:
			return 'LOW'
		elif simu_type == 1:
			return 'HIGH' 
		elif simu_type == 2:
			return 'NS' 
		elif simu_type == 3:
			return 'EW' 
		elif simu_type == 4:
			return constants.simu_type_of_random
	
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
		self._avg_intersection_queue_store_LOW = []             # mỗi giây có bao nhiêu thằng đợi?
		self._avg_waiting_time_per_veh_LOW = []                 # mỗi veh đợi trung bình bao nhiêu giây?

		self._reward_store_HIGH = []
		self._cumulative_wait_store_HIGH = []
		self._avg_intersection_queue_store_HIGH = []
		self._avg_waiting_time_per_veh_HIGH = []                 # mỗi veh đợi trung bình bao nhiêu giây?


		self._reward_store_NS = []
		self._cumulative_wait_store_NS = []
		self._avg_intersection_queue_store_NS = []
		self._avg_waiting_time_per_veh_NS = []                 # mỗi veh đợi trung bình bao nhiêu giây?


		self._reward_store_EW = []
		self._cumulative_wait_store_EW = []
		self._avg_intersection_queue_store_EW = []
		self._avg_waiting_time_per_veh_EW = []                 # mỗi veh đợi trung bình bao nhiêu giây?


		self._avg_waiting_time_per_veh_RANDOM = []                 # mỗi veh đợi trung bình bao nhiêu giây?

	# Run 1 simulation:
	def run_one_episode(self, simu_type, total_episodes):
		
		traffic_code_mode = genorator.gen_route(simu_type, is_random = False)						# gen route file.       
		print('Mode: ', self.get_simu_type_str(traffic_code_mode))
		traci.start(self._sumoCmd)														# then, start sumo
		traci.gui.setZoom('View #0',1500)
		traci.gui.setOffset('View #0',595,916)
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
		# while (traci.simulation.getMinExpectedNumber() > 0):
			# reset current_wa75217it_time:
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
			# print('step: ', self._steps, ' || action: ', action_name, ' || negative reward: ', tot_neg_reward)


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


	def plot_delay(self,data):

		if constants.is_test_random:
			agent = "Agent 2"
			if constants.is_load_Agent_1:
				agent = "Agent 1"
			plt.figure('Evaluate simulation: ' + constants.simu_type_of_random + ' by ' + agent)
		else:
			simu_type = "LOW"
			if constants.simu_types[0] == 0:
				simu_type = "LOW"
			if constants.simu_types[0] == 1:
				simu_type = "HIGH"
			if constants.simu_types[0] == 2:
				simu_type = "NS"
			if constants.simu_types[0] == 3:
				simu_type = "EW"

			agent = "Agent 2"
			if constants.is_load_Agent_1:
				agent = "Agent 1"
			plt.figure('Evaluate simulation: ' + simu_type + ' by ' + agent)

		plt.clf()
		plt.xlabel('Time line (s)')
		plt.ylabel('Average Waiting Time Per Vehicle (s)')
		
		# plot fixed_array:
		plt.plot(constants.fixed_neg_reward, label='STL (33,33)')

		if constants.is_load_Agent_1:
			my_label = "Agent 1"
		else:
			my_label = "Agent 2"
		plt.plot(data, label = my_label)
		plt.legend(loc='upper right')
		plt.pause(0.001)  # pause a bit so that plots are updated


	def get_current_veh(self):
		total_veh = 0
		for vehicle_id in traci.vehicle.getIDList():
			lane_id = traci.vehicle.getLaneID(vehicle_id)
			if lane_id in self.list_lane_ids:
				total_veh +=1
		return total_veh

	# HANDLE THE CORRECT NUMBER OF STEPS TO SIMULATE
	def _simulate(self, steps_todo):
		intersection_queue, summed_wait, arrived_now = self._get_stats()
		return_value = summed_wait
		if (self._steps + steps_todo) >= self._max_steps: # do not do more steps than the maximum number of steps
			steps_todo = self._max_steps - self._steps
		while steps_todo > 0:
			traci.simulationStep() # simulate 1 step in sumo
			self._steps = self._steps + 1
			steps_todo -= 1
			intersection_queue, summed_wait, arrived_now = self._get_stats()
			
			return_value = summed_wait 
			# update (22/05): thời gian chờ trung bình của mỗi xe/ mỗi giây.
			numb_cur_vehs = self.get_current_veh()
			if numb_cur_vehs != 0:
				summed_wait = summed_wait/numb_cur_vehs
			self._summed_wait_store.append(summed_wait)

			# plot:
			if self._steps%100 == 0:
				self.plot_delay(self._summed_wait_store)

			self._sum_intersection_queue += intersection_queue
		return return_value

	# RETRIEVE THE STATS OF THE SIMULATION FOR ONE SINGLE STEP
	def _get_stats(self):
		halt_N = traci.edge.getLastStepHaltingNumber("gneE21") # number of cars in halt in a road
		halt_S = traci.edge.getLastStepHaltingNumber("gneE86")
		halt_E = traci.edge.getLastStepHaltingNumber("gneE89")
		halt_W = traci.edge.getLastStepHaltingNumber("gneE85")
		intersection_queue = halt_N + halt_S + halt_E + halt_W          # total stopping vehicles (at time step t)
		
		wait_N = traci.edge.getWaitingTime("gneE21") # total waiting times of cars in a road
		wait_S = traci.edge.getWaitingTime("gneE86")
		wait_W = traci.edge.getWaitingTime("gneE89")
		wait_E = traci.edge.getWaitingTime("gneE85")
		summed_wait = wait_N + wait_S + wait_W + wait_E                # total waiting time of all vehicles (at time step t)

		return intersection_queue, summed_wait, 0
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
		numb_generated_veh = count_veh.cal_numb_generated_veh(self.get_simu_type_str(traffic_code), is_random = False)

		if traffic_code == 0: # data low
			self._reward_store_LOW.append(tot_neg_reward) # how much negative reward in this episode
			self._cumulative_wait_store_LOW.append(self._sum_intersection_queue) # total number of seconds waited by cars (1 step = 1 second -> 1 car in queue/step = 1 second in queue/step=
			self._avg_intersection_queue_store_LOW.append(self._sum_intersection_queue / self._max_steps) # average number of queued cars per step, in this episode
			self._avg_waiting_time_per_veh_LOW.append(self._sum_intersection_queue / numb_generated_veh)
		if traffic_code == 1: # data high
			self._reward_store_HIGH.append(tot_neg_reward)
			self._cumulative_wait_store_HIGH.append(self._sum_intersection_queue)
			self._avg_intersection_queue_store_HIGH.append(self._sum_intersection_queue / self._max_steps)
			self._avg_waiting_time_per_veh_HIGH.append(self._sum_intersection_queue / numb_generated_veh)
		if traffic_code == 2: # da		# INIT some vars:
			self._reward_store_NS.append(tot_neg_reward)
			self._cumulative_wait_store_NS.append(self._sum_intersection_queue)
			self._avg_intersection_queue_store_NS.append(self._sum_intersection_queue / self._max_steps)
			self._avg_waiting_time_per_veh_NS.append(self._sum_intersection_queue / numb_generated_veh)
		if traffic_code == 3: # data ew
			self._reward_store_EW.append(tot_neg_reward)
			self._cumulative_wait_store_EW.append(self._sum_intersection_queue)
			self._avg_intersection_queue_store_EW.append(self._sum_intersection_queue / self._max_steps)
			self._avg_waiting_time_per_veh_EW.append(self._sum_intersection_queue / numb_generated_veh)
		if traffic_code == 4:
			self._avg_waiting_time_per_veh_RANDOM.append(self._sum_intersection_queue / numb_generated_veh)

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
	def get_lane_cell_motobike(self, lane_len, position, avg_cell):
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
		elif position < avg_cell*11:
			return 10
		elif position < avg_cell*12:
			return 11
		elif position < avg_cell*13:
			return 12
		elif position < avg_cell*14:
			return 13
		elif position < avg_cell*15:
			return 14
		elif position < avg_cell*16:
			return 15
		elif position < avg_cell*17:
			return 16
		elif position < avg_cell*18:
			return 17   
		elif position < avg_cell*19:
			return 18
		elif position < avg_cell*20:
			return 19

	def get_lane_cell_mid_lane(self, lane_len, position, avg_cell):
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

	def get_child_index(self, vehicle_id):
		position = traci.vehicle.getLateralLanePosition(vehicle_id)
		if position > 0.875:
			return 0
		elif position > 0:
			return 1
		elif position > -0.875:
			return 2
		else:
			return 3

	def _get_state(self, I):
		position_matrix = np.zeros([60,60])
		speed_matrix = np.zeros([60,60])

		# for every vehicle in current-time-step:
		for vehicle_id in traci.vehicle.getIDList():

			lane_position = traci.vehicle.getLanePosition(vehicle_id)       # position with the beginning_of_lane
			lane_id = traci.vehicle.getLaneID(vehicle_id)
			speed = traci.vehicle.getSpeed(vehicle_id)
			
			# index of motobike in sub_lane:
			sub_lane_index = 0

			if (lane_id == 'gneE86_1') or (lane_id == 'gneE85_1') or (lane_id == 'gneE21_1') or (lane_id == 'gneE89_1'):
				lane_cell = self.get_lane_cell_motobike(50,lane_position, 2.5)
				sub_lane_index = self.get_child_index(vehicle_id)
			elif (lane_id == 'gneE86_2') or (lane_id == 'gneE85_2') or (lane_id == 'gneE21_2') or (lane_id == 'gneE89_2'):
				lane_cell = self.get_lane_cell_mid_lane(50,lane_position, 6.5)
			else:
				lane_cell = self.get_lane_cell_mid_lane(50,lane_position,7)
				
			if lane_id == 'gneE86_1':                                       # bot top
				position_matrix[36+lane_cell][32+sub_lane_index] = 1
				speed_matrix[36+lane_cell][32+sub_lane_index] = speed
			elif lane_id == 'gneE86_2':
				position_matrix[36+lane_cell][31] = 1
				speed_matrix[36+lane_cell][31] = speed
			elif lane_id == 'gneE86_3':
				position_matrix[36+lane_cell][30] = 1
				speed_matrix[36+lane_cell][30] = speed

			elif lane_id == 'gneE85_1':                                             # top bot
					position_matrix[23-lane_cell][27-sub_lane_index] = 1            
					speed_matrix[23-lane_cell][27-sub_lane_index] = speed
			elif lane_id == 'gneE85_2':
				position_matrix[23-lane_cell][28] = 1
				speed_matrix[23-lane_cell][28] = speed
			elif lane_id == 'gneE85_3':
				position_matrix[23-lane_cell][29] = 1
				speed_matrix[23-lane_cell][29] = speed

			elif lane_id == 'gneE21_1':                             # left right
				position_matrix[32+sub_lane_index][23-lane_cell] = 1
				speed_matrix[32+sub_lane_index][23-lane_cell] = speed
			elif lane_id == 'gneE21_2':
				position_matrix[31][23-lane_cell] = 1
				speed_matrix[31][23-lane_cell] = speed
			elif lane_id == 'gneE21_3':
				position_matrix[30][23-lane_cell] = 1
				speed_matrix[30][23-lane_cell] = speed

			elif lane_id == 'gneE89_1':                             # right left
				position_matrix[27-sub_lane_index][36+lane_cell] = 1
				speed_matrix[27-sub_lane_index][36+lane_cell] = speed
			elif lane_id == 'gneE89_2':
				position_matrix[28][36+lane_cell] = 1
				speed_matrix[28][36+lane_cell] = speed
			elif lane_id == 'gneE89_3':
				position_matrix[29][36+lane_cell] = 1
				speed_matrix[29][36+lane_cell] = speed
		
		# position and speed:
		outputMatrix = [position_matrix, speed_matrix]
		output = np.transpose(outputMatrix) # np.array(outputMatrix)
		output = output.reshape(1,60,60,2)

		# self.showMatrix(speed_matrix)

		# # just position:
		# outputMatrix = [position_matrix]
		# output = np.transpose(outputMatrix) # np.array(outputMatrix)
		# output = output.reshape(1,60,60,1)

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

def get_simu_type(type):
	if type == 0:
		return 'LOW'
	elif type == 1:
		return 'HIGH'
	elif type == 2:
		return 'NS'
	elif type == 3:
		return 'EW'
	elif type ==4:
		return constants.simu_type_of_random
def save_charts(sumo_simu, plot_path, simu_type, neg_pos_rewards, summed_waiting_times):
	# save reward chard:
	plt.rcParams.update({'font.size': 12})
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
	fixed_path_33 = constants.path + 'fixed_system_33/'
	fixed_path_40 = constants.path + 'fixed_system_40/'

	# model path:
	model_path = constants.plot_path_trained_model

	# save comparation:
	comparation_path = constants.comparation_path
	os.makedirs(os.path.dirname(comparation_path), exist_ok=True)



	if not constants.is_test_random:

		# avg waiting_time  33 (fixed)
		avg_waiting_time_LOW_33 = np.load(fixed_path_33+'LOW/_avg_waiting_time_per_veh.npy')
		avg_waiting_time_HIGH_33 = np.load(fixed_path_33+'HIGH/_avg_waiting_time_per_veh.npy')
		avg_waiting_time_NS_33 = np.load(fixed_path_33+'NS/_avg_waiting_time_per_veh.npy')
		avg_waiting_time_EW_33 = np.load(fixed_path_33+'EW/_avg_waiting_time_per_veh.npy')


		# avg waiting_time  40 (fixed)
		avg_waiting_time_LOW_40 = np.load(fixed_path_40+'LOW/_avg_waiting_time_per_veh.npy')
		avg_waiting_time_HIGH_40 = np.load(fixed_path_40+'HIGH/_avg_waiting_time_per_veh.npy')
		avg_waiting_time_NS_40 = np.load(fixed_path_40+'NS/_avg_waiting_time_per_veh.npy')
		avg_waiting_time_EW_40 = np.load(fixed_path_40+'EW/_avg_waiting_time_per_veh.npy')

	    
		# avg waiting_time (model)
		avg_waiting_time_LOW_model = np.load(model_path+'LOW/_avg_waiting_time_per_veh_LOW.npy')
		avg_waiting_time_HIGH_model = np.load(model_path+'HIGH/_avg_waiting_time_per_veh_HIGH.npy')
		avg_waiting_time_NS_model = np.load(model_path+'NS/_avg_waiting_time_per_veh_NS.npy')
		avg_waiting_time_EW_model = np.load(model_path+'EW/_avg_waiting_time_per_veh_EW.npy')

		if constants.simu_types[0] == 0:
			print('Compare AWT LOW: ')
			print('Fixed 33: ', round(avg_waiting_time_LOW_33[0],2))
			print('Fixed 40: ', round(avg_waiting_time_LOW_40[0],2))
			print('Agent: ', round(avg_waiting_time_LOW_model[0],2))
			print('Improvement 33: ', round((1 - avg_waiting_time_LOW_model/avg_waiting_time_LOW_33)[0]*100,2), '%')
			print('Improvement 40: ', round((1 - avg_waiting_time_LOW_model/avg_waiting_time_LOW_40)[0]*100,2), '%')
			print('\n')

		if constants.simu_types[0] == 1:			
			print('Compare AWT HIGH: ')
			print('Fixed 33: ', round(avg_waiting_time_HIGH_33[0],2))
			print('Fixed 40: ', round(avg_waiting_time_HIGH_40[0],2))
			print('Agent: ', round(avg_waiting_time_HIGH_model[0],2))
			print('Improvement 33: ', round((1 - avg_waiting_time_HIGH_model/avg_waiting_time_HIGH_33)[0]*100,2), '%')
			print('Improvement 40: ', round((1 - avg_waiting_time_HIGH_model/avg_waiting_time_HIGH_40)[0]*100,2), '%')
			print('\n')

		if constants.simu_types[0] == 2:
			print('Compare AWT NS: ')
			print('Fixed 33: ', round(avg_waiting_time_NS_33[0],2))
			print('Fixed 40: ', round(avg_waiting_time_NS_40[0],2))
			print('Agent: ', round(avg_waiting_time_NS_model[0],2))
			print('Improvement 33: ', round((1 - avg_waiting_time_NS_model/avg_waiting_time_NS_33)[0]*100,2), '%')
			print('Improvement 40: ', round((1 - avg_waiting_time_NS_model/avg_waiting_time_NS_40)[0]*100,2), '%')
			print('\n')

		if constants.simu_types[0] == 3:
			print('Compare AWT EW: ')
			print('Fixed 33: ', round(avg_waiting_time_EW_33[0],2))
			print('Fixed 40: ', round(avg_waiting_time_EW_40[0],2))
			print('Agent: ', round(avg_waiting_time_EW_model[0],2))
			print('Improvement 33: ', round((1 - avg_waiting_time_EW_model/avg_waiting_time_EW_33)[0]*100,2), '%')
			print('Improvement 40: ', round((1 - avg_waiting_time_EW_model/avg_waiting_time_EW_40)[0]*100,2), '%')
			print('\n')
		
	else:
		# avg waiting_time (fixed)
		avg_waiting_time_RANDOM_33 = np.load(fixed_path_33+constants.simu_type_of_random+'/_avg_waiting_time_per_veh.npy')
		avg_waiting_time_RANDOM_40 = np.load(fixed_path_40+constants.simu_type_of_random+'/_avg_waiting_time_per_veh.npy')

		# avg waiting_time (model)
		avg_waiting_time_RANDOM_model = np.load(model_path+constants.simu_type_of_random+'/_avg_waiting_time_per_veh_RANDOM.npy')
		
		print('Compare AWT RANDOM: ')
		print('Fixed 33: ', avg_waiting_time_RANDOM_33)
		print('Fixed 40: ', avg_waiting_time_RANDOM_40)
		print('Agent: ', avg_waiting_time_RANDOM_model)
		print('Improvement 33: ', 1 - avg_waiting_time_RANDOM_model/avg_waiting_time_RANDOM_33)
		print('Improvement 40: ', 1 - avg_waiting_time_RANDOM_model/avg_waiting_time_RANDOM_40)
		print('\n')


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


	# # # 
	# save_chart(low_fix_reward, low_model_reward, comparation_path, 'REWARD_LOW')        
	# save_chart(high_fix_reward, high_model_reward, comparation_path, 'REWARD_HIGH')        
	# save_chart(ns_fix_reward, ns_model_reward, comparation_path, 'REWARD_NS')        
	# save_chart(ew_fix_reward, ew_model_reward, comparation_path, 'REWARD_EW')        
		
	# save_chart(low_fix_delay, low_model_delay, comparation_path, 'DELAY_LOW')        
	# save_chart(high_fix_delay, high_model_delay, comparation_path, 'DELAY_HIGH')        
	# save_chart(ns_fix_delay, ns_model_delay, comparation_path, 'DELAY_NS')        
	# save_chart(ew_fix_delay, ew_model_delay, comparation_path, 'DELAY_EW')        

def main():

	constants.is_test_random = False
	constants.max_steps = 5400

	if sys.argv[2] == "LOW":
		constants.simu_types = [0]
	if sys.argv[2] == "HIGH":
		constants.simu_types = [1]
	if sys.argv[2] == "NS":
		constants.simu_types = [2]
	if sys.argv[2] == "EW":
		constants.simu_types = [3]

	if sys.argv[1] == "agent2":
		constants.is_load_Agent_1 = False
		constants.loaded_model_name = "180_pp2_50met.h5"
	elif sys.argv[1] == "agent1":
		constants.is_load_Agent_1 = True
		constants.loaded_model_name = "190_pp1_50met.h5"
	

	# ---------------------------- CONFIGURATION for EVALUATION -----------------------------------
	model_name = constants.loaded_model_name
	simu_types = constants.simu_types 
	plot_path = constants.plot_path_trained_model
	os.makedirs(os.path.dirname(plot_path + 'LOW/'), exist_ok=True)
	os.makedirs(os.path.dirname(plot_path + 'HIGH/'), exist_ok=True)
	os.makedirs(os.path.dirname(plot_path + 'NS/'), exist_ok=True)
	os.makedirs(os.path.dirname(plot_path + 'EW/'), exist_ok=True)
	os.makedirs(os.path.dirname(plot_path + 'RANDOM_1/'), exist_ok=True)
	os.makedirs(os.path.dirname(plot_path + 'RANDOM_2/'), exist_ok=True)
	os.makedirs(os.path.dirname(plot_path + 'RANDOM_3/'), exist_ok=True)
	os.makedirs(os.path.dirname(plot_path + 'RANDOM_4/'), exist_ok=True)
	os.makedirs(os.path.dirname(plot_path + 'RANDOM_5/'), exist_ok=True)
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
		print('model name: ', constants.loaded_model_name)
	except:
		print('No models found. Please check again! \n')
		time.sleep(100)

	target_agent = DQNAgent(num_actions, batch_size)       	# target agent
	memory = Memory(memory_size)                            # memory
	sumo_simulation = SumoSimulation(agent, target_agent, memory, gamma, max_steps, sumoCmd, green_duration, yellow_duration) # sumo
	# --------------------------------------------------------------------------------------------------------------------------------------
	for simu_type in simu_types:
		plot_path = constants.plot_path_trained_model
		if simu_type == 0:
			plot_path += 'LOW/'
			constants.fixed_neg_reward = np.load(constants.plot_path_fixed_sys +'LOW/delay_array_fix.npy')
		elif simu_type == 1:
			plot_path += 'HIGH/'
			constants.fixed_neg_reward = np.load(constants.plot_path_fixed_sys +'HIGH/delay_array_fix.npy')
		elif simu_type == 2:
			plot_path += 'NS/'
			constants.fixed_neg_reward = np.load(constants.plot_path_fixed_sys +'NS/delay_array_fix.npy')
		elif simu_type == 3:
			plot_path += 'EW/'
			constants.fixed_neg_reward = np.load(constants.plot_path_fixed_sys +'EW/delay_array_fix.npy')
		elif simu_type == 4:
			plot_path += constants.simu_type_of_random
			constants.fixed_neg_reward = np.load(constants.plot_path_fixed_sys + constants.simu_type_of_random +'/delay_array_fix.npy')
			x = constants.plot_path_fixed_sys + constants.simu_type_of_random +'/delay_array_fix.npy'
			print('Path: ', x)

		save_charts(sumo_simulation, plot_path, "X", [1], [1])

		# START: RUN 1 simulation ------------------------------------------------------------------
		sumo_simulation.run_one_episode(simu_type, total_episodes)
		
		simu_type = get_simu_type(simu_type)
		print('End simulation: ', simu_type)
		
		# get data:
		neg_pos_rewards = sumo_simulation._neg_pos_reward_store			# reward/action-step
		summed_waiting_times = sumo_simulation._summed_wait_store		# waiting_time/1second
		# _avg_waiting_time_per_veh = sumo_simulation._avg_waiting_time_per_veh # waiting_time / 1veh


		# plot charts:
		save_charts(sumo_simulation, plot_path, simu_type, neg_pos_rewards, summed_waiting_times)

		# save array

		if simu_type == 'LOW':
			np.save(constants.plot_path_trained_model + 'LOW/_avg_waiting_time_per_veh_LOW.npy', sumo_simulation._avg_waiting_time_per_veh_LOW)
		if simu_type == 'HIGH':
			np.save(constants.plot_path_trained_model + 'HIGH/_avg_waiting_time_per_veh_HIGH.npy', sumo_simulation._avg_waiting_time_per_veh_HIGH)
		if simu_type == 'NS':
			np.save(constants.plot_path_trained_model + 'NS/_avg_waiting_time_per_veh_NS.npy', sumo_simulation._avg_waiting_time_per_veh_NS)
		if simu_type == 'EW':
			np.save(constants.plot_path_trained_model + 'EW/_avg_waiting_time_per_veh_EW.npy', sumo_simulation._avg_waiting_time_per_veh_EW)
		if simu_type == constants.simu_type_of_random:
			np.save(constants.plot_path_trained_model + constants.simu_type_of_random+'/_avg_waiting_time_per_veh_RANDOM.npy', sumo_simulation._avg_waiting_time_per_veh_RANDOM)

	
	
	# plot compare_charts:
	save_comparation()

	# draw:
	if constants.is_test_random:
		agent = "Agent 2"
		if constants.is_load_Agent_1:
			agent = "Agent 1"
		plt.figure('Evaluate simulation: ' + constants.simu_type_of_random + ' by ' + agent)
	else:
		simu_type = "LOW"
		if constants.simu_types[0] == 0:
			simu_type = "LOW"
		if constants.simu_types[0] == 1:
			simu_type = "HIGH"
		if constants.simu_types[0] == 2:
			simu_type = "NS"
		if constants.simu_types[0] == 3:
			simu_type = "EW"

		agent = "Agent 2"
		if constants.is_load_Agent_1:
			agent = "Agent 1"
		plt.figure('Evaluate simulation: ' + simu_type + ' by ' + agent)


	plt.clf()
	plt.rcParams.update({'font.size': 12})
	plt.xlabel('Time line (s)')
	plt.ylabel('Average Waiting Time Per Vehicle (s)')
	
	# plot fixed_array:
	plt.plot(constants.fixed_neg_reward, label='STL (33,33)')

	if constants.is_load_Agent_1:
		my_label = "Agent 1"
	else:
		my_label = "Agent 2"
	plt.plot(sumo_simulation._summed_wait_store, label = my_label)
	plt.legend(loc='upper right')

	fig = plt.gcf()
	fig.set_size_inches(7, 5)
	fig.savefig(constants.comparation_path + '/comparation.png', dpi=96)

	plt.show(block=True)

if __name__ == "__main__":
	main()

