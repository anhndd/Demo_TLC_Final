
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import count_veh

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import random
import numpy as np
import math
import time
import constants
import timeit
import matplotlib.pyplot as plt
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to kill warning about tensorflow
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

from keras.layers import Input, Conv2D, Flatten, Dense, LeakyReLU, Average, Add, Dot, Subtract, Multiply
from keras.models import Model
from keras.optimizers import Adam
import genorator

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

class SumoSimulation:
	def __init__(self, agent, target_agent, memory, gamma, max_steps, sumoCmd, yellow_duration):
		self._yellow_duration = yellow_duration
		self._max_steps = max_steps
		self._sumoCmd = sumoCmd
		self.list_lane_ids = ['gneE86_1','gneE86_2','gneE86_3','gneE85_1','gneE85_2','gneE85_3','gneE21_1','gneE21_2','gneE21_3','gneE89_1','gneE89_2','gneE89_3']

		self._neg_pos_reward_store = []                             # list reward/ action-step
		self._summed_wait_store = []                       			# list của [tổng thời gian chờ của các xe tại thời điểm t]
		self._carID_that_waited = []
		self._carID_aheaddx = []
		self._carID_sx = []
		self._car_gen_step = []
		self._n_cars_generated = 0

        # performance evaluation:
		self._reward_store = []
		self._cumulative_wait_store = []
		self._avg_intersection_queue_store = []
		self._avg_waiting_time_per_veh = []

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

		# performance evaluation:
		self._reward_store = []
		self._cumulative_wait_store = []
		self._avg_intersection_queue_store = []
		self._avg_waiting_time_per_veh = []
	#  int to string
	def get_action_name(self,action):
		if action == 0:
			return 'EW        '
		elif action == 1:
			return 'EW-Y      ' 
		elif action == 2:
			return 'NS        ' 
		elif action == 3:
			return 'NS-Y      '

	# Select next action in fixed-durations.
	def select_fixed_action(self,action):
		if action == 3:
			return 0
		return action+1

	# Run 1 simulation:
	def run_fixed_duration(self, simu_type, durations_of_phases):

		traffic_code_mode = genorator.gen_route(simu_type, is_random = False)								# gen route file.       

		print('Fixed route, mode: ', self.get_simu_type_str(simu_type))
		traci.start(self._sumoCmd)														# then, start sumo

		traci.gui.setZoom('View #0',1500)
		traci.gui.setOffset('View #0',595,916)

		# reset everything:
		self.reset()
		self._steps = 0			
		self._sum_intersection_queue = 0            
		tot_neg_reward = 0
		pre_wait_time = 0		
		action = 0						# initial action

		# run 1 simulation (1h30m)
		# while self._steps < self._max_steps:
		while (traci.simulation.getMinExpectedNumber() > 0):
			# reset current_wait_time:
			current_wait_time = 0
			#  ============================================================ Perform action ======================
			# 0: EW green   1: EW yellow
			# 2: NS  green  3: NS Yellow
			if action == 0:
				self._set_green_phase(action)
				current_wait_time = self._simulate(durations_of_phases[0])
			elif action == 1:
				self._set_yellow_phase(0)
				current_wait_time = self._simulate(self._yellow_duration)			
			elif action == 2:
				self._set_green_phase(1)
				current_wait_time = self._simulate(durations_of_phases[1])
			elif action == 3:
				self._set_yellow_phase(1)
				current_wait_time = self._simulate(self._yellow_duration)
			#  =================================================================================================================
				
			reward = pre_wait_time - current_wait_time
			self._neg_pos_reward_store.append(reward)

			if reward < 0:
				tot_neg_reward += reward
				
			# reassign:
			pre_wait_time = current_wait_time

			# next action (phase):
			action = self.select_fixed_action(action)

			# print every step:
			# print('step: ', self._steps, ' || action: ',  self.get_action_name(action), ' || negative reward: ', tot_neg_reward)


		self._save_stats(traffic_code_mode, tot_neg_reward)		# mode LOW + total neg-REWARD

		print("Total negative reward: {}, Total_waiting_time: {}, AWT: {}".format(tot_neg_reward, self._sum_intersection_queue, self._avg_waiting_time_per_veh))


		mode = self.get_simu_type_str(traffic_code_mode)
		log_path = constants.plot_path_fixed_sys + mode + '/'
		os.makedirs(os.path.dirname(log_path), exist_ok=True)
		log = open(log_path +'tog_neg_reward.txt', 'a')  # open file text.
		text = mode + ' reward: ' + str(tot_neg_reward) + ' _sum_intersection_queue: ' + str(self._sum_intersection_queue) + ' || AWT: ' + str(self._avg_waiting_time_per_veh) + '\n'
		log.write(text)
		log.close()

		# close gui.
		traci.close(wait = False)


	def plot_delay(self,data):
		if constants.is_test_random:
			plt.figure('Evaluate simulation: ' + constants.simu_type_of_random)
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

			stlName = "STL(" + str(constants.duration) + ',' + str(constants.duration) + ')'
			plt.figure('Evaluate simulation: ' + simu_type + ' by ' + stlName)

		plt.clf()
		plt.xlabel('Time line (s)')
		plt.ylabel('Average Waiting Time Per Vehicle (s)')
		

		stl_label = "STL(" + str(constants.duration) + ',' + str(constants.duration) + ')'
		plt.plot(data, label = stl_label)
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

			# plot this list?: _summed_wait_store
			if self._steps%100 == 0:
				self.plot_delay(self._summed_wait_store)
			self._sum_intersection_queue += intersection_queue
		return return_value

	# RETRIEVE THE STATS OF THE SIMULATION FOR ONE SINGLE STEP
	# return every second
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
		self._reward_store.append(tot_neg_reward) # how much negative reward in this episode
		self._cumulative_wait_store.append(self._sum_intersection_queue) # total number of seconds waited by cars (1 step = 1 second -> 1 car in queue/step = 1 second in queue/step=
		self._avg_intersection_queue_store.append(self._sum_intersection_queue / self._max_steps) # average number of queued cars per step, in this episode

		numb_generated_veh = count_veh.cal_numb_generated_veh(self.get_simu_type_str(traffic_code), is_random = False)
		self._avg_waiting_time_per_veh.append(self._sum_intersection_queue / numb_generated_veh)

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

def save_charts(sumo_simu, plot_path, simu_type, neg_pos_rewards, summed_waiting_times, _avg_waiting_time_per_veh):
	plot_path += sumo_simu.get_simu_type_str(simu_type) + '/'
	os.makedirs(os.path.dirname(plot_path), exist_ok=True)
	
	# save reward chart:
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
	fig.savefig(plot_path + '/reward_fix.png', dpi=96)
	plt.close("all")
	# save np array:
	np.save(plot_path+'reward_array_fix',np.array(data))

	# save waiting time chart:
	data = summed_waiting_times
	plt.plot(data)
	plt.ylabel("Waiting Time (s)")
	plt.xlabel("Timeline (s)")
	plt.margins(0)
	min_val = min(data)
	max_val = max(data)
	plt.ylim(min_val + 0.05 * min_val, max_val + 0.05 * max_val)
	plt.xlim(0, len(data))
	fig = plt.gcf()
	fig.set_size_inches(20, 11.25)
	fig.savefig(plot_path + 'delay_fix.png', dpi=96)
	plt.close("all")	
	# save np array:
	np.save(plot_path+'delay_array_fix',np.array(data))
	

	data = _avg_waiting_time_per_veh
	np.save(plot_path+'_avg_waiting_time_per_veh', np.array(data))

def show_infor_fixed():

	# load AWT:
	if not constants.is_test_random:

		if sys.argv[2] == "LOW":
			AWT_L_fixed = np.load(constants.plot_path_fixed_sys+'LOW/_avg_waiting_time_per_veh.npy')
			print('LOW AWT:  ', round(AWT_L_fixed[0],2))
		if sys.argv[2] == "HIGH":
			AWT_H_fixed = np.load(constants.plot_path_fixed_sys+'HIGH/_avg_waiting_time_per_veh.npy')
			print('HIGH AWT: ', round(AWT_H_fixed[0],2))
		if sys.argv[2] == "NS":
			AWT_NS_fixed = np.load(constants.plot_path_fixed_sys+'NS/_avg_waiting_time_per_veh.npy')
			print('NS AWT:   ', round(AWT_NS_fixed[0],2))
		if sys.argv[2] == "EW":
			AWT_EW_fixed = np.load(constants.plot_path_fixed_sys+'EW/_avg_waiting_time_per_veh.npy')
			print('EW AWT:   ', round(AWT_EW_fixed[0],2))
		
	else:
		AWT_CUSTOM_fixed = np.load(constants.plot_path_fixed_sys+ constants.simu_type_of_random + '/_avg_waiting_time_per_veh.npy')
		print('\n\n 1. Average waiting time' + constants.simu_type_of_random  + 'simulation (STL system): ')
		print('RANDOM AWT: ', AWT_CUSTOM_fixed[0])		


	# neg reward:???

def main():
	
	# .....
	constants.is_test_random = False
	constants.max_steps = 5400

	if sys.argv[1]:
		constants.duration = sys.argv[1]

	if sys.argv[2] == "LOW":
		constants.simu_types = [0]
	if sys.argv[2] == "HIGH":
		constants.simu_types = [1]
	if sys.argv[2] == "NS":
		constants.simu_types = [2]
	if sys.argv[2] == "EW":
		constants.simu_types = [3]


	# ---------------------------- CONFIGURATION for EVALUATION FIXED DURATIONS -----------------------------------
	simu_types = constants.simu_types								# L H NS EW  or RANDOM simulations.
	durations_of_phases = [int(constants.duration),int(constants.duration)]
	
	plot_path = constants.path + 'fixed_system_'+str(constants.duration)+'/'
	constants.plot_path_fixed_sys = plot_path

	os.makedirs(os.path.dirname(plot_path), exist_ok=True)
	yellow_duration = constants.yellow_duration
	max_steps = constants.max_steps
	# ---------------------------- CONFIGURATION for EVALUATION FIXED DURATIONS -----------------------------------

	# others (not edit)
	batch_size = constants.batch_size
	gamma = constants.gamma			
	memory_size = constants.memory_size
	num_actions = constants.num_actions
	sumoCmd = constants.sumoCmd
	agent = DQNAgent(num_actions, batch_size)              # primary agent
	target_agent = DQNAgent(num_actions, batch_size)       # target agent
	memory = Memory(memory_size)                             # memory
	sumo_simulation = SumoSimulation(agent, target_agent, memory, gamma, max_steps, sumoCmd, yellow_duration) # sumo

	# 
	print('Testing STL: ', durations_of_phases)

	# RUN 4 type or RANDOM:
	for simu_type in simu_types:

		# START: RUN 1 simulation ------------------------------------------------------------------
		sumo_simulation.run_fixed_duration(simu_type, durations_of_phases)
		print('End simulation: ')
		
		# get data:
		neg_pos_rewards = sumo_simulation._neg_pos_reward_store			# reward/action-step
		summed_waiting_times = sumo_simulation._summed_wait_store		# waiting_time / 1second
		_avg_waiting_time_per_veh = sumo_simulation._avg_waiting_time_per_veh # waiting_time / 1veh
		# log:
		plt.close("all")
		save_charts(sumo_simulation, plot_path, simu_type, neg_pos_rewards, summed_waiting_times, _avg_waiting_time_per_veh)
		

	# If you have evaluated (don't run 4 simu again), run this one:
	show_infor_fixed()



	if constants.is_test_random:
		plt.figure('Evaluate simulation: ' + constants.simu_type_of_random)
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

		stlName = "STL(" + str(constants.duration) + ',' + str(constants.duration) + ')'
		plt.figure('Evaluate simulation: ' + simu_type + ' by ' + stlName)


	plt.clf()
	plt.rcParams.update({'font.size': 12})
	plt.xlabel('Time line (s)')
	plt.ylabel('Average Waiting Time Per Vehicle (s)')
	
	stl_label = "STL(" + str(constants.duration) + ',' + str(constants.duration) + ')'
	plt.plot(sumo_simulation._summed_wait_store, label = stl_label)
	plt.legend(loc='upper right')
	plt.pause(0.001)  # pause a bit so that plots are updated
	plt.show(block=True)

if __name__ == "__main__":
	main()

