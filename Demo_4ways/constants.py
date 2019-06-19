import numpy as np

# Control training process:
isShowGUI = True
duration = 33   #STL duration

is_3way = False
is_4way = True

is_test_random = True
simu_type_of_random = 'RANDOM_3'

# for testing trained model:
is_load_Agent_1 = False
loaded_model_name = '180_pp2_50met.h5'      # for (RANDOM)
    


# constants:
batch_size = 100
gamma = 0.75					
a_dec = 3.5
num_actions = 2 
memory_size = 20000
sumoBinary = "/usr/bin/sumo" + ("-gui" if isShowGUI else "")



if is_4way == True:
    if is_test_random:
        simu_types = [4]
    else:
        simu_types = [0,1,2,3]
            
    path = "./model/Final_Model_FixedRoute/"                        # path for saving alls.
    plot_path_fixed_sys = path + 'fixed_system_'+str(duration)+'/'
    durations_of_phases = [duration,duration]
    max_steps = 5400
    

    sumoCmd = [sumoBinary, "-c", "intersection/sumoconfig.sumoconfig", "--no-step-log", "true", '--no-warnings','--start']
    light_id = '4628048104'
    fork_path = ""
    green_duration = 20
    yellow_duration = 4
    plot_path_trained_model = path + 'trained_model/'
    green_duration_eval = green_duration                
    yellow_duration_eval = yellow_duration
    fixed_neg_reward = []
    total_episodes = 200
    total_ep_for_epislon = 100

elif is_3way == True:
    total_episodes = 200
    total_ep_for_epislon = 100
    
    max_steps = 3500                            
    sumoCmd = [sumoBinary, "-c", "intersection/fork/sumoconfig.sumoconfig", "--no-step-log", "true", '--no-warnings','--start']
    light_id = '6270856337'
    path = "./model/model_3way/"    
    green_duration = 20
    yellow_duration = 4
    fork_path = "/fork"
    durations_of_phases = [33,33]                           # for fixed system
    simu_types = [0,1,2,3] 							     # for TRAINED + FIXED 
    # 0: LOW, 1:HIGH, 3: NS 4:EW
    plot_path_fixed_sys = path + 'fixed_system/'

    # for testing trained model:
    loaded_model_name = 'last_model_3way.h5'      # for model system
    plot_path_trained_model = path + 'trained_model/'
    green_duration_eval = green_duration                
    yellow_duration_eval = yellow_duration

# compare
comparation_path = path + 'comparition/'