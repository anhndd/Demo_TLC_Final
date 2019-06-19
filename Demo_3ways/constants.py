import numpy as np

# config for training:
isShowGUI = True



# constants:
batch_size = 100
gamma = 0.75					
a_dec = 3.5
num_actions = 2
memory_size = 20000             # old version: 50.000. Revert if bug.
sumoBinary = "/usr/bin/sumo" + ("-gui" if isShowGUI else "")

is_test_random = False

is_3way = True
is_4way = False

# is_3way = False
# is_4way = True


if is_4way == True:

    path = "./model/Final_Model_FixedRoute/"    
    total_episodes = 200
    total_ep_for_epislon = 100

    # CONFIGURATION FOR STL SYSTEM:
    # ver 1: fixed 33:
    plot_path_fixed_sys = path + 'fixed_system_33/'
    durations_of_phases = [33,33] 

    # ver 2:
    # plot_path_fixed_sys = path + 'fixed_system_40/'
    # durations_of_phases = [40,40]
    
    # ver 1: SIMULATION FIXED (H L EW NS)
    # simu_types = [0,1,2,3] 							     # for TRAINED + FIXED 

    # ver 2: RANDOM ROUTE (for Agent 1 and Agent 2)
    simu_types = [4]
    is_test_random = True




    sumoCmd = [sumoBinary, "-c", "intersection/sumoconfig.sumoconfig", "--no-step-log", "true", '--no-warnings','--start']
    light_id = '4628048104'
    fork_path = ""
    max_steps = 5400
    # /---------------------------------------------------
    green_duration = 20
    yellow_duration = 4

    # for testing trained model:
    loaded_model_name = '65_pp1_60met.h5'      # for model system
    is_Agent1 = True
    
    
    
    plot_path_trained_model = path + 'trained_model/'
    green_duration_eval = green_duration                
    yellow_duration_eval = yellow_duration
    fixed_neg_reward = []

elif is_3way == True:
    total_episodes = 200
    total_ep_for_epislon = 40

    
    max_steps = 3500                            
    sumoCmd = [sumoBinary, "-c", "intersection/3ways_random_routes/sumoconfig.sumoconfig", "--no-step-log", "true", '--no-warnings','--start']
    light_id = '6270856337'
    path = "./model/model_3way/"    
    green_duration = 20
    yellow_duration = 4
    fork_path = "/3ways_random_routes"
    simu_types = [0,1,2,3] 							     # for TRAINED + FIXED 

    # ver 33:
    plot_path_fixed_sys = path + 'fixed_system_33/'
    durations_of_phases = [33,33] 

    # ver 40,40:
    # plot_path_fixed_sys = path + 'fixed_system_40/'
    # durations_of_phases = [40,40]
    
    # ver 2: RANDOM ROUTE (for Agent 1 and Agent 2)
    simu_types = [4]
    is_test_random = True

    # simu_type_of_random = 'RANDOM_1'        # never comment.
    # simu_type_of_random = 'RANDOM_2'        # never comment.
    simu_type_of_random = 'RANDOM_3'        # never comment.
    # simu_type_of_random = 'RANDOM_4'        # never comment.
    # simu_type_of_random = 'RANDOM_5'        # never comment.



    # max_steps = 10000

    
    

    # for testing trained model:
    loaded_model_name = 'Agent2_200.h5'      # for model system
    # loaded_model_name = 'Agent1_200.h5'      # for model system

    is_load_Agent_1 = False                 # 






    plot_path_trained_model = path + 'trained_model/'
    green_duration_eval = green_duration                
    yellow_duration_eval = yellow_duration

# compare
comparation_path = path + 'comparition/'