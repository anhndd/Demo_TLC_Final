# This file to generate the comparition between FIXED AVG WT and TRAINING PROCESS
import constants
import numpy as np
import matplotlib.pyplot as plt
import os

# load fixed time 33 avg waiting time:
AWT_L_fixed_33 = np.load(constants.path + 'fixed_system_33/LOW/_avg_waiting_time_per_veh.npy')
AWT_H_fixed_33 = np.load(constants.path + 'fixed_system_33/HIGH/_avg_waiting_time_per_veh.npy')
AWT_NS_fixed_33 = np.load(constants.path + 'fixed_system_33/NS/_avg_waiting_time_per_veh.npy')
AWT_EW_fixed_33 = np.load(constants.path + 'fixed_system_33/EW/_avg_waiting_time_per_veh.npy')

AWT_L_fixed_arr_33 = []
AWT_H_fixed_arr_33 = []
AWT_NS_fixed_arr_33 = []
AWT_EW_fixed_arr_33 = []

# load fixed time 40 avg waiting time:
AWT_L_fixed_40 = np.load(constants.path + 'fixed_system_40/LOW/_avg_waiting_time_per_veh.npy')
AWT_H_fixed_40 = np.load(constants.path + 'fixed_system_40/HIGH/_avg_waiting_time_per_veh.npy')
AWT_NS_fixed_40 = np.load(constants.path + 'fixed_system_40/NS/_avg_waiting_time_per_veh.npy')
AWT_EW_fixed_40 = np.load(constants.path + 'fixed_system_40/EW/_avg_waiting_time_per_veh.npy')

AWT_L_fixed_arr_40 = []
AWT_H_fixed_arr_40 = []
AWT_NS_fixed_arr_40 = []
AWT_EW_fixed_arr_40 = []



# load model (in training process) avg waiting time:
# avg waiting time of RANDOM ROUTE
AWT_L_model = np.load(constants.path+'_avg_waiting_time_per_veh_LOW.npy')
AWT_H_model = np.load(constants.path+'_avg_waiting_time_per_veh_HIGH.npy')
AWT_NS_model = np.load(constants.path+'_avg_waiting_time_per_veh_NS.npy')
AWT_EW_model = np.load(constants.path+'_avg_waiting_time_per_veh_EW.npy')


# # avg waiting time of FIXED ROUTE
# AWT_L_model = np.load(constants.path+'_avg_waiting_time_per_veh_LOW_fixed_route.npy')
# AWT_H_model = np.load(constants.path+'_avg_waiting_time_per_veh_HIGH_fixed_route.npy')
# AWT_NS_model = np.load(constants.path+'_avg_waiting_time_per_veh_NS_fixed_route.npy')
# AWT_EW_model = np.load(constants.path+'_avg_waiting_time_per_veh_EW_fixed_route.npy')



for i in range(len(AWT_EW_model)):
    # 33:
    AWT_L_fixed_arr_33.append(AWT_L_fixed_33[0])    
    AWT_H_fixed_arr_33.append(AWT_H_fixed_33[0])    
    AWT_NS_fixed_arr_33.append(AWT_NS_fixed_33[0])    
    AWT_EW_fixed_arr_33.append(AWT_EW_fixed_33[0])    

    # 40:
    AWT_L_fixed_arr_40.append(AWT_L_fixed_40[0])    
    AWT_H_fixed_arr_40.append(AWT_H_fixed_40[0])    
    AWT_NS_fixed_arr_40.append(AWT_NS_fixed_40[0])    
    AWT_EW_fixed_arr_40.append(AWT_EW_fixed_40[0])    


# save img:
def save_chart(arr_model, arr_fixed_33, arr_fixed_40, type):
    plt.rcParams.update({'font.size': 18})
    plt.title("Average waiting time - " + type)
    plt.plot(arr_model, label='Agent')  
    plt.ylabel("Average waiting time (s)")
    plt.xlabel("Episode-th")
    
    plt.plot(arr_fixed_33, label = 'STL (33,33)')
    plt.plot(arr_fixed_40, label = 'STL (40,40)')

    fig = plt.gcf()
    fig.set_size_inches(20, 11.25)
    plt.legend(loc='upper right')
    os.makedirs(os.path.dirname(constants.comparation_path), exist_ok=True)
    fig.savefig(constants.comparation_path + '/compare_'+ type +'.png', dpi=96)
    plt.show()


save_chart(AWT_L_model, AWT_L_fixed_arr_33, AWT_L_fixed_arr_40, "LOW")
save_chart(AWT_H_model, AWT_H_fixed_arr_33, AWT_H_fixed_arr_40, "HIGH")
save_chart(AWT_NS_model, AWT_NS_fixed_arr_33, AWT_NS_fixed_arr_40, "NS")
save_chart(AWT_EW_model, AWT_EW_fixed_arr_33, AWT_EW_fixed_arr_40, "EW")


