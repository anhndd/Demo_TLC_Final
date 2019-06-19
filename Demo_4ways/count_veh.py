from xml.dom import minidom
import constants

def cal_numb_generated_veh(simu_type, is_random):
    if is_random:
        mydoc = minidom.parse('intersection/result.rou_' + simu_type +'.xml')
    else:
        print('You are counting vehicles on FIXED ROUTE, mode: ', simu_type)
        if constants.is_test_random:
            mydoc = minidom.parse('intersection/random_routes_for_evaluation/result.rou_' + simu_type +'.xml')
        else:
            mydoc = minidom.parse('intersection/fixed_routes/result.rou_' + simu_type +'.xml')


    flows = mydoc.getElementsByTagName('flow')

    # all item attributes
    count = 0
    for elem in flows:
        count += int(elem.attributes['number'].value)
    
    return count

# print(cal_numb_generated_veh('LOW', False))
# print(cal_numb_generated_veh('HIGH', False))
# print(cal_numb_generated_veh('NS', False))
# print(cal_numb_generated_veh('EW', False))
# print(cal_numb_generated_veh('RANDOM', False))
# print(cal_numb_generated_veh('RANDOM_1', False))
# print(cal_numb_generated_veh('RANDOM_2', False))
# print(cal_numb_generated_veh('RANDOM_3', False))
# print(cal_numb_generated_veh('RANDOM_4', False))
# print(cal_numb_generated_veh('RANDOM_5', False))
