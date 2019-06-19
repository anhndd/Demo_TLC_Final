from xml.dom import minidom
import constants


def cal_numb_generated_veh(simu_type, is_random):

    if is_random:
        if constants.is_4way:
            mydoc = minidom.parse('intersection/result.rou_' + simu_type +'.xml')
        elif constants.is_3way:
            mydoc = minidom.parse('intersection/3ways_random_routes/result.rou_' + simu_type +'.xml')
    else:
        print('You are counting vehicles on FIXED ROUTE, mode: ', simu_type)
        if constants.is_test_random:
            print('Cal number vehicle of ', simu_type)
            mydoc = minidom.parse('intersection/3ways_random_routes/random_routes_for_evaluation/result.rou_' + simu_type +'.xml')
        else:
            if constants.is_4way:
                mydoc = minidom.parse('intersection/fixed_routes/result.rou_' + simu_type +'.xml')
            elif constants.is_3way:
                mydoc = minidom.parse('intersection/3ways_random_routes/fixed_routes/result.rou_' + simu_type +'.xml')
                
    flows = mydoc.getElementsByTagName('flow')
    # all item attributes
    count = 0
    for elem in flows:
        count += int(elem.attributes['number'].value)
    
    return count


# print(cal_numb_generated_veh('LOW', False))
# print(cal_numb_generated_veh('HIGH', False))
# print(cal_numb_generated_veh('NS', False))
# print(cal_numb_generated_veh('RANDOM_1',False))
# print(cal_numb_generated_veh('RANDOM_2',False))
# print(cal_numb_generated_veh('RANDOM_3',False))
# print(cal_numb_generated_veh('RANDOM_4',False))
# print(cal_numb_generated_veh('RANDOM_5',False))
# print(cal_numb_generated_veh('EW', False))