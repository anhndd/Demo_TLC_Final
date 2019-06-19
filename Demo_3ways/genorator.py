from __future__ import print_function
import time
import constants
import gen_custom_route

# gen 4ways || 3ways
def gen_route(episode, is_random):
    if episode % 4 == 0:
        simu_type = "LOW"
    elif episode % 4 == 1:
        simu_type = "HIGH"
    elif episode % 4 == 2:
        simu_type = "NS"
    elif episode % 4 == 3:
        simu_type = "EW"

    if constants.is_test_random:
        simu_type = constants.simu_type_of_random

    fork_path = constants.fork_path
    
    # gen RADOM_ROUTE 
    if is_random:
        # 4-ways:
        if constants.is_4way:
            gen_custom_route.gen_4way_route(simu_type)
            route_path = """<route-files value="myroutes.rou_"""+simu_type+""".xml"/>"""
        # 3-ways:
        elif constants.is_3way:
            gen_custom_route.gen_3way_route(simu_type)
            route_path = """<route-files value="myroutes.rou_"""+simu_type+""".xml"/>"""
    
    # load FIXED_ROUTE
    else:
        print('Load fixed route, mode: ', simu_type)
        route_path = """<route-files value="fixed_routes/myroutes.rou_"""+simu_type+""".xml"/>"""
        if constants.is_test_random:
            print('Load fixed route, mode: ', simu_type)
            route_path = """<route-files value="random_routes_for_evaluation/myroutes.rou_"""+simu_type+""".xml"/>"""

    with open("intersection"+fork_path+"/sumoconfig.sumoconfig", "w") as routes:
        print("""<?xml version="1.0" encoding="UTF-8"?>
        <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
            <input>
                <net-file value="net.net.xml"/>
                    """, file = routes)
        print(route_path, file = routes)
        print("""    </input>
            <time>
                <begin value="0"/>
                <end value="20000"/>
            </time>
            <processing>
                <lateral-resolution value="0.875"/>
            </processing>
            <report>
                <xml-validation value="never"/>
                <duration-log.disable value="true"/>
                <no-step-log value="true"/>
            </report>

        </configuration> """, file = routes)
    
    if constants.is_test_random:
        return 4
    return episode % 4


