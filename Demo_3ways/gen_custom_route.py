from __future__ import print_function
import os
import time
from random import randint
import constants

def gen_4way_route(route_type):
        if route_type == 'LOW':
                # FROM E: 
                lane_EW = randint(50,100)
                lane_ES = randint(50,100)   # turn right

                # FROM W:
                lane_WE = randint(50,100)
                lane_WN = randint(50,100)   # turn right

                # FROM N:       
                lane_SN = randint(50,100)
                lane_SW = randint(50,100)   # turn right

                # FROM S:
                lane_NS = randint(50,100)
                lane_NE = randint(50,100)   # turn right
        elif route_type == 'HIGH':
                # FROM E: 
                lane_EW = randint(125,175)
                lane_ES = randint(125,175)   # turn right

                # FROM W:
                lane_WE = randint(200,250)
                lane_WN = randint(125,175)   # turn right

                # FROM N:       
                lane_SN = randint(125,175)
                lane_SW = randint(125,175)   # turn right

                # FROM S:
                lane_NS = randint(200,250)
                lane_NE = randint(125,175)   # turn right
        elif route_type == 'NS':
                # FROM E: 
                lane_EW = randint(50,100)
                lane_ES = randint(50,100)   # turn right

                # FROM W:
                lane_WE = randint(50,100)
                lane_WN = randint(50,100)   # turn right

                # FROM N:       
                lane_SN = randint(125,175)
                lane_SW = randint(125,175)   # turn right

                # FROM S:
                lane_NS = randint(200,250)
                lane_NE = randint(125,175)   # turn right
        elif route_type == 'EW':
                # FROM E: 
                lane_EW = randint(125,175)
                lane_ES = randint(125,175)   # turn right

                # FROM W:
                lane_WE = randint(200,250)
                lane_WN = randint(125,175)   # turn right

                # FROM N:       
                lane_SN = randint(50,100)
                lane_SW = randint(50,100)   # turn right

                # FROM S:
                lane_NS = randint(50,100)
                lane_NE = randint(50,100)   # turn right
        elif route_type == 'RANDOM':
                # FROM E: 
                lane_EW = randint(0,300)
                lane_ES = randint(0,300)   # turn right

                # FROM W:
                lane_WE = randint(0,300)
                lane_WN = randint(0,300)   # turn right

                # FROM N:       
                lane_SN = randint(0,300)
                lane_SW = randint(0,300)  # turn right

                # FROM S:
                lane_NS = randint(0,300)
                lane_NE = randint(0,300)  # turn right

        #  TAXI
        numb_taxies_EW = lane_EW 
        numb_taxies_ES = lane_ES
        numb_taxies_WN = lane_WN
        numb_taxies_WE = lane_WE
        numb_taxies_NE = lane_NE
        numb_taxies_NS = lane_NS
        numb_taxies_SW = lane_SW
        numb_taxies_SN = lane_SN

        # BUS-----------------
        numb_bus_EW = lane_EW 
        numb_bus_ES = lane_ES
        numb_bus_WN = lane_WN
        numb_bus_WE = lane_WE
        numb_bus_NE = lane_NE
        numb_bus_NS = lane_NS
        numb_bus_SW = lane_SW
        numb_bus_SN = lane_SN

        # MOTO -----------------
        numb_moto_EW = lane_EW 
        numb_moto_ES = lane_ES
        numb_moto_WN = lane_WN
        numb_moto_WE = lane_WE
        numb_moto_NE = lane_NE
        numb_moto_NS = lane_NS
        numb_moto_SW = lane_SW
        numb_moto_SN = lane_SN


        total_taxies = numb_taxies_ES + numb_taxies_EW + numb_taxies_NE + numb_taxies_NS + numb_taxies_SN + numb_taxies_SW + numb_taxies_WE + numb_taxies_WN
        total_bus = numb_bus_ES + numb_bus_EW + numb_bus_NE + numb_bus_NS + numb_bus_SN + numb_bus_SW + numb_bus_WE + numb_bus_WN
        total_moto = numb_moto_ES + numb_moto_EW + numb_moto_NE + numb_moto_NS + numb_moto_SN + numb_moto_SW + numb_moto_WE + numb_moto_WN
        total_vehicles = total_taxies + total_bus + total_moto
        print('Type: ', route_type, ' || total_vehicles: ', total_vehicles)
        
        with open("intersection/result.rou_"+ route_type +".xml", "w") as routes:
                content = """<?xml version="1.0" encoding="UTF-8"?>
        <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

        <vType id="taxi" vClass="taxi" guiShape="passenger/van" minGap="2" latAlignment="left" lcPushy="1"/>
        <flow id="myflow28" begin="0" end="3200" number=\"""" + str(numb_taxies_SN) +"""\" from="gneE81" to="gneE63" type="taxi"/>
        <flow id="myflow29" begin="0" end="3200" number=\"""" + str(numb_taxies_NS) +"""\" from="gneE64" to="gneE80" type="taxi"/>
        <flow id="myflow30" begin="0" end="3200" number=\"""" + str(numb_taxies_WE) +"""\" from="gneE52" to="gneE20" type="taxi"/>
        <flow id="myflow31" begin="0" end="3200" number=\"""" + str(numb_taxies_EW) +"""\" from="gneE0" to="gneE51" type="taxi"/>

        <flow id="myflow32" begin="0" end="3200" number=\"""" + str(numb_taxies_SW) +"""\" from="gneE81" to="gneE51" type="taxi"/>
        <flow id="myflow34" begin="0" end="3200" number=\"""" + str(numb_taxies_NE) +"""\" from="gneE64" to="gneE20" type="taxi"/>

        <flow id="myflow37" begin="0" end="3200" number=\"""" + str(numb_taxies_ES) +"""\" from="gneE0" to="gneE80" type="taxi"/>
        <flow id="myflow38" begin="0" end="3200" number=\"""" + str(numb_taxies_WN) +"""\" from="gneE52" to="gneE63" type="taxi"/>

        <vType id="bus" vClass="bus" guiShape="bus" minGap="2" length="9.44" width="2.45" height="3.1"
                latAlignment="left" lcPushy="1"/>
        <flow id="myflow" begin="0" end="3200" number=\"""" + str(numb_bus_SN) +"""\" from="gneE81" to="gneE63" type="bus"/>
        <flow id="myflow1" begin="0" end="3200" number=\"""" + str(numb_bus_NS) +"""\" from="gneE64" to="gneE80" type="bus"/>
        <flow id="myflow2" begin="0" end="3200" number=\"""" + str(numb_bus_WE) +"""\" from="gneE52" to="gneE20" type="bus"/>
        <flow id="myflow3" begin="0" end="3200" number=\"""" + str(numb_bus_EW) +"""\" from="gneE0" to="gneE51" type="bus"/>
        <flow id="myflow8" begin="0" end="3200" number=\"""" + str(numb_bus_SW) +"""\" from="gneE81" to="gneE51" type="bus"/>
        <flow id="myflow10" begin="0" end="3200" number=\"""" + str(numb_bus_NE) +"""\" from="gneE64" to="gneE20" type="bus"/>
        <flow id="myflow13" begin="0" end="3200" number=\"""" + str(numb_bus_ES) +"""\" from="gneE0" to="gneE80" type="bus"/>
        <flow id="myflow14" begin="0" end="3200" number=\"""" + str(numb_bus_WN) +"""\" from="gneE52" to="gneE63" type="bus"/>


                <vType id="motorcycle" vClass="motorcycle" minGap="0.5" length="2.034" width="0.74" height="1.152"
                latAlignment="right" lcPushy="1"/>
        <flow id="myflow16" begin="0" end="3200" number=\"""" + str(numb_moto_SN) +"""\" from="gneE81" to="gneE63" type="motorcycle"/>
        <flow id="myflow17" begin="0" end="3200" number=\"""" + str(numb_moto_NS) +"""\" from="gneE64" to="gneE80" type="motorcycle"/>
        <flow id="myflow18" begin="0" end="3200" number=\"""" + str(numb_moto_WE) +"""\" from="gneE52" to="gneE20" type="motorcycle"/>
        <flow id="myflow19" begin="0" end="3200" number=\"""" + str(numb_moto_EW) +"""\" from="gneE0" to="gneE51" type="motorcycle"/>
        <flow id="myflow20" begin="0" end="3200" number=\"""" + str(numb_moto_SW) +"""\" from="gneE81" to="gneE51" type="motorcycle"/>
        <flow id="myflow22" begin="0" end="3200" number=\"""" + str(numb_moto_NE) +"""\" from="gneE64" to="gneE20" type="motorcycle"/>
        <flow id="myflow24" begin="0" end="3200" number=\"""" + str(numb_moto_ES) +"""\" from="gneE0" to="gneE63" type="motorcycle"/>
        <flow id="myflow26" begin="0" end="3200" number=\"""" + str(numb_moto_WN) +"""\" from="gneE52" to="gneE63" type="motorcycle"/>

        </routes> """
                print(content, file = routes)

        os.chdir('intersection/')
        os.system('duarouter -n net.net.xml -r result.rou_'+ route_type +'.xml --randomize-flows -o myroutes.rou_'+ route_type +'.xml')
        os.chdir('../')

def gen_3way_route(route_type):
        if route_type == 'LOW':
                # from RIGHT:
                lane_RT = randint(0,50)
                lane_RB = randint(0,50)
                
                # from TOP:
                lane_TB = randint(0,50)
                lane_TR = randint(0,50)
                
                # from BOT:
                lane_BT = randint(0,50)
                lane_BR = randint(0,50)
        elif route_type == 'HIGH':
                # from RIGHT:
                lane_RT = randint(100,150)
                lane_RB = randint(100,150)
                
                # from TOP:
                lane_TB = randint(100,150)
                lane_TR = randint(100,150)
                
                # from BOT:
                lane_BT = randint(100,150)
                lane_BR = randint(100,150)
        elif route_type == 'NS':
                # from RIGHT:
                lane_RT = randint(0,50)
                lane_RB = randint(0,50)
                
                # from TOP:
                lane_TB = randint(100,150)
                lane_TR = randint(100,150)
                
                # from BOT:
                lane_BT = randint(100,150)
                lane_BR = randint(100,150)
        elif route_type == 'EW':
                # from RIGHT:
                lane_RT = randint(100,150)
                lane_RB = randint(100,150)
                
                # from TOP:
                lane_TB = randint(0,50)
                lane_TR = randint(0,50)
                
                # from BOT:
                lane_BT = randint(0,50)
                lane_BR = randint(0,50)
        elif route_type == 'RANDOM_1' or (route_type == 'RANDOM_2') or (route_type == 'RANDOM_3') or (route_type == 'RANDOM_4') or (route_type == 'RANDOM_5'):
                # from RIGHT:
                lane_RT = randint(0,200)
                lane_RB = randint(0,200)
                
                # from TOP:
                lane_TB = randint(0,200)
                lane_TR = randint(0,200)
                
                # from BOT:
                lane_BT = randint(0,200)
                lane_BR = randint(0,200)

        #  TAXI
        numb_taxi_RT = lane_RT
        numb_taxi_RB = lane_RB

        numb_taxi_TB = lane_TB
        numb_taxi_TR = lane_TR
        
        numb_taxi_BT = lane_BT
        numb_taxi_BR = lane_BR


        # BUS-----------------
        numb_bus_RT = lane_RT
        numb_bus_RB = lane_RB

        numb_bus_TB = lane_TB
        numb_bus_TR = lane_TR
        
        numb_bus_BT = lane_BT
        numb_bus_BR = lane_BR

        # MOTO -----------------
        numb_moto_RT = lane_RT
        numb_moto_RB = lane_RB

        numb_moto_TB = lane_TB
        numb_moto_TR = lane_TR
        
        numb_moto_BT = lane_BT
        numb_moto_BR = lane_BR


        total_taxies = numb_taxi_RT + numb_taxi_RB + numb_taxi_TB + numb_taxi_TR + numb_taxi_BT + numb_taxi_BR
        total_bus = numb_bus_RT + numb_bus_RB + numb_bus_TB + numb_bus_TR + numb_bus_BT + numb_bus_BR
        total_moto = numb_moto_RT + numb_moto_RB + numb_moto_TB + numb_moto_TR + numb_moto_BT + numb_moto_BR
        total_vehicles = total_taxies + total_bus + total_moto
        print('Generate RANDOM_ROUTE, type: ', route_type, ' || total_vehicles: ', total_vehicles)
        
        with open("intersection/"+ constants.fork_path +"/result.rou_"+ route_type +".xml", "w") as routes:
                content = """<?xml version="1.0" encoding="UTF-8"?>
        <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

        <vType id="bus" vClass="bus" guiShape="bus" minGap="2" length="9.44" width="2.45" height="3.1"
                latAlignment="left" lcPushy="1"/>
        <flow id="myflow0" begin="0" end="3000" number=\"""" + str(numb_moto_RT) +"""\" from="right_in" to="top_out" type="bus"/>
        <flow id="myflow1" begin="0" end="3000" number=\"""" + str(numb_moto_RB) +"""\" from="right_in" to="bottom_out" type="bus"/>
        <flow id="myflow2" begin="0" end="3000" number=\"""" + str(numb_moto_TB) +"""\" from="top_in" to="bottom_out" type="bus"/>
        <flow id="myflow3" begin="0" end="3000" number=\"""" + str(numb_moto_BT) +"""\" from="bottom_in" to="top_out" type="bus"/>
        <flow id="myflow4" begin="0" end="3000" number=\"""" + str(numb_moto_BR) +"""\" from="bottom_in" to="right_out" type="bus"/>
        <flow id="myflow5" begin="0" end="3000" number=\"""" + str(numb_moto_TR) +"""\" from="top_in" to="right_out" type="bus"/>

        <vType id="motorcycle" vClass="motorcycle" minGap="0.5" length="2.034" width="0.74" height="1.152"
                latAlignment="right" lcPushy="1"/>
        <flow id="myflow6" begin="0" end="3000" number=\"""" + str(numb_bus_RT) +"""\" from="right_in" to="top_out" type="motorcycle"/>
        <flow id="myflow7" begin="0" end="3000" number=\"""" + str(numb_bus_RB) +"""\" from="right_in" to="bottom_out" type="motorcycle"/>
        <flow id="myflow8" begin="0" end="3000" number=\"""" + str(numb_bus_TB) +"""\" from="top_in" to="bottom_out" type="motorcycle"/>
        <flow id="myflow9" begin="0" end="3000" number=\"""" + str(numb_bus_BT) +"""\" from="bottom_in" to="top_out" type="motorcycle"/>
        <flow id="myflow10" begin="0" end="3000" number=\"""" + str(numb_bus_BR) +"""\" from="bottom_in" to="right_out" type="motorcycle"/>
        <flow id="myflow11" begin="0" end="3000" number=\"""" + str(numb_bus_TR) +"""\" from="top_in" to="right_out" type="motorcycle"/>

        <vType id="taxi" vClass="taxi" guiShape="passenger/van" minGap="2"
                latAlignment="left" lcPushy="1"/>
        <flow id="myflow12" begin="0" end="3000" number=\"""" + str(numb_taxi_RT) +"""\" from="right_in" to="top_out" type="taxi"/>
        <flow id="myflow13" begin="0" end="3000" number=\"""" + str(numb_taxi_RB) +"""\" from="right_in" to="bottom_out" type="taxi"/>
        <flow id="myflow14" begin="0" end="3000" number=\"""" + str(numb_taxi_TB) +"""\" from="top_in" to="bottom_out" type="taxi"/>
        <flow id="myflow15" begin="0" end="3000" number=\"""" + str(numb_taxi_BT) +"""\" from="bottom_in" to="top_out" type="taxi"/>
        <flow id="myflow16" begin="0" end="3000" number=\"""" + str(numb_taxi_BR) +"""\" from="bottom_in" to="right_out" type="taxi"/>
        <flow id="myflow17" begin="0" end="3000" number=\"""" + str(numb_taxi_TR) +"""\" from="top_in" to="right_out" type="taxi"/>

        </routes> """
                print(content, file = routes)

        os.chdir('intersection/3ways_random_routes')
        os.system('duarouter -n net.net.xml -r result.rou_'+ route_type +'.xml --randomize-flows -o myroutes.rou_'+ route_type +'.xml')
        os.chdir('../')
        os.chdir('../')



# gen_3way_route('LOW')
# gen_3way_route('HIGH')
# gen_3way_route('NS')
# gen_3way_route('EW')
# gen_3way_route('LOW')
# gen_4way_route('RANDOM')
# gen_3way_route('RANDOM_1')
# gen_3way_route('RANDOM_2')
# gen_3way_route('RANDOM_3')
# gen_3way_route('RANDOM_4')
# gen_3way_route('RANDOM_5')
