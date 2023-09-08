# time of simulation is based on minute
from scipy import stats
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# پاداش بر مبنای حالت متعادل مثلا توسط منشی، سنجیده شود و داده شود
class Helthcare_Simulation:

    def __init__(self, param_number_of_general_Dr, param_number_of_Special_Dr , 
                 param_reservation_ratio, param_severe_proportion,
                 inter_arrival_param, disruption_inter_arrival_param,
                  service_time_param, laboratory_service_time_param,
                  percent_need_laboratory,
                  simulation_time,param_quit = 0.15, num_actions = 6, learning_rate=0.15,
                  discount_factor=0.9, exploration_rate=0.1 , num_states = 50,intervals = [0,5,15],
                  penalties= {'Overtime_work' : 10 , 'Wating_inside_reserve': 0.1 ,
                              'Normal_wating_after_reserve':.2 ,'Reqularizator_reward': 15,
                              'Get_service_with_special':5,'Dont_get_service_ontime': 5*2},
                  threshold_queue_quit = 4, using_q_learning = True):
        
        self.waiting_time_replication_average = []
        self.finishing_customers_replication_average = []
        self.sorted_fel = None
        self.current_event = None
        self.patient = None
        self.service_time_laboratory = None
        self.future_event_list = list()
        self.data = dict()
        self.state = dict()
        self.inter_arrival_param = inter_arrival_param
        self.disruption_inter_arrival_param = disruption_inter_arrival_param
        self.service_time_param = service_time_param
        self.laboratory_service_time_param = laboratory_service_time_param
        self.percent_need_laboratory = percent_need_laboratory
        self.simulation_time = simulation_time
        self.param_severe_proportion = param_severe_proportion
        self.param_reservation_ratio = param_reservation_ratio
        self.param_number_of_general_Dr = param_number_of_general_Dr
        self.param_quit = param_quit
        self.param_number_of_Special_Dr = param_number_of_Special_Dr
        self.threshold_queue_quit = threshold_queue_quit
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        # self.q_table = np.zeros((num_states,1,2, num_actions)) # 1 is types of patient, 2 is number of shifts
        self.intervals = intervals
        shape = (3,) * self.num_actions + (30,7, self.num_actions,) # 3 is interval!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.q_table = np.zeros(shape) # 30 is days of one month, 7 is number of service decks
        self.future_state = dict()
        self.penalties = penalties
        self.using_q_learning = using_q_learning
        
        self.reffering_number_in_month_normal = list()
        self.reffering_number_in_month_seriously = list()
        self.today_date = 0
        
        self.reserve_each_day_normal = dict()
        for i in range(1,31):
            self.reserve_each_day_normal[i] = 0
            
        self.clock = 0
        self.trace_list = []
        


    def data_def(self):
        
    
        
        self.data['patients'] = dict()
        """ 
        patients dictionary is implemented to track each patient's entrance time, service start time, and service end time, 
        service laboratory start time, and service laboratory end time, for instance:{1:[22, 25, 29, null, null]}. It is
        noteworthy to mention that not all the patients use laboratory service so the 4th and 5th elements of each list is set
        to null for initialization.
        """

        # The dictionary below is needed to store the last time for which each queue has changed in length.
        self.data['Last Time Queue Length Changed'] = dict()
        self.data['Last Time Queue Length Changed']['Normal Queue'] = 0
        self.data['Last Time Queue Length Changed']['Seriously Queue'] = 0
        self.data['Last Time Queue Length Changed']['Normal Reservation Queue'] = 0
        self.data['Last Time Queue Length Changed']['Seriously Reservation Queue'] = 0
        self.data['Last Time Queue Length Changed']['Normal laboratory Queue'] = 0
        self.data['Last Time Queue Length Changed']['Seriously laboratory Queue'] = 0

        # The dictionary below is needed to store all patients' data in each queue
        self.data['Queue patients'] = dict()
        self.data['Queue patients']['Normal Queue'] = dict()
        self.data['Queue patients']['Seriously Queue'] = dict()
        self.data['Queue patients']['Normal Reservation Queue'] = dict()
        self.data['Queue patients']['Seriously Reservation Queue'] = dict()
        self.data['Queue patients']['Normal laboratory Queue'] = dict()
        self.data['Queue patients']['Seriously laboratory Queue'] = dict()

        # The dictionary below is needed to store the last length of each queue.
        self.data['Last Queue Length'] = dict()
        self.data['Last Queue Length']['Normal Queue'] = 0
        self.data['Last Queue Length']['Seriously Queue'] = 0
        self.data['Last Queue Length']['Normal Reservation Queue'] = 0
        self.data['Last Queue Length']['Seriously Reservation Queue'] = 0
        self.data['Last Queue Length']['Normal laboratory Queue'] = 0
        self.data['Last Queue Length']['Seriously laboratory Queue'] = 0

        # The dictionary below is needed to store the last time for which each Dr status has been changed.
        self.data['Last Time Dr Status Changed'] = dict()
        self.data['Last Time Dr Status Changed']['Special_Dr'] = 0
        self.data['Last Time Dr Status Changed']['General_Dr'] = 0
        self.data['Last Time Dr Status Changed']['laboratory'] = 0

        # The dictionary below is needed to store the last Dr status.
        self.data['Last Dr Status'] = dict()
        self.data['Last Dr Status']['Special_Dr'] = 0
        self.data['Last Dr Status']['General_Dr'] = 0
        self.data['Last Dr Status']['laboratory'] = 0

        # These crumb data are stored for the purpose that is obviously expressed.
        self.data['Last Time Disruption start'] = -1440
        self.data['Number Of No Waiting Seriously patient'] = 0
        self.data['patients To waiting In laboratory Queue'] = []
        self.data['Number of Seriously patients'] = 0

        # The dictionary below is needed to store the maximum length of each queue during the simulation.
        self.data['Maximum Queue Length'] = dict()
        self.data['Maximum Queue Length']['Normal Queue'] = 0
        self.data['Maximum Queue Length']['Seriously Queue'] = 0
        self.data['Maximum Queue Length']['Normal Reservation Queue'] = 0
        self.data['Maximum Queue Length']['Seriously Reservation Queue'] = 0
        self.data['Maximum Queue Length']['Normal laboratory Queue'] = 0
        self.data['Maximum Queue Length']['Seriously laboratory Queue'] = 0

        # The dictionary below is needed to store the maximum waiting time of patients in each queue during the simulation.
        self.data['Maximum Waiting time'] = dict()
        self.data['Maximum Waiting time']['Normal Queue'] = 0
        self.data['Maximum Waiting time']['Seriously Queue'] = 0
        self.data['Maximum Waiting time']['Normal Reservation Queue'] = 0
        self.data['Maximum Waiting time']['Seriously Reservation Queue'] = 0
        self.data['Maximum Waiting time']['Normal laboratory Queue'] = 0
        self.data['Maximum Waiting time']['Seriously laboratory Queue'] = 0

        # Cumulative statistics that are necessary to assess the system performance measures.
        self.data['Cumulative Stats'] = dict()
        self.data['Cumulative Stats']['Seriously patients System Duration time'] = 0
        self.data['Cumulative Stats']['Number of Seriously patients with no Delay'] = 0
        self.data['Cumulative Stats']['Number of Seriously patients'] = 0

        # This specific dictionary in cumulative stats is assigned to store area under each queue length curve.
        self.data['Cumulative Stats']['Area Under Queue Length Curve'] = dict()
        self.data['Cumulative Stats']['Area Under Queue Length Curve']['Normal Queue'] = 0
        self.data['Cumulative Stats']['Area Under Queue Length Curve']['Seriously Queue'] = 0
        self.data['Cumulative Stats']['Area Under Queue Length Curve']['Normal Reservation Queue'] = 0
        self.data['Cumulative Stats']['Area Under Queue Length Curve']['Seriously Reservation Queue'] = 0
        self.data['Cumulative Stats']['Area Under Queue Length Curve']['Normal laboratory Queue'] = 0
        self.data['Cumulative Stats']['Area Under Queue Length Curve']['Seriously laboratory Queue'] = 0

        # This specific dictionary in cumulative stats is assigned to store area under waiting time for patients in each queue.
        self.data['Cumulative Stats']['Area Under Waiting time'] = dict()
        self.data['Cumulative Stats']['Area Under Waiting time']['Normal Queue'] = 0
        self.data['Cumulative Stats']['Area Under Waiting time']['Seriously Queue'] = 0
        self.data['Cumulative Stats']['Area Under Waiting time']['Normal Reservation Queue'] = 0
        self.data['Cumulative Stats']['Area Under Waiting time']['Seriously Reservation Queue'] = 0
        self.data['Cumulative Stats']['Area Under Waiting time']['Normal laboratory Queue'] = 0
        self.data['Cumulative Stats']['Area Under Waiting time']['Seriously laboratory Queue'] = 0

        # This specific dictionary in cumulative stats is assigned to store area under each Dr busy time.
        self.data['Cumulative Stats']['Area Under Dr Busy time'] = dict()
        self.data['Cumulative Stats']['Area Under Dr Busy time']['General_Dr'] = 0
        self.data['Cumulative Stats']['Area Under Dr Busy time']['Special_Dr'] = 0
        self.data['Cumulative Stats']['Area Under Dr Busy time']['laboratory'] = 0
        
    def get_action(self, state,type_patient ):
        
        temp_num_action_reserve = []
        for j in range(self.today_date + 1,self.today_date + self.num_actions+1):
            if j>30 :
                i = j-30
            else: 
                i = j

            if self.reserve_each_day_normal[i] >= self.intervals[2]:
                temp_num_action_reserve.append( 2 )
            elif self.reserve_each_day_normal[i] < self.intervals[0]:
                temp_num_action_reserve.append( 0 )
            else:
                temp_num_action_reserve.append( 1 )
        temp_num_action_reserve2 = tuple(temp_num_action_reserve)
        

        if np.random.rand() <= self.exploration_rate:
            temp_action = np.random.randint(1,self.num_actions+1)
            if (self.today_date + temp_action)%30 ==0:
                temp_future_day = 30
            else:
                temp_future_day = (self.today_date + temp_action)%30
            self.reserve_each_day_normal[temp_future_day] +=1
            return temp_action,temp_num_action_reserve
        else:
            temp_action = np.argmax(self.q_table[temp_num_action_reserve2][self.today_date-1 , type_patient-1 , :]) # -1 is for starting index from 0    
            temp_action += 1
            
            if (self.today_date + temp_action)%30 ==0:
                temp_future_day = 30
            else:
                temp_future_day = (self.today_date + temp_action)%30
            self.reserve_each_day_normal[temp_future_day] +=1
            return temp_action,temp_num_action_reserve
    # کیو لرنینگ ما دو مرحله ایه و دوبار اپدیت میشه    
    def update_q_value(self, state, type_patient , action, reward, num_action_reserve): 
        # num_action_reserve = !!!!!!!!!!
        #!!!!!!
        num_action_reserve = tuple(num_action_reserve)
            
        if (self.today_date- action) >=1:
            temp_setted_day = self.today_date- action
        else:
            temp_setted_day = 30 + (self.today_date- action)

        current_q = self.q_table[num_action_reserve][temp_setted_day - 1  ,type_patient-1 , action-1] 
        max_next_q = np.max(self.q_table[num_action_reserve][ self.today_date - 1 ,type_patient-1 , :]) # we replace self.today_date - 1  with self.today_date - 1 - action becuase it's changed
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_next_q)
        self.q_table[num_action_reserve][ temp_setted_day - 1 , type_patient-1 , action-1] = new_q
       
        
    def starting_state(self):

        # State variables declaration
        self.state['Normal Queue'] = 0
        self.state['Seriously Queue'] = 0
        self.state['Normal Reservation Queue'] = 0
        self.state['Seriously Reservation Queue'] = 0
        self.state['Special_Dr Dr Status'] = 0
        self.state['General_Dr Dr Status'] = 0
        self.state['laboratory Dr Status'] = 0
        self.state['Seriously laboratory Queue'] = 0
        self.state['Normal laboratory Queue'] = 0
        self.state['Shift Status'] = 0

        # Data: will save every essential data
        self.data_def()
        
        # FEL initialization, and Starting events that initialize the simulation
        self.future_event_list.append({'Event Type': 'Shift Start/End', 'Event Time': 0, 'patient': ''})
        self.future_event_list.append({'Event Type': 'Month Change', 'Event Time': 0, 'patient': ''})
        self.future_event_list.append({'Event Type': 'Day Change', 'Event Time': 0.001, 'patient': ''})# 0.001 is for prevent from interference
        
        self.future_event_list.append({'Event Type': 'Refer Start', 'Event Time': 0, 'patient': [1, 'Normal', 'General_Dr', 0]})
        self.future_event_list.append({'Event Type': 'Refer Start', 'Event Time': 0, 'patient': [2, 'Seriously', 'Special_Dr', 0]})

        return self.state, self.future_event_list, self.data



    def fel_maker(self, event_type: str, patient: list, disruption: str = 'No'):
        """
        This function is supposed to set the next event into future event list
        param future_event_list: list that contains all future events
        param event_type: types of each event that can occur in this simulation
        param clock: simulation clock
        param state: dictionary that contains all state variables
        param patient: a list that filled with each patient's attributes
        param disruption: whether the next event is set in disruption conditions or not
        """
        event_time = 0

        if event_type == 'Refer Start':
            if self.state['Shift Status'] ==1:# نیازی به تغیر ندارد زیرا ویتینگ درون روز لحاظ نمی شود!!!
                if disruption == 'No':  # if the system is not in the disruption condition ...
                    event_time = self.clock + self.exponential(self.inter_arrival_param[self.state['Shift Status']])

                else:
                    event_time = self.clock + self.exponential(self.disruption_inter_arrival_param[self.state['Shift Status']])
            else:
                if disruption == 'No':  # if the system is not in the disruption condition ...
                    event_time = self.clock + 720 + self.exponential(self.inter_arrival_param[self.state['Shift Status']-1]) # -1 because we don't assign inter arival parameter for shift of 2

                else:
                    event_time = self.clock + 720 + self.exponential(self.disruption_inter_arrival_param[self.state['Shift Status']-1])


        elif event_type == 'Refer End':
            # event_time = self.clock + self.exponential(self.service_time_param[patient[2]][patient[1]])
            event_time = self.clock + random.normalvariate(self.service_time_param[patient[1]][patient[4]], 3) 


        elif event_type == 'laboratory Refer End':
            event_time = self.clock + self.exponential(self.laboratory_service_time_param)

        elif event_type == 'Disruption Start':
            event_time = self.clock + 1440 * random.randint(1, 30)

        elif event_type == 'Month Change':
            event_time = self.clock + 30 * 1440
            
        elif event_type == 'Day Change':
            event_time = self.clock + 1440

        elif event_type == 'Queue Quit':
            if patient[1] == 'Normal':  # if a normal patient want to quit from his/her queue ...
                event_time = self.clock + self.uniform(5, max(25, self.state['Normal Queue']))
            else:
                event_time = self.clock + self.uniform(5, max(25, self.state['Seriously Queue']))

        elif event_type == 'Shift Start/End':
            event_time = self.clock + 720
            
        elif event_type == 'Reservation':
            # considering creation of replicate key in this dictionary
            if self.using_q_learning:
                
                if patient[1] == 'Normal':
                    output = self.get_action(self.state['Normal Queue'],type_patient = patient[4] )
                    action,num_action_reserve = output[0], output[1]
                    # 1 is becuase action start from 0 and (self.clock %1440) it is, because we want patient refer in the begening of day
                    event_time = self.clock + (action) *1440 - (self.clock %1440) +np.random.uniform(1,10) # np.random.uniform(1,10) is for showing dely of patient in refering to hospital
                    self.future_state[event_time] = [action ,self.state['Normal Queue'],
                                                     patient[4] ,self.state['Shift Status'],
                                                     num_action_reserve]

                
            else:# assign time in greedy mode or Secretary
                if patient[1] == 'Normal':
                    temp_reserve_each_day_normal = {}
                    for i in range(self.today_date+1, self.today_date + 11):
                        if i ==30:
                            # (+ i-self.today_date) is for nearest day has more chance for choosing by Secretary
                            temp_reserve_each_day_normal[30] = self.reserve_each_day_normal[30] + i-self.today_date
                        else:
                            temp_reserve_each_day_normal[i%30] = self.reserve_each_day_normal[i%30] + i-self.today_date
                    action = min(temp_reserve_each_day_normal,key = temp_reserve_each_day_normal.get)
                    event_time = self.clock + (action) *1440 - (self.clock %1440) +np.random.uniform(1,10) 

                    if (self.today_date + action)%30 ==0:
                        temp_future_day = 30
                    else:
                        temp_future_day = (self.today_date + action)%30
                    self.reserve_each_day_normal[temp_future_day] +=1
                
                else:
                    pass
                    # action = random.randint(1, self.num_actions)
                    # event_time = self.clock + (action)

    

        new_event = {'Event Type': event_type, 'Event Time': event_time, 'patient': patient}
        self.future_event_list.append(new_event)



    @staticmethod
    def exponential(beta: float) -> float:
        """
        param lambda_param: mean parameter of exponential distribution
        return: random variate that conforms to exponential distribution
        """
        r = random.random()
        return -beta * math.log(r)



    @staticmethod
    def uniform(a: float, b: float) -> float:
        """
        param a: lower bound for uniform dist.
        param b: upper bound for uniform dist.
        return: random variate that obey uniform dist.
        """
        r = random.random()
        return a + (b - a) * r



    @staticmethod
    def discrete_uniform(a: int, b: int) -> int:
        """
        param a: lower bound for discrete uniform dist.
        param b: upper bound for discrete uniform dist.
        return: random variate that obey discrete uniform dist.
        """
        r = random.random()
        for inc in range(a, b + 1):
            if (r < (inc + 1) / (b - a + 1)) and (r >= inc / (b - a + 1)):
                print(r)
                return inc



    def data_Dr_calculater(self, name: str):
        """
        This function is supposed to calculate area under each Dr busy time.
        param data: the dictionary that store every essential data
        param name: each Dr name, whether is Special_Dr, General_Dr or laboratory
        """
        self.data['Cumulative Stats']['Area Under Dr Busy time'][name] += self.state['{} Dr Status'.format(name)] \
                                                                * (self.clock - self.data['Last Time Dr Status Changed'][name])
        self.data['Last Time Dr Status Changed'][name] = self.clock



    def data_queue_calculater(self, name: str, temp=0):
        """
        This function is supposed to calculate area under each queue length curve,
        and also the maximum queue length.
        """
        self.data['Cumulative Stats']['Area Under Queue Length Curve']['{} Queue'.format(name)] += (self.state['{} Queue'.format(name)] - temp) \
                                                    * (self.clock - self.data['Last Time Queue Length Changed']['{} Queue'.format(name)])
        self.data['Last Time Queue Length Changed']['{} Queue'.format(name)] = self.clock
        self.data['Maximum Queue Length']['{} Queue'.format(name)] = max(self.data['Maximum Queue Length']['{} Queue'.format(name)],
                                                                    (self.state['{} Queue'.format(name)] - temp + 1))



    def data_queue_patient(self, name: str, status='', laboratory_need='No', has_Reservation='No'):
        parameter = {"No": 1, "Yes": 3}  # to store clock in right position

        if status != 'Exit':
            first_patient_in_queue = min(self.data['Queue patients']['{} Queue'.format(name)],
                                          key=self.data['Queue patients']['{} Queue'.format(name)].get)
            potential = self.data['Queue patients']['{} Queue'.format(name)][first_patient_in_queue][1]
            self.data['patients'][first_patient_in_queue][parameter[laboratory_need]] = self.clock
            self.data['Queue patients']['{} Queue'.format(name)].pop(first_patient_in_queue, None)

            if has_Reservation == 'Yes':
                self.data['patients'][first_patient_in_queue][6] = "it_is_Reservation"
            return first_patient_in_queue, potential

        else:
            self.data['patients'][self.patient[0]][1] = status
            self.data['Queue patients']['{} Queue'.format(name)].pop(self.patient[0], None)

    def Reservation(self):
        self.data['patients'][self.patient[0]][8] = self.clock
        
        if self.state['Shift Status'] ==1:
            temp_Overtime_work = 0
        else:
            temp_Overtime_work = self.penalties['Overtime_work']
            
        if self.patient[1] == 'Normal':  # if a normal patient Refer ...

            if self.state['General_Dr Dr Status'] == self.param_number_of_general_Dr:  # if all General_Dr Dr are busy ...
                if self.state['Special_Dr Dr Status'] < self.param_number_of_Special_Dr:  # if at least one Special_Dr Dr is free ...
                    self.data['Last Dr Status']['Special_Dr'] = self.state['Special_Dr Dr Status']
                    self.data_Dr_calculater('Special_Dr')
                    self.state['Special_Dr Dr Status'] += 1
                    self.patient[2] = 'Special_Dr'
                    self.data['patients'][self.patient[0]][1] = self.clock
                    self.state['Normal Reservation Queue'] -= 1
                    self.fel_maker('Refer End', [self.patient[0], 'Normal', 'Special_Dr', 0,
                                                 self.data['patients'][self.patient[0]][9]])
                    
                    # 5 is for assinging special Dr to Normal patient
                    if self.using_q_learning :
                        reward= -(self.future_state[self.clock][0] * self.penalties['Wating_inside_reserve'] +
                                  self.state['Normal Queue'] * self.penalties['Normal_wating_after_reserve'] +
                                  self.penalties['Get_service_with_special'] + 
                                  temp_Overtime_work * (self.clock%720))+ self.penalties['Reqularizator_reward']  # self.state['Normal Queue'] is in reward becuase normal patient shouldn't be use special dr
                        self.update_q_value(state = self.future_state[self.clock][1], type_patient = self.future_state[self.clock][2] , 
                                             action = self.future_state[self.clock][0],
                                            reward = reward, num_action_reserve= self.future_state[self.clock][4])

                else:  # if all Special_Dr servers are also busy at the time ...
                    self.data['Last Queue Length']['Normal Queue'] = self.state['Normal Queue']
                    self.state['Normal Queue'] += 1
                    self.state['Normal Reservation Queue'] -= 1
                    self.data['Queue patients']['Normal Queue'][self.patient[0]] = [self.clock, 0]
                    
                    self.data_queue_calculater('Normal', temp=1)
                    
                    if self.using_q_learning :
                        # 10 is for don't assinging any Dr to Normal patient
                        reward= -(self.future_state[self.clock][0] * self.penalties['Wating_inside_reserve'] +
                                  self.state['Normal Queue'] * self.penalties['Normal_wating_after_reserve'] +
                                  self.penalties['Dont_get_service_ontime']  + 
                                  temp_Overtime_work * (self.clock%720) ) + self.penalties['Reqularizator_reward']
                        
                        self.update_q_value(state = self.future_state[self.clock][1], type_patient = self.future_state[self.clock][2] , 
                                             action = self.future_state[self.clock][0],
                                            reward = reward,  num_action_reserve= self.future_state[self.clock][4])                    
                    
                    
            else:  # if at least one General_Dr is free ...
                self.data['Last Dr Status']['General_Dr'] = self.state['General_Dr Dr Status']
                self.data_Dr_calculater('General_Dr')
                self.state['General_Dr Dr Status'] += 1
                self.state['Normal Reservation Queue'] -= 1
                self.patient[2] = 'General_Dr'
                self.data['patients'][self.patient[0]][1] = self.clock
                self.fel_maker('Refer End', [self.patient[0], 'Normal', 'General_Dr', 0,
                                             self.data['patients'][self.patient[0]][9]])
                
                if self.using_q_learning :
                    reward= -(self.future_state[self.clock][0] * self.penalties['Wating_inside_reserve'] +
                              temp_Overtime_work * (self.clock%720) ) + self.penalties['Reqularizator_reward']
                    self.update_q_value(state = self.future_state[self.clock][1], type_patient = self.future_state[self.clock][2] , 
                                         action = self.future_state[self.clock][0],
                                        reward = reward,  num_action_reserve= self.future_state[self.clock][4])  
                
        # else:  # if a Seriously patient Refer ...
        #     self.data['Number of Seriously patients'] += 1
        #     self.state['Seriously Reservation Queue'] -= 1
        #     if self.state['Special_Dr Dr Status'] < self.param_number_of_Special_Dr:  # if at least one Special_Dr Dr is free ...
        #         self.data['Last Dr Status']['Special_Dr'] = self.state['Special_Dr Dr Status']
        #         self.data_Dr_calculater('Special_Dr')
        #         self.state['Special_Dr Dr Status'] += 1
        #         self.patient[2] = 'Special_Dr'
        #         self.data['patients'][self.patient[0]][1] = self.clock
        #         self.data['Number Of No Waiting Seriously patient'] += 1
        #         self.fel_maker('Refer End', [self.patient[0], 'Seriously', 'Special_Dr', 0])
                
        #         reward= -(self.future_state[self.clock][0] * 0.2  ) * 2 + 60 # 2 is for seriously patient 
        #         self.update_q_value(state = self.future_state[self.clock][1], type_patient = self.future_state[self.clock][2] , 
        #                              action = self.future_state[self.clock][0],
        #                             reward = reward, next_state = self.state['Seriously Queue'])

        #     else:  # if all Special_Dr servers are busy ...
        #         self.data['Last Queue Length']['Seriously Queue'] = self.state['Seriously Queue']
        #         self.state['Seriously Queue'] += 1
        #         self.data['Queue patients']['Seriously Queue'][self.patient[0]] = [self.clock, 0] 

        #         self.data_queue_calculater('Seriously', temp=1) 
                
        #         reward= -(self.future_state[self.clock][0] * 0.2  + self.state['Seriously Queue'] + 10 ) * 2 + 60# 2 is for seriously patient 
        #         self.update_q_value(state = self.future_state[self.clock][1], type_patient = self.future_state[self.clock][2] , 
        #                              action = self.future_state[self.clock][0],
        #                             reward = reward, next_state = self.state['Seriously Queue'])                  

    


    def refer_start(self):
        """
        This function is supposed to implement Refer start event that is fully described in project's report.
        """
        self.data['patients'][self.patient[0]] = [self.clock,     -1,           -1,         None,
                                                  None,  self.patient[1],   None, 
                                                  None,   None, random.randint(1, 7)]  # -1 means that the patient did not serve
                   # arrival ,start service, end service, start tech, end tech, type patient,
                   #        has laboratory_need, type Dr, 8: time of reserve
                   #  9: number of services deck(random between 1 to 7 with normal distribution)

        if self.patient[1] == 'Normal':  # if a normal patient Refer ...
            if self.state['General_Dr Dr Status'] == self.param_number_of_general_Dr:  # if all General_Dr Dr are busy ...
                if self.state['Special_Dr Dr Status'] < self.param_number_of_Special_Dr:  # if at least one Special_Dr Dr is free ...
                    self.data['Last Dr Status']['Special_Dr'] = self.state['Special_Dr Dr Status']
                    self.data_Dr_calculater('Special_Dr')
                    self.state['Special_Dr Dr Status'] += 1
                    self.patient[2] = 'Special_Dr'
                    self.data['patients'][self.patient[0]][1] = self.clock
                    self.fel_maker('Refer End', [self.patient[0], 'Normal', 'Special_Dr', 0,
                                                 self.data['patients'][self.patient[0]][9]])

                else:  # if all Special_Dr servers are also busy at the time ...
                    Temp = "No_Reservation"
                    self.data['Last Queue Length']['Normal Queue'] = self.state['Normal Queue']
                    self.state['Normal Queue'] += 1
                    self.data['Queue patients']['Normal Queue'][self.patient[0]] = [self.clock, 0]

                    if self.state['Normal Queue'] >= (self.threshold_queue_quit):  # if normal queue length is more than 4 ...
                        if random.random() <= self.param_reservation_ratio:  # according to historical data half of patients will choose to use Reservation option
                            self.data['Last Queue Length']['Normal Queue'] = self.state['Normal Queue']
                            self.state['Normal Queue'] -= 1
                            self.data['Queue patients']['Normal Queue'].pop(self.patient[0], None)
                            self.data['Last Queue Length']['Normal Queue'] = self.state['Normal Reservation Queue']
                            self.state['Normal Reservation Queue'] += 1
                            self.data['Queue patients']['Normal Reservation Queue'][self.patient[0]] = [self.clock, 0]
                            Temp = "Reservation"
                            # print(self.patient[0])
                            self.fel_maker('Reservation', [self.patient[0],'Normal',None,0,
                                                           self.data['patients'][self.patient[0]][9]], disruption='No')

                        else:  # patients that did not use Reservation option
                            if random.random() <= self.param_quit:  # according to historical data, 15% of them will choose to quit after some time
                                self.data['Queue patients']['Normal Queue'][self.patient[0]] = [self.clock, 1]
                                self.fel_maker('Queue Quit', self.patient)

                    if Temp == "Reservation":  # If patient would like to use Reservation option
                        self.data_queue_calculater('Normal Reservation', temp=1)

                    elif Temp == "No_Reservation":  # If patient do not want to use Reservation option
                        self.data_queue_calculater('Normal', temp=1)

            else:  # if at least one General_Dr Dr is free ...
                self.data['Last Dr Status']['General_Dr'] = self.state['General_Dr Dr Status']
                self.data_Dr_calculater('General_Dr')
                self.state['General_Dr Dr Status'] += 1
                self.patient[2] = 'General_Dr'
                self.data['patients'][self.patient[0]][1] = self.clock
                self.fel_maker('Refer End', [self.patient[0], 'Normal', 'General_Dr', 0,
                                             self.data['patients'][self.patient[0]][9]])

        else:  # if a Seriously patient Refer ...
            self.data['Number of Seriously patients'] += 1
            if self.state['Special_Dr Dr Status'] < self.param_number_of_Special_Dr:  # if at least one Special_Dr Dr is free ...
                self.data['Last Dr Status']['Special_Dr'] = self.state['Special_Dr Dr Status']
                self.data_Dr_calculater('Special_Dr')
                self.state['Special_Dr Dr Status'] += 1
                self.patient[2] = 'Special_Dr'
                self.data['patients'][self.patient[0]][1] = self.clock
                self.data['Number Of No Waiting Seriously patient'] += 1
                self.fel_maker('Refer End', [self.patient[0], 'Seriously', 'Special_Dr', 0,
                                             self.data['patients'][self.patient[0]][9]])

            else:  # if all Special_Dr servers are busy ...
                self.data['Last Queue Length']['Seriously Queue'] = self.state['Seriously Queue']
                Temp = "No_Reservation"
                self.state['Seriously Queue'] += 1
                self.data['Queue patients']['Seriously Queue'][self.patient[0]] = [self.clock, 0]

                # should all of seriously patient get service!!!!!
                # if self.state['Seriously Queue'] > (self.threshold_queue_quit + 1):  # if Seriously queue length is more than 4 ...
                #     if False: # each seriously patient should get service in a that day he refered.
                    
                    # if random.random() <= self.param_reservation_ratio:  # according to historical data half of patients will choose to use Reservation option
                          
                        # self.data['Last Queue Length']['Seriously Queue'] = self.state['Seriously Queue']
                        # self.state['Seriously Queue'] -= 1
                        # self.data['Queue patients']['Seriously Queue'].pop(self.patient[0], None)
                        # self.data['Last Queue Length']['Seriously Reservation Queue'] = self.state['Seriously Reservation Queue']
                        # self.state['Seriously Reservation Queue'] += 1
                        # self.data['Queue patients']['Seriously Reservation Queue'][self.patient[0]] = [self.clock, 0]
                        # Temp = "Reservation"

                        # self.fel_maker('Reservation', [self.patient[0],'Seriously',None], disruption='No')
                            
                #     else:  # patients that did not use Reservation option
                #         if random.random() <= self.param_quit:  # according to historical data, 15% of them will choose to quit after some time
                #             self.data['Queue patients']['Seriously Queue'][self.patient[0]] = [self.clock, 1]
                #             self.fel_maker('Queue Quit', self.patient)

                # if Temp == "Reservation":  # If patient would like to use Reservation option
                #     self.data_queue_calculater('Seriously Reservation', temp=1)

                # elif Temp == "No_Reservation":  # If patient do not want to use Reservation option
                #     self.data_queue_calculater('Seriously', temp=1)





    def refer_end(self):
        """
        This function is supposed to implement Refer end event that is fully described in project's report.
        """
        self.data['patients'][self.patient[0]][2] = self.clock  # here we store patient's Refer-end time in patient's dictionary
        self.data['patients'][self.patient[0]][7] = self.patient[2]  # here we store patient's service type in patient's dictionary

        if random.random() < self.percent_need_laboratory:  # according to historical data, 15% of patients need laboratory advice
            if self.state['laboratory Dr Status'] == 2:  # if all laboratory patients are busy at the time ...
                if self.patient[1] == 'Normal':  # if a normal patient wants to use laboratory advice ...
                    self.data['Last Queue Length']['Normal laboratory Queue'] = self.state['Normal laboratory Queue']
                    self.data_queue_calculater('Normal laboratory')
                    self.state['Normal laboratory Queue'] += 1
                    self.data['Queue patients']['Normal laboratory Queue'][self.patient[0]] = [self.clock, 0]
                    self.data['patients To waiting In laboratory Queue'].append(self.patient[0])

                elif self.patient[1] == 'Seriously':  # if a Seriously patient wants to use laboratory advice ...
                    self.data['Last Queue Length']['Seriously laboratory Queue'] = self.state['Seriously laboratory Queue']
                    self.data_queue_calculater('Seriously laboratory')
                    self.state['Seriously laboratory Queue'] += 1
                    self.data['Queue patients']['Seriously laboratory Queue'][self.patient[0]] = [self.clock, 0]
                    self.data['patients To waiting In laboratory Queue'].append(self.patient[0])

            elif self.state['laboratory Dr Status'] < 2:  # if at least one laboratory Dr is free at the time ...
                self.data['Last Dr Status']['laboratory'] = self.state['laboratory Dr Status']
                self.data_Dr_calculater('laboratory')
                self.state['laboratory Dr Status'] += 1
                self.data['patients'][self.patient[0]][3] = self.clock
                self.fel_maker('laboratory Refer End', self.patient)

        if self.patient[2] == 'Special_Dr':  # if the Dr that want to end his/her last Refer is Special_Dr ...
            if self.state['Seriously Queue'] > 0:  # whether the Seriously queue is empty or not ...
                self.data['Last Queue Length']['Seriously Queue'] = self.state['Seriously Queue']
                self.data_queue_calculater('Seriously')
                self.state['Seriously Queue'] -= 1
                first_patient_in_queue, potential = self.data_queue_patient('Seriously')
                self.fel_maker('Refer End', [first_patient_in_queue, 'Seriously', 'Special_Dr',
                                             potential, self.data['patients'][self.patient[0]][9]])

            else:  # Seriously queue is empty ...   
                if self.state['Normal Queue'] == 0:  # if normal patient's queue is empty ...
                    if False:  # if we are in 2nd or 3rd shift of a day 
                    # (self.state['Shift Status'] == 2) or (
                    #         self.state['Shift Status'] == 3) or (self.state['Shift Status'] == 1)
                        if self.state['Seriously Reservation Queue'] > 0:  # if Seriously patient's Reservation queue is not empty ...
                            self.data['Last Queue Length']['Seriously Reservation Queue'] = self.state['Seriously Reservation Queue']
                            self.data_queue_calculater('Seriously Reservation')
                            self.state['Seriously Reservation Queue'] -= 1
                            first_patient_in_queue, potential = self.data_queue_patient('Seriously Reservation', has_Reservation= "Yes")
                            self.fel_maker('Refer End', [first_patient_in_queue, "Seriously", 'Special_Dr',
                                                         potential, self.data['patients'][self.patient[0]][9] ])

                        else:  # if Seriously patient's Reservation queue is empty ...
                            if self.state['Normal Reservation Queue'] > 0:  # whether normal patient's Reservation queue is not empty ...
                                self.data['Last Queue Length']['Normal Reservation Queue'] = self.state['Normal Reservation Queue']
                                self.data_queue_calculater('Normal Reservation')
                                self.state['Normal Reservation Queue'] -= 1
                                first_patient_in_queue, potential = self.data_queue_patient('Normal Reservation', has_Reservation= "Yes")
                                self.fel_maker('Refer End', [first_patient_in_queue, "Normal", 'Special_Dr',
                                                             potential, self.data['patients'][self.patient[0]][9]])

                            else:  # normal patient's Reservation queue is empty too ...
                                self.data['Last Dr Status']['Special_Dr'] = self.state['Special_Dr Dr Status']
                                self.data_Dr_calculater('Special_Dr')
                                self.state['Special_Dr Dr Status'] -= 1

                    else:  # if we are in the 1st shift of the day ...
                        self.data['Last Dr Status']['Special_Dr'] = self.state['Special_Dr Dr Status']
                        self.data_Dr_calculater('Special_Dr')
                        self.state['Special_Dr Dr Status'] -= 1

                elif self.state['Normal Queue'] > 0:  # whether normal patient's queue is not empty ...
                    self.data['Last Queue Length']['Normal Queue'] = self.state['Normal Queue']
                    self.data_queue_calculater('Normal')
                    self.state['Normal Queue'] -= 1
                    first_patient_in_queue, potential = self.data_queue_patient('Normal')
                    self.fel_maker('Refer End', [first_patient_in_queue, 'Normal', 'Special_Dr',
                                                 potential, self.data['patients'][self.patient[0]][9]])

        elif self.patient[2] == 'General_Dr':  # if the Dr that want to end his/her last Refer is General_Dr ...
            if self.state['Normal Queue'] > 0:  # if normal patient's queue is not empty ...
                self.data['Last Queue Length']['Normal Queue'] = self.state['Normal Queue']
                self.data_queue_calculater('Normal')
                self.state['Normal Queue'] -= 1
                first_patient_in_queue, potential = self.data_queue_patient('Normal')
                self.fel_maker('Refer End', [first_patient_in_queue, 'Normal', 'General_Dr',
                                             potential, self.data['patients'][self.patient[0]][9]])

            elif self.state['Normal Queue'] == 0:  # if normal patient's queue is empty ...
                if False:  # if we are in 2nd or 3rd shift of a day
                # (self.state['Shift Status'] == 2) or (
                #         self.state['Shift Status'] == 3) or (self.state['Shift Status'] == 1)
                    if self.state['Normal Reservation Queue'] > 0:  # if Seriously patient's Reservation queue is not empty at the moment ...
                        self.data_queue_calculater('Normal Reservation')
                        self.state['Normal Reservation Queue'] -= 1
                        first_patient_in_queue, potential = self.data_queue_patient('Normal Reservation', has_Reservation= "Yes")
                        self.data['patients'][first_patient_in_queue][1] = self.clock
                        self.fel_maker('Refer End', [first_patient_in_queue, "Normal", 'General_Dr',
                                                     potential, self.data['patients'][self.patient[0]][9]])

                    else:  # if Seriously patient's Reservation queue is empty at the moment ...
                        self.data['Last Dr Status']['General_Dr'] = self.state['General_Dr Dr Status']
                        self.data_Dr_calculater('General_Dr')
                        self.state['General_Dr Dr Status'] -= 1

                else:  # if we are in 1st shift of a day
                    self.data['Last Dr Status']['General_Dr'] = self.state['General_Dr Dr Status']
                    self.data_Dr_calculater('General_Dr')
                    self.state['General_Dr Dr Status'] -= 1



    def laboratory_refer_end(self):
        """
        This function is supposed to implement laboratory Refer end event that is fully described in project's report.
        It is important to mention that laboratory Refer start is planned in Refer end event.
        """
        self.data['patients'][self.patient[0]][4] = self.clock  # here we store patient's laboratory Refer-end time in patient's dictionary

        if self.state['Seriously laboratory Queue'] == 0:  # whether there is no Seriously patient in the laboratory queue ...
            if self.state['Normal laboratory Queue'] == 0:  # if there is also no normal patient in the laboratory queue ...
                self.data['Last Dr Status']['laboratory'] = self.state['laboratory Dr Status']
                self.data_Dr_calculater('laboratory')
                self.state['laboratory Dr Status'] -= 1

            else:  # if there are at least one normal patient in the laboratory queue ...
                self.data['Last Queue Length']['Normal laboratory Queue'] = self.state['Normal laboratory Queue']
                self.data_queue_calculater('Normal laboratory')
                self.state['Normal laboratory Queue'] -= 1
                first_patient_in_queue, potential = self.data_queue_patient('Normal laboratory', laboratory_need="Yes")
                self.fel_maker('laboratory Refer End', [first_patient_in_queue, "", "", potential])

        else:  # whether there are at least one Seriously patient in the laboratory queue ...
            self.data['Last Queue Length']['Seriously laboratory Queue'] = self.state['Seriously laboratory Queue']
            self.data_queue_calculater('Seriously laboratory')
            self.state['Seriously laboratory Queue'] -= 1
            first_patient_in_queue, potential = self.data_queue_patient('Seriously laboratory', laboratory_need="Yes")
            self.fel_maker('laboratory Refer End', [first_patient_in_queue, "", "", potential])



    def disruption_start(self):
        """
        This function ganna store last time disruption occurred, and it's defined only to have one function for each event.
        """
        self.data['Last Time Disruption start'] = self.clock



    def month_change(self): # ممکنه روز و ماه تداخل کنند سر زمان شروع سرویس گیری!!!!!
        """
        This function is supposed to implement month change.
        
        """
        # the number of referring patients in each month:
        def random_generated(max_val = 100):
            # Set the range of possible values for the elements
            min_val = 1
            max_val = max_val
    
            # Set the number of elements in the list
            num_elements = 30
    
            # Generate the list using a loop
            lst = []
            for i in range(num_elements):
               # Calculate the value for this element based on its position in the list
               val = max_val - (max_val - min_val) * i / (num_elements - 1)
               
               # Add a random integer near the calculated value to add some randomness
               lst.append(max(int(random.normalvariate(val, max_val / num_elements)),min_val * 2))
            return lst
        
        self.reffering_number_in_month_normal = random_generated(max_val = 100)
        self.reffering_number_in_month_seriously = random_generated(max_val = 20)
        
        self.today_date = 0
        
        self.fel_maker('Disruption Start', self.patient, disruption='Yes')
        self.fel_maker('Month Change', self.patient)

    def day_change_and_setting_number_of_patients(self):
        """
        This function is supposed to implement month change.
        """
        temp_length = len(self.data['patients'])
        self.today_date = self.today_date + 1
        
        ##
        if self.today_date  ==1:
            temp_yesterday = 30
        else:
            temp_yesterday = self.today_date - 1
        self.reserve_each_day_normal[temp_yesterday] = 0
        ##
        
        # adjust_reffering_day(self):
        temp_refer_normal = self.reffering_number_in_month_normal.pop(0)
        # adding refer normal entering to before reservation
        self.reserve_each_day_normal[self.today_date] += temp_refer_normal
        
        for k in range(temp_refer_normal):
        
            new_user = [temp_length + 1+k, '', '', 0]
            new_user[1] = 'Normal'
    
            self.fel_maker('Refer Start', new_user, disruption='No')
            
        temp_refer_seriously = self.reffering_number_in_month_seriously.pop(0)
        
        for k in range(temp_refer_seriously):
        
            new_user = [temp_length + temp_refer_normal + 1+k, '', '', 0]
            new_user[1] = 'Seriously'
    
            self.fel_maker('Refer Start', new_user, disruption='No')
            
            # if self.clock >= self.data['Last Time Disruption start'] + 1440:  # whether next patient should be scheduled in disruption condition or not
            #     self.fel_maker('Refer Start', new_user, disruption='No')
    
            # else:
            #     self.fel_maker('Refer Start', new_user, disruption='Yes')
        
        self.fel_maker('Day Change', self.patient)


    def queue_quit(self):
        """
        This function is supposed to implement queue quit event for patients that have potential to do so.
        """
        try: # for a rare steedy state, we should using try
            if self.data['patients'][self.patient[0]][1] == -1:  # if it is !=-1 then mean of he start receiving service before quit of queue
                if self.patient[1] == 'Normal':
                    self.data['Last Queue Length']['Normal Queue'] = self.state['Normal Queue']
                    self.data_queue_calculater('Normal')
                    self.state['Normal Queue'] -= 1
                    self.data_queue_patient('Normal', status='Exit')
    
                else:
                    self.data['Number of Seriously patients'] -= 1
                    self.data['Last Queue Length']['Seriously Queue'] = self.state['Seriously Queue']
                    self.data_queue_calculater('Seriously')
                    self.state['Seriously Queue'] -= 1
                    self.data_queue_patient('Seriously', status='Exit')
        except:
            pass


    def shift_start_end(self):
        """
        This function is supposed to implement shift change.
        """
        if self.clock % 1440 < 720:  # if mod(clock, 1440) < 480, this means we are still in first shift
            self.state['Shift Status'] = 1

        # elif (self.clock % 1440 >= 480) and (self.clock % 1440 < 960):  # if 480 < mod(clock, 1440) < 960, this means we are in second shift
        #     self.state['Shift Status'] = 2

        else:
            self.state['Shift Status'] = 2  # if none of the above, so we are in third shift

            # self.state['Shift Status'] = 3  # if none of the above, so we are in third shift

        self.fel_maker('Shift Start/End', self.patient)
        
    

    def simulation(self, trace_creator = False, T0 = 0) -> dict:
        """
        This function is meant to do the simulation by help of introduced events.
        param simulation_time: this project is terminating simulation, so this parameter is simulation end time.
        return: data and state dictionary will be returned after one replication is done.
        """
        self.starting_state()
        self.future_event_list.append({'Event Type': 'Shift Start/End', 'Event Time': self.simulation_time, 'patient': ''})

        while self.clock < self.simulation_time:
            self.sorted_fel = sorted(self.future_event_list, key=lambda x: x['Event Time'])
            self.current_event = self.sorted_fel[0]  # find imminent event
            self.clock = self.current_event['Event Time']  # advance time to current event time
            self.patient = self.current_event['patient']  # find the patient of that event

            if self.clock < self.simulation_time:  # the if block below is ganna Refer proper event function for that event type
                if self.current_event['Event Type'] == 'Refer Start':
                    self.refer_start()

                elif self.current_event['Event Type'] == 'Refer End':
                    self.refer_end()

                elif self.current_event['Event Type'] == 'laboratory Refer End':
                    self.laboratory_refer_end()

                elif self.current_event['Event Type'] == 'Disruption Start':
                    self.disruption_start()

                elif self.current_event['Event Type'] == 'Month Change':
                    self.month_change()

                elif self.current_event['Event Type'] == 'Day Change':
                    self.day_change_and_setting_number_of_patients()
                    
                elif self.current_event['Event Type'] == 'Queue Quit':
                    self.queue_quit()

                elif self.current_event['Event Type'] == 'Shift Start/End':
                    self.shift_start_end()
                    
                elif self.current_event['Event Type'] == 'Reservation':
                    self.Reservation()
                self.future_event_list.remove(self.current_event)

            else:  # if simulation time is passed after simulation end time, so FEL must be cleared
                self.future_event_list.clear()
            
            if trace_creator:  # Tihs code block is supposed to create trace for each current event and append it to the trace list
                trace_data = list(self.state.values())
                trace_data.insert(0, round(self.clock, 3))
                trace_data.insert(0, self.current_event)
                fel_copy = self.sorted_fel.copy()

                while len(fel_copy) > 0:  # Filling trace with events of future event list
                    trace_data.append(list(filter(None, fel_copy.pop(0).values())))
                self.trace_list.append(trace_data)
            if (T0 > 0) and ( (T0-9) < self.clock < T0 ):

                temp_queue_user = {}
                temp_queue = self.data['Queue patients']

                for i in temp_queue.keys():
                    for j in temp_queue[i]:
                        temp_queue_user[j] = self.data['patients'][j]
                        
                temp_busy_user = {}
                for i in self.data['patients']:
                    if (self.data['patients'][i][1] == -1) or (self.data['patients'][i][2] == -1) or ((self.data['patients'][i][3] is not None) and (self.data['patients'][i][4] is None)):
                        temp_busy_user[i] = self.data['patients'][i]
                ##    
                self.data_def()
                ##
                self.data['Queue patients'] = temp_queue
                self.data['patients'] = temp_queue_user 
                for i in temp_busy_user:
                    self.data['patients'][i] = temp_busy_user[i]
                
        return self.data, self.state, self.trace_list



    def trace_excel_maker(self):
        """
        This function is only meant to create a trace excel
        """
        self.simulation(trace_creator=True)
        trace = pd.DataFrame(self.trace_list)

        columns = list(self.state.keys())  # list of excel columns headers
        columns.insert(0, 'Clock')
        columns.insert(1, 'Current Event')
        columns.extend([f'fel{i}' for i in range(1, trace.shape[1] - 11)])  # to add future event list to trace dataframe
        trace = pd.DataFrame(self.trace_list, columns=columns)
        trace.to_excel('C:/patients/Lenovo/Desktop/trace_dataframe.xlsx', engine='xlsxwriter')



    @staticmethod
    def plotting(x, y, waiting_time_moving_replication_average = None, x_label="inter_arrival_param", title ='Normal Queue'):
        if waiting_time_moving_replication_average is not None:  # This part is using for list to have moving average value
            plt.figure(figsize=(10, 7))
            plt.plot(x, y, alpha=0.2,  label ="Real plot")
            plt.plot(x, waiting_time_moving_replication_average, '--', label = "Moving average",linewidth = 2)
            plt.title(title, size=14)
            error = [np.std(y) for _ in range(len(x))]
            z = np.polyfit(x, y, 4)  # This part is essential for shadow on moving average plot
            p = np.poly1d(z)
            z1 = np.polyfit(x, error, 8)
            p1 = np.poly1d(z1)
            plt.fill_between(x, (p(x)-p1(x)/2), (p(x)+p1(x)/2), alpha=0.2)
            plt.xlabel(x_label)
            plt.legend()
            plt.show()
        else:  # This part is using for list to have not moving average value
            plt.figure(figsize=(3, 2))
            plt.plot(x, y, alpha=0.2)
            z = np.polyfit(x, y, 4)  # This part is related for fitting a 4th degree polynomial for our main plot
            p = np.poly1d(z)
            plt.plot(x, p(x), '--')
            plt.title(title, size=14)
            error = [np.std(y) for _ in range(len(x))]
            z1 = np.polyfit(x, error, 8)
            p1 = np.poly1d(z1)
            plt.fill_between(x, (p(x)-p1(x)/2), (p(x)+p1(x)/2), alpha=0.2)
            plt.xlabel(x_label)
            plt.show()


# ======================================================================================







def calculate_kpi(system_config: object) -> dict:
    """
    This function is meant to calculate all KPIs that described in project's report, then it stored them all
    in a dictionary called kpi_results
    return: kpi_results
    """
    data, state, trace_list = system_config.simulation(T0 = 10*24*60)
    cumulative = {"General_Dr": 0, "Special_Dr": 0, "laboratory": 0}

    kpi_results = dict()

    # In order to find number of people in each queue
    kpi_results['Numbers'] = {}
    kpi_results['Numbers']['Seriously Queue'] = 0
    kpi_results['Numbers']['Normal Queue'] = 0
    kpi_results['Numbers']['Seriously laboratory Queue'] = 0
    kpi_results['Numbers']['Normal laboratory Queue'] = 0
    kpi_results['Numbers']['Seriously Reservation Queue'] = 0
    kpi_results['Numbers']['Normal Reservation Queue'] = 0

    # Maximum number of people in each queue
    kpi_results['Max Queue Time'] = {}
    kpi_results['Max Queue Time']['Seriously laboratory Queue'] = 0
    kpi_results['Max Queue Time']['Seriously Queue'] = 0
    kpi_results['Max Queue Time']['Normal laboratory Queue'] = 0
    kpi_results['Max Queue Time']['Normal Queue'] = 0
    kpi_results['Max Queue Time']['Seriously Reservation Queue'] = 0
    kpi_results['Max Queue Time']['Normal Reservation Queue'] = 0

    # Area under queue time curve
    kpi_results['Average Queue Time'] = {}
    kpi_results['Average Queue Time']['Seriously Queue'] = 0
    kpi_results['Average Queue Time']['Normal Queue'] = 0
    kpi_results['Average Queue Time']['Seriously laboratory Queue'] = 0
    kpi_results['Average Queue Time']['Normal laboratory Queue'] = 0
    kpi_results['Average Queue Time']['Seriously Reservation Queue'] = 0
    kpi_results['Average Queue Time']['Normal Reservation Queue'] = 0

    kpi_results['Average of Reserve_sceduling  Time'] = {}
    kpi_results['Average of Reserve_sceduling  Time']['Seriously'] = 0
    kpi_results['Average of Reserve_sceduling  Time']['Normal'] = 0
    for i in data['patients'].keys():  # for each patient:
        if (data['patients'][i][2] != -1) and (data['patients'][i][1] != -1) and (data['patients'][i][1] != "Exit"):  # Which he/she served in the system
            if data['patients'][i][7] == "General_Dr":  # If the patient is General_Dr ...
                cumulative["General_Dr"] += data['patients'][i][2] - data['patients'][i][1]

            else:
                cumulative["Special_Dr"] += data['patients'][i][2] - data['patients'][i][1]

            # arrival ,start service, end service, start tech, end tech, type patient, has call_back, type service
                                        

            if data['patients'][i][5] == "Seriously":  # If the patient is Seriously ...
                
                if data['patients'][i][8] is None:  # If the patient has not Refer back ...
                    kpi_results['Average Queue Time']['Seriously Queue'] += data['patients'][i][1] - data['patients'][i][0]
                    kpi_results['Numbers']['Seriously Queue'] += 1
                    kpi_results['Max Queue Time']['Seriously Queue'] = \
                                        max(kpi_results['Max Queue Time']['Seriously Queue'], (data['patients'][i][1] - data['patients'][i][0]))
                elif data['patients'][i][8] != None:  # If the patient has Refer back ...
                    kpi_results['Average Queue Time']['Seriously Reservation Queue'] += data['patients'][i][1] - data['patients'][i][8]
                    kpi_results['Numbers']['Seriously Reservation Queue'] += 1
                    kpi_results['Max Queue Time']['Seriously Reservation Queue'] = \
                                            max(kpi_results['Max Queue Time']['Seriously Reservation Queue'], (data['patients'][i][1] - data['patients'][i][8]))
                                            
                    kpi_results['Average of Reserve_sceduling  Time']['Seriously'] += (data['patients'][i][8] - data['patients'][i][0])

                if (data['patients'][i][3] is not None) and (data['patients'][i][4] is not None):  # If the patient has laboratory Refer ...
                    kpi_results['Numbers']['Seriously laboratory Queue'] += 1
                    kpi_results['Average Queue Time']['Seriously laboratory Queue'] += data['patients'][i][3] - data['patients'][i][2]
                    kpi_results['Max Queue Time']['Seriously laboratory Queue'] = \
                                            max(kpi_results['Max Queue Time']['Seriously laboratory Queue'], (data['patients'][i][3] - data['patients'][i][2]))
                    cumulative["laboratory"] += data['patients'][i][4] - data['patients'][i][3]

            if data['patients'][i][5] == "Normal":  # If the patient is normal ...
                if data['patients'][i][8] is None:  # If the patient has not Refer back ...
                    kpi_results['Average Queue Time']['Normal Queue'] += data['patients'][i][1] - data['patients'][i][0]
                    kpi_results['Numbers']['Normal Queue'] += 1
                    kpi_results['Max Queue Time']['Normal Queue'] = \
                                            max(kpi_results['Max Queue Time']['Normal Queue'], (data['patients'][i][1] - data['patients'][i][0]))

                elif data['patients'][i][8] != None:  # If the patient has Refer back ...
                    kpi_results['Average Queue Time']['Normal Reservation Queue'] += data['patients'][i][1] - data['patients'][i][8]
                    kpi_results['Numbers']['Normal Reservation Queue'] += 1
                    kpi_results['Max Queue Time']['Normal Reservation Queue'] = \
                                            max(kpi_results['Max Queue Time']['Normal Reservation Queue'], (data['patients'][i][1] - data['patients'][i][8]))

                    kpi_results['Average of Reserve_sceduling  Time']['Normal'] += (data['patients'][i][8] - data['patients'][i][0])
                    

                if (data['patients'][i][3] is not None) and (data['patients'][i][4] is not None):  # If the patient has laboratory Refer ...
                    kpi_results['Numbers']['Normal laboratory Queue'] += 1
                    kpi_results['Average Queue Time']['Normal laboratory Queue'] += data['patients'][i][3] - data['patients'][i][2]
                    kpi_results['Max Queue Time']['Normal laboratory Queue'] = \
                                            max(kpi_results['Max Queue Time']['Normal laboratory Queue'], (data['patients'][i][3] - data['patients'][i][2]))
                    cumulative["laboratory"] += data['patients'][i][4] - data['patients'][i][3]

    # To reach each kpi percentage it is needed to divide the cumulative value of them to its number (count)
    kpi_results['Average Queue Time']['Seriously Queue'] = kpi_results['Average Queue Time']['Seriously Queue'] / \
                                                          (kpi_results['Numbers']['Seriously Queue'] + 1)  # We add one to each cumulative number for worst case scenario where that number is 0, for example when there is no Refer back in system
    kpi_results['Average Queue Time']['Normal Queue'] = kpi_results['Average Queue Time']['Normal Queue'] / \
                                                        (kpi_results['Numbers']['Normal Queue'] + 1)
    kpi_results['Average Queue Time']['Seriously laboratory Queue'] = kpi_results['Average Queue Time']['Seriously laboratory Queue'] / \
                                                                    (kpi_results['Numbers']['Seriously laboratory Queue'] + 1)
    kpi_results['Average Queue Time']['Normal laboratory Queue'] = kpi_results['Average Queue Time']['Normal laboratory Queue'] / \
                                                                  (kpi_results['Numbers']['Normal laboratory Queue'] + 1)
    kpi_results['Average Queue Time']['Seriously Reservation Queue'] = kpi_results['Average Queue Time']['Seriously Reservation Queue'] / \
                                                                  (kpi_results['Numbers']['Seriously Reservation Queue'] + 1)
    kpi_results['Average Queue Time']['Normal Reservation Queue'] = kpi_results['Average Queue Time']['Normal Reservation Queue'] / \
                                                                  (kpi_results['Numbers']['Normal Reservation Queue'] + 1)
                                                                      
                                                    
    kpi_results['Average of Reserve_sceduling  Time']['Normal']  = kpi_results['Average of Reserve_sceduling  Time']['Normal'] / \
                                                                    (kpi_results['Numbers']['Normal Reservation Queue'] + 1)
    kpi_results['Average of Reserve_sceduling  Time']['Seriously']  = kpi_results['Average of Reserve_sceduling  Time']['Seriously'] / \
                                                                    (kpi_results['Numbers']['Seriously Reservation Queue'] + 1)
                                                    
    Dr_number = {"General_Dr": system_config.param_number_of_general_Dr, "Special_Dr": 2, "laboratory": 2}
    kpi_results['Seriously patients time in system duration'] = 0
    kpi_results['Number of Seriously patients in system with no waiting'] = 0

    for i in data['patients'].keys():  # Loop through each patient and calculate some of KPIs
        if (data['patients'][i][2] != -1) and (data['patients'][i][1] != -1) and (data['patients'][i][1] != "Exit"):  # If patient did not quit its queue during waiting time
            if (data['patients'][i][5] == "Seriously") and (data['patients'][i][6] is None):  # If the patient is normal and has not Refer back
                if (data['patients'][i][3] is None) and (data['patients'][i][4] is None):  # If patient has not laboratory Refer
                    kpi_results['Seriously patients time in system duration'] += data['patients'][i][2] - data['patients'][i][0]

                    if (data['patients'][i][1] - data['patients'][i][0]) == 0:  # If patient's waiting time is equal to 0
                        kpi_results['Number of Seriously patients in system with no waiting'] += 1

                elif (data['patients'][i][3] is not None) and (data['patients'][i][4] is not None):  # If patient has laboratory Refer
                    kpi_results['Seriously patients time in system duration'] += data['patients'][i][4] - data['patients'][i][0]

                    if (data['patients'][i][3] - data['patients'][i][2] == 0) and (
                            data['patients'][i][1] - data['patients'][i][0] == 0):
                        kpi_results['Number of Seriously patients in system with no waiting'] += 1

    kpi_results['Seriously patients time in system duration'] = kpi_results['Seriously patients time in system duration'] / \
                    ((kpi_results['Numbers']['Seriously Queue']) + 1)
    kpi_results['Number of Seriously patients in system with no waiting'] = kpi_results['Number of Seriously patients in system with no waiting'] / \
                    ((kpi_results['Numbers']['Seriously Queue']) + 1)
    kpi_results['Average Queue Length'] = {}

    for i in data['Cumulative Stats']['Area Under Queue Length Curve'].keys():  # Average Queue Length calculation
        kpi_results['Average Queue Length'][i] = data['Cumulative Stats']['Area Under Queue Length Curve'][i] / (system_config.simulation_time /2) # in each 2 shift we work one inside one shift

    kpi_results['Max Queue Length'] = {}
    for i in data['Maximum Queue Length'].keys():  # Maximum Queue Length calculation
        kpi_results['Max Queue Length'][i] = data['Maximum Queue Length'][i]

    kpi_results['Dr Utilization'] = {}
    for i in data['Cumulative Stats']['Area Under Dr Busy time'].keys():  # Dr Utilization calculation
        kpi_results['Dr Utilization'][i] = cumulative[i] / (system_config.simulation_time * Dr_number[i])

    return kpi_results


# =============================================================================
 

# =============================================================================


 
# =============================================================================


def warm_up(simulation_time=30*24*60, show_warmup = True,using_q_learning = True , num_of_replications = 10):
    """
    This function is meant to find cold period of specific kpi
    """
    # Initialize parameters
    num_of_replications = num_of_replications
    frame_length = 90# 720 %90 =0
    window_size = 10

    # Set up a data structure to save required outputs in each replication
    finishing_customers_frame_count = dict()  # keys are replications
    waiting_time_frame_aggregate = dict()  # keys are replications
    waiting_time_replication_average = []
    finishing_customers_replication_average = []

    # Function to calculate moving average of a list over a sliding window of length m.
    def moving_average(input_list, m):
        output_list = []
        n = len(input_list)
        for i in range(n):
            output_list.append(sum(input_list[max(i - m // 2, 2 * i - n + 1, 0):min(i + m // 2 + 1, 2 * i + 1, n)]) / (
                    min(i + m // 2, 2 * i, n - 1) - max(i - m // 2, 2 * i - n + 1, 0) + 1))
        return output_list

    # Function to calculate the number of customers who finish getting service in one time-frame
    # frame: [start_time, end_time]
    def calculate_number_of_finishing_customers(start_time, end_time, customers_data):

        number_of_finishing_customers = 0

        for customer in customers_data:
            if (customers_data[customer][2] != -1) and (customers_data[customer][1] != -1) and (customers_data[customer][1] != "Exit"):  # Which he/she served in the system
                if customers_data[customer][5] == "Normal":
                    if customers_data[customer][6] is None:
                        if start_time < customers_data[customer][2] <= end_time:  # Time Service Ends: 2
                            number_of_finishing_customers += 1
                        elif customers_data[customer][2] > end_time:
                            break

        return number_of_finishing_customers

    def calculate_aggregate_queue_waiting_time(start_time, end_time, users_data):

        cumulative_waiting_time = 0
        counter_user = 0 
        for user in users_data:
            if (users_data[user][2] != -1) and (users_data[user][1] != -1) and (users_data[user][1] != "Exit"):  # Which he/she served in the system
                   
                if users_data[user][5] == "Normal":
                    counter_user += 1
                    if users_data[user][8] is None:
                        # if the user has arrived in this time-frame ...
                        if start_time <= users_data[user][0] < end_time:  # Arrival Time: 0
                            # if the user starts getting service in this time-frame...

                            if users_data[user][1] < end_time:  # Time Service Begins: 1
                                cumulative_waiting_time += users_data[user][1] - \
                                                           users_data[user][0]
                            # else if the user will start getting service after this time-frame...
                            else:
                                cumulative_waiting_time += end_time - \
                                                           users_data[user][0]
                        # if the user has arrived before the beginning of this time-frame
                        # but starts getting service during this time-frame...
                        elif start_time < users_data[user][1] < end_time:
                            cumulative_waiting_time += users_data[user][1] - \
                                                       start_time
                        # There might be another (very rare) corner case. What is it? Handle it if you want.
                        elif users_data[user][0] > end_time:
                            break
                    else: # for patient with reservation time!
                        # if the user has arrived in this time-frame ...
                        if start_time <= users_data[user][8] < end_time:  # Arrival Time: 0
                            # if the user starts getting service in this time-frame...

                            if users_data[user][1] < end_time:  # Time Service Begins: 1
                                cumulative_waiting_time += users_data[user][1] - \
                                                           users_data[user][8]
                            # else if the user will start getting service after this time-frame...
                            else:
                                cumulative_waiting_time += end_time - \
                                                           users_data[user][8]
                        # if the user has arrived before the beginning of this time-frame
                        # but starts getting service during this time-frame...
                        elif start_time < users_data[user][1] < end_time:
                            cumulative_waiting_time += users_data[user][1] - \
                                                       start_time
                        # There might be another (very rare) corner case. What is it? Handle it if you want.
                        elif users_data[user][8] > end_time:
                            break                  
                        
        return cumulative_waiting_time/counter_user

    # Just use the frames with full information (drop last 6 frames)
    num_of_frames = simulation_time // frame_length - 2
    x = [i for i in range(1, num_of_frames//2 + 1)]

    for replication in range(1, num_of_replications + 1):
        System_I = Helthcare_Simulation(param_number_of_general_Dr=3,param_number_of_Special_Dr = 2,
                                        param_reservation_ratio=0.9999, param_severe_proportion=0.4, simulation_time=30 *24*60,
                                 inter_arrival_param={1: 1.1}, # inter_arrival_param={1: 1.1, 2: 1.1, 3: 1.1}
                                 disruption_inter_arrival_param={1: 1.1},
                                 service_time_param={"General_Dr": {'Normal':7 },
                                                     "Special_Dr": {'Normal': 3, "Seriously":5 }}, laboratory_service_time_param=10,
                                 percent_need_laboratory=0.15,param_quit = 0,                 
                                 penalties= {'Overtime_work' : 10 , 'Wating_inside_reserve': 0.1 ,
                                'Normal_wating_after_reserve':.2 ,'Reqularizator_reward': 15,
                                'Get_service_with_special':5, 'Dont_get_service_ontime': 5*2},threshold_queue_quit = 4,
                                using_q_learning=using_q_learning)

        simulation_data = System_I.simulation()[0]
        customers_data = simulation_data['patients']
        finishing_customers_frame_count[replication] = []
        waiting_time_frame_aggregate[replication] = []

        # do calculations frame by frame
        for time in range(0, num_of_frames * frame_length, frame_length):
            if time %1440 <720:
                finishing_customers_frame_count[replication].append(
                    calculate_number_of_finishing_customers(time, time + frame_length, customers_data))
    
                waiting_time_frame_aggregate[replication].append(
                    calculate_aggregate_queue_waiting_time(time, time + frame_length, customers_data))

    for i in range(num_of_frames//2):
        average_finishing_customers = 0
        average_waiting_time = 0

        for replication in range(1, num_of_replications + 1):
            average_finishing_customers += finishing_customers_frame_count[replication][i] * (1 / num_of_replications)
            average_waiting_time += waiting_time_frame_aggregate[replication][i] * (1 / num_of_replications)

        finishing_customers_replication_average.append(average_finishing_customers)
        waiting_time_replication_average.append(average_waiting_time)

    # we are replaced moving average with regression equation
    finishing_customers_moving_replication_average = moving_average(finishing_customers_replication_average, window_size)
    waiting_time_moving_replication_average = moving_average(waiting_time_replication_average, window_size)

    # temp_show_warmup is for deleting cold period(warm up)
    if show_warmup:
        temp_show_warmup = 0
    else:
        temp_show_warmup = len(waiting_time_replication_average)//3
    System_I.plotting(x[temp_show_warmup:], waiting_time_replication_average[temp_show_warmup:],
                      waiting_time_moving_replication_average[temp_show_warmup:], x_label ="Frame(No)", title ='waiting_time_average for patient in each frame')
    System_I.plotting(x[temp_show_warmup:], finishing_customers_replication_average[temp_show_warmup:],
                      finishing_customers_moving_replication_average[temp_show_warmup:], x_label ="Frame(No)", title ='number of finishing_customers_average in each frame')
    return waiting_time_frame_aggregate #!!!!!


# =============================================================================

System_I = Helthcare_Simulation(param_number_of_general_Dr=3,param_number_of_Special_Dr = 2,
                                param_reservation_ratio=0.9999, param_severe_proportion=0.4, simulation_time=300 *24*60,
                         inter_arrival_param={1: 1.1}, # inter_arrival_param={1: 1.1, 2: 1.1, 3: 1.1}
                         disruption_inter_arrival_param={1: 1.1},
                         service_time_param={"Normal": {1:7 , 2:8, 3:10 , 4:15, 5:26, 6:42 ,7: 58  },
                                             "Seriously": {1:9 , 2:10, 3:13 , 4:15, 5:26, 6:42 ,7: 58 }},
                         laboratory_service_time_param=10,
                         percent_need_laboratory=0,param_quit = 0,                 
                         penalties= {'Overtime_work' : 10 , 'Wating_inside_reserve': 0.1 ,
                        'Normal_wating_after_reserve':.2 ,'Reqularizator_reward': 15,
                        'Get_service_with_special':5, 'Dont_get_service_ontime': 5*2},threshold_queue_quit = 0,
                        using_q_learning= True)


# =============================================================================
data, state,trace_list = System_I.simulation()
# kpi2 = calculate_kpi(System_I)
# q_table = System_I.q_table
# v = q_table[:, 1, 1, 1, 1, 1,5,1,:]
# vv= System_I.future_state

# print(np.argmax(v, axis=1))
# for i in range(1000):
#     if data['patients'][i+1][8] != None:
#         print(data['patients'][i+1])
        
        
# for i in range(1000):
#     if (data['patients'][i+1][5] == 'Seriously') and (data['patients'][i+1][8] != None):
#         print(data['patients'][i+1])       

# =============================================================================
waiting_time_frame_aggregate = warm_up(show_warmup= False,using_q_learning = False , num_of_replications = 1)
