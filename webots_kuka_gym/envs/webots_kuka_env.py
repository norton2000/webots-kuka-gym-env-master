
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from limits import filter_limits, scale_to_joints
from learning_parameters import *

# Import from webots classes
from controller import Supervisor, Motor, Camera


class WebotsKukaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  #Added

        #self.desired_pos = np.array([0.42, 1.0, 0])
        self.desired_pos = np.array([-0.3, 0.43, 0.103])

        self._supervisor = Supervisor()
        self._timestep = 64

        self._touch = 0
        self._num_rew = 0
        # List of objects to be taken into consideration
        self._objects_names = []
        self._objects = {}
        #self.finger_names = "FINGER"
        #self.finger = self._supervisor.getFromDef("FINGER")
        self.finger_names = ["TouchSensor1", "TouchSensor2"]
        self.fingers = {}

        for fin in self.finger_names:
            self.fingers[fin] = self._supervisor.getFromDef(fin)


        # List of links name
        self._link_names = ["arm1", "arm2", "arm3", "arm4", "arm5", "finger1", "finger2"]

        # Get motors
        self._link_objects = {}
        for link in self._link_names:
            self._link_objects[link] = self._supervisor.getMotor(link)
        
        #Added
        self._touch_sensor_names = ["touch sensor1" , "touch sensor2"]
         #Added
        self._touch_sensor_object = {}
        for sensor in self._touch_sensor_names:
            self._touch_sensor_object[sensor] = self._supervisor.getTouchSensor(sensor)
            self._touch_sensor_object[sensor].enable(10)

        # Get sensors
        self._link_position_sensors = {}
        self._min_position = {}
        self._max_position = {}
        for link in self._link_names:
            self._link_position_sensors[link] = self._link_objects[link].getPositionSensor()
            self._link_position_sensors[link].enable(self._timestep)
            self._min_position[link] = self._link_objects[link].getMinPosition()
            self._max_position[link] = self._link_objects[link].getMaxPosition()

        self._supervisor.step(self._timestep)

        # self.camera = self._supervisor.getCamera("camera");
        # self.camera.enable(self._timestep);


###### UTIL FUNCTIONS - START ######

    def get_links_num(self):
        return len(self._link_names)

    def set_timestep_simulation(self, step):
        self._timestep = step

    def set_objects_names(self, names):
        self._objects_names = names
        for obj in self._objects_names:
            self._objects[obj] = self._supervisor.getFromDef(obj)

    #Added
    def set_finger_name(self, name):
        self.finger_names = name
        self.finger = self._supervisor.getFromDef(name)

    def get_link_positions(self):
        positions = []
        for link in self._link_names:
            positions.append(self._link_position_sensors[link].getValue())
        return np.array(positions)

    def set_link_positions(self, positions):
        #assert(len(positions)==len(self._link_names))
        i = 0
        for link in self._link_names:
            self._link_objects[link].setPosition(positions[i])
            i = i + 1

    def get_objects_positions(self):
        obj_positions = {}
        for obj in self._objects_names:
            obj_positions[obj] = self._objects[obj].getPosition()
        return obj_positions

    #Added
    def object_positions(self):
        obj_position = self._objects[self._objects_names[0]].getPosition()
        return obj_position

    def get_state(self):
        state = []
        for link in self._link_names:
            state = state + [self._link_position_sensors[link].getValue()]
        for obj in self._objects_names:
            state = state + self._objects[obj].getPosition()
        return np.array(state)

    #Modified
    def object_finger_distance(self, object):
        
        finger_pos_00 = self.fingers["TouchSensor1"].getPosition()
        finger_pos_00_array = np.array(finger_pos_00)

        finger_pos_01 = self.fingers["TouchSensor2"].getPosition()
        finger_pos_01_array = np.array(finger_pos_01)

        '''
        finger_pos_00 = self._link_objects["finger1"].getPosition()
        finger_pos_00_array = np.array(
            (finger_pos_00.x, finger_pos_00.y, finger_pos_00.z)
        )
        finger_pos_01 = self._link_objects["finger2"].getPosition()
        finger_pos_01_array = np.array(
            (finger_pos_01.x, finger_pos_01.y, finger_pos_01.z)
        )
        
        finger_pos_unique = self.finger.getPosition()
        finger_pos_array = np.array(finger_pos_unique)
        '''

        obj_pos = self.object_positions()
        #obj_pos_array = np.array((obj_pos.x, obj_pos.y, obj_pos.z))
        obj_pos_array = np.array(obj_pos)
        
        finger_distances = [
            np.linalg.norm(finger_pos_00_array - obj_pos_array),
            np.linalg.norm(finger_pos_01_array - obj_pos_array),
            #np.linalg.norm(finger_pos_array - obj_pos_array),
        ]

        return np.array(finger_distances).mean()

###### UTIL FUNCTIONS -  END  ######

###### GYM FUNCTIONS -  START  ######

    def step(self, action):
        new_state = []
        reward = 0
        done = False
        obs = []

        obs = self._get_obs()                       #Added
        #done = self._is_done(obs)                  
        reward = self._compute_reward(obs, done)    #Added
        for sensor in self._touch_sensor_names:         #Added
           self._touch_sensor_object[sensor].enable(10)   #Added
        self.set_link_positions(action)
        self._supervisor.step(self._timestep)

        return new_state, reward, done, obs

    def _compute_reward(self, observations, done):
        sensor_mean = (observations["TOUCH_SENSORS"] / 200).mean()

        obj_pos = observations["OBJECT_POSITION"]
        floor_distance = self.alpha * np.exp(
            -(np.linalg.norm(obj_pos - self.desired_pos) ** 2)
            / (2 * dist_dev_alpha ** 2)
        )

        finger_distance = self.beta * np.exp(
            -(self.object_finger_distance(self._objects) ** 2)
            / (2 * dist_dev_beta ** 2)
        )

        touch = self.gamma * sensor_mean
        
        touch_distance = touch * np.exp(
            -(self.object_finger_distance(self._objects) ** 2)
            / (2 * dist_dev_gamma ** 2)
        )
        self._num_rew += 1
        self._touch += touch_distance
        reward = floor_distance + finger_distance + touch_distance

        '''
        print("------------------------")
        print("Floor Distance: ")
        print(floor_distance)
        print("Finger Distance")
        print(finger_distance)
        print("Reward: ")
        print(reward)
        print("------------------------")
        '''

        # print(
        #     "Dfl: %-10.8f Dfi: %-10.8f D*T: %-10.8f"
        #     % (floor_distance, finger_distance, touch_distance)
        # )

        return reward

    #Modified
    def _get_obs(self):
        joint_positions = self.get_link_positions()

        # joint_positions = np.array([self.get_joint_function(joint).position[0] for joint in self.joint_names])
        touch_sensors = np.array([np.linalg.norm(self._touch_sensor_object[sensor].getValues()) for sensor in self._touch_sensor_names])   #Commentato da Edo #Modificato da LoLu
        #print(touch_sensors)
        # obj = self.object_positions[self.object_names[0]] # otre position
        obj = self.object_positions()
        # obj_position = np.array([obj.x, obj.y, obj.z])
        obj_position = np.array(obj)

        obs = {
            "JOINT_POSITIONS": joint_positions,
            "TOUCH_SENSORS": touch_sensors,
            "OBJECT_POSITION": obj_position,
        }
        return obs


    def reset(self):
        #print("Reset")
        #if(self._num_rew != 0):
            #print(self._touch/self._num_rew)

        #self._touch = 0
        #self._num_rew = 0
        #print()
        self._supervisor.simulationReset()
        self._supervisor.simulationResetPhysics()
        self._supervisor.step(1)

        for link in self._link_names:
            self._link_position_sensors[link].enable(self._timestep)

    def render(self, mode='human', close=False):
        pass

###### GYM FUNCTIONS -   END   ######
