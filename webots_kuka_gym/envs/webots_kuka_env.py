
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from limits import filter_limits, scale_to_joints
from learning_parameters import *
from random import randrange
#from ArffPrinter import ArffPrinter

# Import from webots classes
from controller import Supervisor, Motor, Camera

class WebotsKukaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.desired_pos = np.array([-0.15, 0.185, 0])

        self.objects_initial_positions = {
            "ring":[0.53, 0.015, 0],
            "ball":[0.4, 0.025, 0],
            "box" :[0.5, 0.02, 0]
        }

        self._supervisor = Supervisor()
        self._timestep = 32

        self._touch = 0
        self._num_rew = 0
        # List of objects to be taken into consideration
        self._objects_names = []
        self._objects = {}

        # Get Fingers
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

        # Get Touch sensor  (Added)
        self._touch_sensor_names = ["touch sensor1" , "touch sensor2"]
         #Added
        self._touch_sensor_object = {}
        for sensor in self._touch_sensor_names:
            self._touch_sensor_object[sensor] = self._supervisor.getTouchSensor(sensor)
            self._touch_sensor_object[sensor].enable(self._timestep)

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

        self._floor = 0
        self._finger = 0
        self._touch = 0

        self._i = 0


###### UTIL FUNCTIONS - START ######

    def get_links_num(self):
        return len(self._link_names)

    def set_timestep_simulation(self, step):
        self._timestep = step

    def set_objects_names(self, names):
        self._objects_names = names
        for obj in self._objects_names:
            self._objects[obj] = self._supervisor.getFromDef(obj)

    def set_finger_name(self, name):
        self.finger_names = name
        self.finger = self._supervisor.getFromDef(name)

    def get_link_positions(self):
        positions = []
        for link in self._link_names:
            positions.append(self._link_position_sensors[link].getValue())
        return np.array(positions)

    def set_link_positions(self, positions):
        assert(len(positions)==len(self._link_names))
        i = 0
        for link in self._link_names:
            self._link_objects[link].setPosition(positions[i])
            i = i + 1

    def get_objects_positions(self):
        obj_positions = {}
        for obj in self._objects_names:
            obj_positions[obj] = self._objects[obj].getPosition()
        return obj_positions

    def object_position(self):
        obj_position = self._objects[self._objects_names[0]].getPosition()
        return obj_position

    def get_state(self):
        state = []
        for link in self._link_names:
            state = state + [self._link_position_sensors[link].getValue()]
        for obj in self._objects_names:
            state = state + self._objects[obj].getPosition()
        return np.array(state)

    def object_finger_distance(self, object_name):

        finger_pos_00 = self.fingers["TouchSensor1"].getPosition()
        finger_pos_00_array = np.array(finger_pos_00)

        finger_pos_01 = self.fingers["TouchSensor2"].getPosition()
        finger_pos_01_array = np.array(finger_pos_01)

        obj_pos = self.get_objects_positions()[object_name]
        obj_pos_array = np.array(obj_pos)

        if(object_name=="ring" or object_name=="box"):
            finger_distances = [
                np.linalg.norm(finger_pos_00_array - obj_pos_array + [0,0,+0.019]),
                np.linalg.norm(finger_pos_01_array - obj_pos_array + [0,0,0.019])
            ]
        else:
            finger_distances = [
                np.linalg.norm(finger_pos_00_array - obj_pos_array + [0,0,+0]),
                np.linalg.norm(finger_pos_01_array - obj_pos_array + [0,0,0])
            ]

        return np.array(finger_distances).mean()

    def randomizeEnvironment(self):
        object_notToGrasp = self._supervisor.getFromDef(self._objects_names[1])
        _translation = object_notToGrasp.getField("translation")
        array_pos = []
        array_pos.append(object_notToGrasp.getPosition())
        array_pos.append([1,0,1])
        array_pos.append([-0.15, 0.195, 0])
        rand = randrange(0,3)
        new_position = array_pos[rand]
        _translation.setSFVec3f(new_position)

###### UTIL FUNCTIONS -  END  ######

###### GYM FUNCTIONS -  START  ######

    def step(self, action):
        new_state = []
        reward = 0
        done = False
        obs = []

        obs = self._get_obs()
        reward = self._compute_reward(obs, done)
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
            -((self.object_finger_distance(self._objects_names[0])+0.9) ** 2)
            / (2 * dist_dev_beta ** 2)
        )

        touch = self.gamma * sensor_mean

        touch_distance = touch * np.exp(
            -(self.object_finger_distance(self._objects_names[0]) ** 2)
            / (2 * dist_dev_gamma ** 2)
        )
        self._num_rew += 1
        self._touch += touch_distance

        #reward = floor_distance + finger_distance + touch_distance
        reward = finger_distance + touch_distance

        self._floor += floor_distance
        self._finger += finger_distance
        self._touch += touch_distance

        self._i += 1

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

    def _get_obs(self):
        joint_positions = self.get_link_positions()

        touch_sensors = np.array([np.linalg.norm(self._touch_sensor_object[sensor].getValues()) for sensor in self._touch_sensor_names])
        obj = self.object_position()
        obj_position = np.array(obj)

        obs = {
            "JOINT_POSITIONS": joint_positions,
            "TOUCH_SENSORS": touch_sensors,
            "OBJECT_POSITION": obj_position,
        }
        return obs


    def reset(self):

        #print("Reset")

        #@

        if(self._i != 0):
            print("floor: ", self._floor, "   finger: ", self._finger, "   touch: ", self._touch)
        self._i = 0
        self._floor = 0
        self._finger = 0
        self._touch = 0
        self._supervisor.simulationReset()
        self._supervisor.simulationResetPhysics()
        self._supervisor.step(1)

        #self.randomizeEnvironment()

        for link in self._link_names:
            self._link_position_sensors[link].enable(self._timestep)

        for sensor in self._touch_sensor_names:
            self._touch_sensor_object[sensor].enable(self._timestep)

    def render(self, mode='human', close=False):
        pass

###### GYM FUNCTIONS -   END   ######

###### CLASSIFIER FUNCITIONS - START ######

    def _objectPositionClassifier(self, object_position, object_initial_position):
        object_position = np.array(object_position)
        object_initial_position = np.array(object_initial_position)
        diff = np.linalg.norm(object_position - object_initial_position)
        return diff<=0.001

    def _fingerDistanceClassifier(self, finger_distance):
        return finger_distance<=0.05

    def _desiredPosClassifier(self, object_position, desired_pos):
        diff = np.linalg.norm(object_position - desired_pos)
        return diff<=0.03

    def _touchSensorsClassifier(self, touch_sensors):
        esito = True;
        for touch_sensor in touch_sensors:
            esito = esito and (touch_sensor >= 300)
        return esito

    def _jointPositionClassifier(self, joint_positions):
        esito = True;
        for joint_position in joint_positions:
            esito = esito and (abs(joint_position) <= 0.01)
        return esito

    def _getValuesFromSensors(self):
        obs = self._get_obs()
        values = {}
        i = 0
        obj_names = self._objects_names.copy()
        obj_names.sort()
        for obj_name in self._objects_names:
            values[i] = self._objectPositionClassifier(self.get_objects_positions()[obj_name], self.objects_initial_positions[obj_name])
            values[i+1] = self._fingerDistanceClassifier(self.object_finger_distance(obj_name))
            values[i+2] = self._desiredPosClassifier(self.get_objects_positions()[obj_name], self.desired_pos)
            i+=3
        values[i] = self._touchSensorsClassifier(obs["TOUCH_SENSORS"])
        values[i+1] = self._jointPositionClassifier(self.get_link_positions())
        return values

###### CLASSIFIER FUNCITION -   END   ######
