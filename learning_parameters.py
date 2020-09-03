#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from ArffPrinter import ArffPrinter
from Classifiers import optionsClassifier


# DMP parameters
dmp_num_theta = 10  # number of parameters
dmp_rtime = 20  # number of points to be interpolated
dmp_stime = 40 # number of simulation's timesteps
dmp_dt = 0.25  # integration time over dmp_rtime
dmp_sigma = 0.05  # standard deviation of each parameters 0.05

# BBO parameters
bbo_lmb = 0.005
bbo_epochs = 30
bbo_episodes = 25
bbo_num_dmps = 7
bbo_sigma_max = 0.2
bbo_sigma_arm_scale = 0.0017
bbo_sigma_joint_scales =  [bbo_sigma_arm_scale,
                           bbo_sigma_arm_scale,
                           bbo_sigma_arm_scale,
                           bbo_sigma_arm_scale,
                           1,1,1]
bbo_sigma = bbo_sigma_max * np.hstack(
    [
        np.ones(dmp_num_theta + 2) * x
        for x in bbo_sigma_joint_scales
    ]
)  # constant sample variance
bbo_sigma_decay_amp = 0.015          # variable sample variance
bbo_sigma_decay_start = 0.1
bbo_sigma_decay_period = 0.1
init_gap = 1
continue_learning = False                                    #False when start a new Learning
write_arff_file = False                                      #False when you don't want the arffprinter to write on files

# YOUBOT learning_parameters
dist_dev_alpha = 0.15
dist_dev_beta = 0.37
dist_dev_gamma = 0.012
alpha = 0.3                     # floor distance
beta = 1                        # finger_distance
gamma = 1.5                     # touch_distance
max_rew = 80
sigma_moving_average = True
sigma_moving_average_h = 0.2

object_to_grasp = 'otre'
# object

from limits import *


class SimulationManager:
    def __init__(self, env):
        """
        :env: a openai_ros env
        """
        self.env = env
        self.init_gap = init_gap
        self.optionsClassifier = optionsClassifier()

    def init_trj(self, ro):
        return np.hstack((np.zeros([ro.shape[0], self.init_gap]), ro))

    def __call__(self, rollouts):
        """
        :rollouts: nparray [episode_num, dmp_num, timesteps]
        """
        n_episodes, n_joints, timesteps = rollouts.shape
        rews = np.zeros([n_episodes, timesteps + self.init_gap])
        #rewsOfEpisodes = np.zeros([n_episodes])

        for episode in range(n_episodes):

            print("episode %d" % episode)

            # simulate with the current joint trajectory to read rewards
            rollout = np.squeeze(rollouts[episode, :, :])
            rollout = filter_limits(scale_to_joints(rollout))
            rollout = self.init_trj(rollout)

            self.env.reset()

            for t in range(timesteps + self.init_gap):
                if t is 1:
                    pre = self.env._getValuesFromSensors()
                action = rollout[:, t]
                obs, reward, done, info = self.env.step(action)
                rews[episode, t] = reward
                #rewsOfEpisodes[episode] = np.sum(rews[episode,])

            post = self.env._getValuesFromSensors()

            if write_arff_file:
                self.optionsClassifier.classifier(pre, post)

        #plt.plot(rewsOfEpisodes)
        #print(rewsOfEpisodes)
        return rews[:, self.init_gap :]


class ExploreSimulationManager:
    def __init__(self, env):
        """
        :env: a openai_ros env
        """
        self.env = env
        self.init_gap = init_gap

    def init_trj(self, ro):
        return np.hstack((np.zeros([ro.shape[0], self.init_gap]), ro))

    def __call__(self, rollouts):
        """
        :rollouts: nparray [episode_num, dmp_num, timesteps]
        """
        n_episodes, n_joints, timesteps = rollouts.shape
        rews = np.zeros([n_episodes, timesteps + self.init_gap])

        for episode in range(n_episodes):

            print("episode %d" % episode)

            # simulate with the current joint trajectory to read rewards
            rollout = np.squeeze(rollouts[episode, :, :])
            rollout = self.init_trj(rollout)
            rollout = filter_limits(scale_to_joints(rollout))

            self.env.explore_reset()
            for t in range(timesteps + self.init_gap):
                action = rollout[:, t]
                obs, reward, done, info = self.env.step(action)
                rews[episode, t] = reward

        return rews[:, self.init_gap :]


class RunManager(SimulationManager):
    def __call__(self, rollout):
        """
        :rollouts: nparray [dmp_num, timesteps]
        """

        n_joints, timesteps = rollout.shape

        # simulate with the current joint trajectory to read rewards
        rollout = self.init_trj(rollout)
        rollout = filter_limits(scale_to_joints(rollout))

        for t in range(timesteps + self.init_gap):
            action = rollout[:, t]
            obs, reward, done, info = self.env.step(action)


class TestSimulationManager(SimulationManager):
    def __call__(self, rollouts):
        """
        :rollouts: nparray [episode_num, dmp_num, timesteps]
        """

        n_episodes, n_joints, timesteps = rollouts.shape
        rews = np.zeros([n_episodes, timesteps + self.init_gap])

        for episode in range(n_episodes):

            # simulate with the current joint trajectory to read rewards
            rollout = np.squeeze(rollouts[episode, :, :])
            rollout = self.init_trj(rollout)
            rollout = filter_limits(scale_to_joints(rollout))

            reads = np.zeros_like(rollout)

            self.env.reset()
            for t in range(timesteps + self.init_gap):
                action = rollout[:, t]
                obs, reward, done, info = self.env.step(action)
                reads[:, t] = obs["JOINT_POSITIONS"]
                rews[episode, t] = reward

        return rews[:, self.init_gap :], reads

class ResetJointPositionManager:
    def __init__(self, env):
        """
        :env: a openai_ros env
        """
        self.env = env
        self.init_gap = init_gap

    def __call__(self, rollouts):
        """
        :rollouts: nparray [episode_num, dmp_num, timesteps]
        """
        n_episodes, n_joints, timesteps = rollouts.shape
        rews = np.zeros([n_episodes, timesteps + self.init_gap])

        for episode in range(n_episodes):

            print("episode %d" % episode)

            # all the joints to zero
            rollout = np.zeros([n_joints, timesteps])

            self.env.reset()
            for t in range(timesteps):
                action = rollout[:, t]
                obs, reward, done, info = self.env.step(action)
                rews[episode, t] = reward

        return rews
