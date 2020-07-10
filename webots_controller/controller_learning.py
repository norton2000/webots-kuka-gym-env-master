#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from controller import Robot

import gym, os, glob, time
import numpy as np
import DMp
from DMp.bbo_pdmp import BBO, rew_softmax
import webots_kuka_gym
import matplotlib.pyplot as plt
from gym import spaces
from limits import filter_limits, scale_from_joints, scale_to_joints
from learning_parameters import *

#rospy.init_node("youbot_rl", anonymous=True, log_level=rospy.ERROR)
env = gym.make('webots-kuka-v0')
#env.obj_to_grasp = object_to_grasp  # otre or cube
objects = ["box"]
env.set_objects_names(objects)

#env.gazebo.set_physics_parameters(time_step, update_rate, solver_iter)
#env.speed_up_factor = time_step * update_rate
env.set_timestep_simulation(128) #by me
env.reset()

# the BBO object
bbo = BBO(
    num_params=dmp_num_theta,
    dmp_stime=dmp_stime,
    dmp_rtime=dmp_rtime,
    dmp_dt=dmp_dt,
    dmp_sigma=dmp_sigma,
    num_rollouts=bbo_episodes,
    num_dmps=bbo_num_dmps,
    sigma=bbo_sigma,
    lmb=bbo_lmb,
    epochs=bbo_epochs,
    sigma_decay_amp=bbo_sigma_decay_amp,
    sigma_decay_start=bbo_sigma_decay_start,
    sigma_decay_period=bbo_sigma_decay_period,
    softmax=rew_softmax,
    cost_func=SimulationManager(env),
    max_rew = max_rew,
    sigma_moving_average = sigma_moving_average,
    sigma_moving_average_h = sigma_moving_average_h
)

if continue_learning is True:
    Sks = np.load("Sk.npy")
    last_episode_weights = np.load("bbo_thetas.npy")
    weights = last_episode_weights[np.argmax(Sks[-1, :]), :]
    bbo.theta = weights.copy()
else:
    target_trajectory = np.load("target_trajectory.npy")
    bbo.set_weights(scale_from_joints(target_trajectory.T))

# BBO learning iterations
rew = np.zeros(bbo_epochs)
Sks = np.zeros([bbo_epochs, bbo_episodes])
rew_variance = np.zeros(bbo_epochs)
max_Sk = None
current_best = None

for k in range(bbo_epochs):
    # simulaton step
    start = time.time()
    rollouts, rews, Sk = bbo.iteration()
    end = time.time()
    rew[k] = np.max(Sk)
    Sks[k, :] = Sk
    rew_variance[k] = np.std(Sk)
    best_rollout = rollouts[np.argmax(Sk)]
    if max_Sk is not None:
        if np.max(Sk) > max_Sk:
            np.save("current_best_rollout", best_rollout)
            np.save("current_best_theta", bbo.theta)
            max_Sk = np.max(Sk)
    else:
        max_Sk = np.max(Sk)
    np.save("rewards", rew[:(k+1)])
    np.save("rewards_variance", rew_variance[:(k+1)])
    np.save("Sk", Sks[:(k+1),:])
    np.save("bbo_thetas", bbo.thetas)
    print("{:#4d} {:10.4f} -- {}".format(k, rew[k], end - start))
