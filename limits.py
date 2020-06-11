#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

'''
limits_names = [
    "joint_1_limit",
    "joint_2_limit",
    "joint_3_limit",
    "joint_4_limit",
    "joint_5_limit",
    "base_to_finger00_joint",
    "base_to_finger10_joint",
]
'''

# limits_values = [
#     [-np.pi * (180 / 180), np.pi * (180 / 180)],
#     [-np.pi * (180 / 180), np.pi * (180 / 180)],
#     # [-np.pi*(146+151)/180, 0],
#     [-np.pi * (151 / 180), np.pi * (146 / 180)],
#     [-np.pi * (102.5 / 180), np.pi * (102.5 / 180)],
#     [-np.pi * (167.5 / 180), np.pi * (167.5 / 180)],
#     [0, np.pi * (90 / 180)],
#     [0, np.pi * (90 / 180)],
#     [-np.pi * (90 / 180), 0],
#     [-np.pi * (90 / 180), 0],
# ]

limits_values = [
    [-2.9496, 2.9496], #[-np.pi / 2.0, np.pi / 2.0],
    [-1.13, np.pi / 2.0],   #[-np.pi / 2.0, np.pi * 0.75],
    [-2.635, 2.54818], #[-np.pi / 2.0, np.pi / 2.0],
    [-1.78, 1.78], #[-np.pi / 2.0, np.pi / 2.0],
    [-2.92, 2.92], #[-np.pi / 2.0, np.pi / 2.0],
    [0, 0.025],         #[0, np.pi / 2.0]
    [0, 0.025],         #[0, np.pi / 2.0]
    #[-np.pi / 2.0, np.pi],     #With 9 parametres (Obsolet)
    #[-np.pi / 2.0, np.pi],     #With 9 parametres (Obsolet)
]

# initial joint (proportional to the limit interval)
reset = [0.5, 0.25, 0.5, 0.5, 0.5, 0, 0]        #With 9 parametres    reset = [0.5, 0.25, 0.5, 0.5, 0.5, 0, 0, 1, 1] 


def filter_limits(joints):
    for trj, limit in enumerate(limits_values):
        joints[trj] = np.minimum(limit[1], np.maximum(limit[0], joints[trj]))
    '''         #Obsolet (only with 9 parametres)
    #joints[-2] = -np.maximum(0, np.minimum(2 * joints[-4], (np.pi/2  + joints[-2])))
    #joints[-1] = -np.maximum(0, np.minimum(2 * joints[-3], (np.pi/2  + joints[-1])))
    '''
    return joints


def scale_to_joints(values):
    joints = np.zeros_like(values)
    for trj, limit in enumerate(limits_values):
        joints[trj] = (
            ((limit[1] - limit[0]) / 2.0) * values[trj]
            + (limit[1] - limit[0]) * reset[trj]
            + limit[0]
        )

    return joints


def scale_from_joints(joints):
    values = np.zeros_like(joints)
    for trj, limit in enumerate(limits_values):
        values[trj] = (
            2
            * (-limit[0] + reset[trj] * (limit[0] - limit[1]) + joints[trj])
            / (limit[1] - limit[0])
        )
    return values


if __name__ == "__main__":
    x0 = np.vstack([-np.ones(9), np.ones(9)]).T
    x1 = scale_to_joints(x0)
    x00 = scale_from_joints(x1)

    print(x0)
    print()
    print(x1)
    print()
    print(x00)

    print(scale_to_joints(np.random.uniform(-1, 1, 9).reshape(-1, 1)))
    print(scale_to_joints(np.random.uniform(-1, 1, 9)))
    print(scale_to_joints(np.ones(9).reshape(-1, 1)))
    print(scale_to_joints(np.ones(9)))
