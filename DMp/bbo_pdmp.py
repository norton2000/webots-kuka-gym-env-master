#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
from .pdmp import DMP
import matplotlib.pyplot as plt


np.set_printoptions(precision=3, suppress=True)


def cost_softmax(x, lmb):
    e = np.exp(-(x - np.min(x)) / lmb)
    return e / sum(e)


def rew_softmax(x, lmb):
    e = np.exp((x - np.max(x)) / lmb)
    return e / sum(e)


def invsigm(x, a, b, c):
    return a * (1 - 1 / (1 + np.exp(-(x - c) / b)))


class BBO(object):
    "P^2BB: Policy Improvement through Black Vox Optimization"

    def __init__(
        self,
        num_params=10,
        bins_hparams=None,
        dmp_stime=100,
        dmp_rtime=20,
        dmp_dt=0.1,
        dmp_sigma=0.1,
        num_rollouts=20,
        num_dmps=1,
        sigma=0.001,
        lmb=0.1,
        epochs=100,
        sigma_decay_amp=1,
        sigma_decay_start=0,
        sigma_decay_period=0.1,
        softmax=rew_softmax,
        cost_func=None,
        run_func=None,
        max_rew = 100000,
        sigma_moving_average_h = 0.1,
        sigma_moving_average = False
    ):
        """
        :param num_params: Integer. Number of parameters to optimize
        :param n um_rollouts: Integer. number of rollouts per iteration
        :param num_dmps: Integer, number of dmps
        :param bins_hparams: list(Integer), number of bins for each hparam
        :param dmp_stime: Integer, length of the interpolated trajectories in timesteps
        :param dmp_rtime: Integer, length of the trajectories in timesteps
        :param dmp_dt: Float, integration step
        :param dmp_sigma: Float, standard deviation of dmp gaussian basis functions
        :param sigma: Float. Amount of exploration around the mean of parameters
        :param lmb: Float. Temperature of the evaluation softmax
        :param epochs: Integer. Number of iterations
        :param sigma_decay_amp: Initial additive amplitude of exploration
        :param sigma_decay_start: i Time point of decay [0, 1]
        :param sigma_decay_period: Decaying period of additive amplitude of exploration [0, 1]
        :param softmax: softmax function to use
        :param cost_func: the cost function - runs the environment and compute costs of the policy at each timestep
        :param run_func: runs the environment
        :param max_rew: value of the maximum reward for episode
        :param sigma_moving_average_h: the increment of the moving average (dt/tau)
        :param sigma_moving_average: compute moving average and modulate variance
        """

        self.dmp_stime = dmp_stime
        self.dmp_rtime = dmp_rtime
        self.dmp_dt = dmp_dt
        self.dmp_sigma = dmp_sigma
        self.sigma = sigma
        self.lmb = lmb
        self.num_rollouts = num_rollouts
        self.num_dmps = num_dmps
        self.bins_hparams = bins_hparams
        self.num_dmp_params = num_params
        self.num_params = int(self.num_dmps * (num_params + 2))
        self.theta = np.zeros(self.num_params)
        self.Cov = np.eye(self.num_params, self.num_params)
        self.epochs = epochs
        self.decay_amp = sigma_decay_amp
        self.decay_start = sigma_decay_start
        self.decay_period = sigma_decay_period
        self.epoch = 0
        self.sigma_moving_average = 0
        self.is_sigma_moving_average_active = sigma_moving_average
        self.max_sk_rew = max_rew
        self.sigma_moving_average_h = sigma_moving_average_h

        # create dmps
        self.dmps = []
        for x in range(self.num_dmps):
            self.dmps.append(
                [
                    DMP(
                        n=self.num_dmp_params,
                        s=0,
                        g=1,
                        pdim=self.bins_hparams,
                        stime=self.dmp_rtime,
                        dt=self.dmp_dt,
                        sigma=self.dmp_sigma,
                    )
                    for k in range(self.num_rollouts)
                ]
            )

        # define softmax
        self.softmax = softmax
        # define the cost function
        self.cost_func = cost_func
        # define the run function
        self.run_func = run_func

    def sample(self):
        """ Get num_rollouts samples from the current parameters mean
        """
        delta_sigma = invsigm(
            self.epoch,
            self.decay_amp,
            self.epochs * self.decay_period,
            self.epochs * self.decay_start,
        )
        Sigma = self.sigma + delta_sigma

        curr_sigma = Sigma.copy()
        if self.is_sigma_moving_average_active:
            curr_sigma *= (1 - np.tanh(self.sigma_moving_average/self.max_sk_rew))
            print(self.sigma_moving_average)
            print(curr_sigma)

        # matrix of deviations from the parameters mean
        self.eps = np.random.multivariate_normal(
            np.zeros(self.num_params), self.Cov * curr_sigma, self.num_rollouts
        )

    def update(self, Sk):
        """ Update parameters

            :param Sk: array(Float), rollout costs in an iteration
        """
        # Cost-related probabilities of sampled parameters
        probs = self.softmax(Sk, self.lmb).reshape(self.num_rollouts, 1)
        # update with the weighted average of sampled parameters
        self.theta += np.sum(self.eps * probs, 0)

    def set_weights(self, target_rollout):
        rng = self.num_dmp_params + 2
        for idx, dmp in enumerate(self.dmps):
            dmp[0].generate_weights(target_rollout[idx, :])
            dmp_theta = self.theta[(idx * rng) : ((idx + 1) * rng)]
            dmp_theta[1:-1] = dmp[0].theta.ravel()
            dmp_theta[0] = dmp[0].s
            dmp_theta[-1] = dmp[0].g

    def rollouts(self, thetas):
        """ Produce a rollout
            :param thetas: array(num_rollouts X num_params/num_dmps)
            :return: list(array(num_rollouts, stime)) rollouts
        """
        rollouts = []

        rng = self.num_dmp_params + 2
        for k, theta in enumerate(thetas):
            thetak = theta.copy()
            dmp_rollouts = []
            for idx, dmp in enumerate(self.dmps):
                dmp_theta = thetak[(idx * rng) : ((idx + 1) * rng)]
                dmp[k].reset()
                dmp[k].theta = dmp_theta[1:-1]
                dmp[k].set_start(dmp_theta[0])
                dmp[k].set_goal(dmp_theta[-1])
                dmp[k].rollout()
                _, rollout = dmp[k].interpolate(self.dmp_stime)
                dmp_rollouts.append(rollout)
            rollouts.append(np.vstack(dmp_rollouts))
        rollouts = np.array(rollouts)
        return rollouts

    def outcomes(self, rollouts):
        """
        compute outcomes for a stack of rollouts
        :param rollouts: array(num_episodes,num_dmps, stime)
                for each episode a stack of num_dmp rollouts
        """
        errs = self.cost_func(rollouts)
        return errs

    def run(self):
        rollout = self.rollouts(self.theta.reshape(1, -1))[0]
        self.run_func(rollout)

    def eval(self, errs):
        """ evaluate rollouts
            :param errs: list(array(float)), Matrices containing DMPs' errors
                 at each timestep (columns) of each rollout (rows)
            return: array(float), overall cost of each rollout
        """
        self.err = np.mean(np.mean(errs, 1))  # store the mean square error

        # compute costs
        Sk = np.zeros(self.num_rollouts)
        for k in range(self.num_rollouts):
            Sk[k] = 0
            # final costs
            Sk[k] += errs[k, -1]
            for j in range(len(errs[k])):
                # cost-to-go integral
                Sk[k] += errs[k, j:-1].sum()
            # # regularization
            # thetak = self.theta + self.eps[k]
            # Sk[k] += 0.5 * np.mean(self.sigma) * (thetak).dot(thetak)

        return Sk

    def iteration(self, explore=True):
        """ Run an iteration
            :param explore: Bool, If the iteration is for training (True)
                or test (False)
            :return: (rollouts, total value of the iteration)
        """
        self.sample()
        if explore is True:
            self.thetas = self.theta + self.eps
        else:
            self.thetas = self.theta + np.zeros_like(self.eps)
        rollouts = self.rollouts(self.thetas)
        costs = self.outcomes(rollouts.copy())
        Sk = self.eval(costs)
        if explore is True:
            self.update(Sk)
            if self.is_sigma_moving_average_active:
                self.sigma_moving_average += \
                        self.sigma_moving_average_h*(-self.sigma_moving_average + \
                        np.max(Sk))
        self.epoch += 1
        return rollouts.copy(), costs, Sk

    def moving_average(self, Sk):
        pass

# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # consts
    dmp_num_theta = 20
    dmp_stime = 100
    dmp_dt = 0.2
    dmp_sigma = 0.1

    bbo_sigma = 1.0e-03
    bbo_lmb = 0.1
    bbo_epochs = 100
    bbo_K = 45
    bbo_num_dmps = 2

    # target trajectory
    partial_stime = int(dmp_stime * 0.7)
    x = np.linspace(0.0, 1.0, partial_stime)
    a = x * 2 * np.pi
    targetx = x * np.cos(a) + x
    targetx /= max(targetx)
    targetx = np.hstack((targetx, np.ones(int(dmp_stime - partial_stime))))
    targety = x * np.sin(a) + x
    targety /= max(targety)
    targety = np.hstack((targety, np.ones(dmp_stime - partial_stime)))

    # plot target
    fig1 = plt.figure()
    ax01 = fig1.add_subplot(211)
    ax01.plot(targetx)
    ax01.plot(targety)
    ax02 = fig1.add_subplot(212)
    ax02.plot(targetx, targety)

    # make a target list for the bbo object
    target = [targetx, targety]

    # make a cost function for the bbo object
    def supervised_cost_func(rollouts):
        trgts = np.array(target)
        trgts = trgts.reshape(bbo_num_dmps, 1, dmp_stime)
        return (trgts - rollouts) ** 2

    # the BBO object
    bbo = BBO(
        num_params=dmp_num_theta,
        dmp_stime=dmp_stime,
        dmp_dt=dmp_dt,
        dmp_sigma=dmp_sigma,
        num_rollouts=bbo_K,
        num_dmps=bbo_num_dmps,
        sigma=bbo_sigma,
        lmb=bbo_lmb,
        epochs=bbo_epochs,
        sigma_decay_amp=0.0,
        sigma_decay_period=0.1,
        softmax=cost_softmax,
        cost_func=supervised_cost_func,
    )

    # prepare for iterations
    costs = np.zeros(bbo_epochs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(costs)
    # iterate
    for t in range(bbo_epochs):
        rs, _ = bbo.iteration()
        costs[t] = bbo.err
        line.set_ydata(costs)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)
    print()

    # test
    rollouts, _ = bbo.iteration(explore=False)

    # plot test
    fig2 = plt.figure()
    ax11 = fig2.add_subplot(211)
    ax11.plot(rs[0].T, lw=0.2, color="#220000")
    ax11.plot(rs[1].T, lw=0.2, color="#002200")
    ax11.plot(rollouts[0].T, lw=0.2, color="#884444")
    ax11.plot(rollouts[1].T, lw=0.2, color="#448844")
    ax12 = fig2.add_subplot(212)
    ax12.plot(targetx, targety, lw=2, color="red")
    ax12.plot(rs[0].T, rs[1].T, lw=0.2, color="black")
    ax12.plot(rollouts[0].T, rollouts[1].T, color="green", lw=3)
    plt.show()
