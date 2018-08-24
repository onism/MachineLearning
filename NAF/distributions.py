#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from scipy.stats import multivariate_normal
import numpy as np
import torch

class Distr(object):
    """docstring for Distr"""
    hasenergyf = False
    hassplr = False

    def energy(self, x):
        raise NotImplementedError

    def sampler(self, x):
        raise NotImplementedError


class Mixture(Distr):
    """docstring for Mixture"""
    hasenergyf = True
    hassplr = True
    def __init__(self, probs, distr_list):
        self.probs = np.asarray(probs)
        self.distr_list = distr_list

    def energy(self, x):
        pdf = np.nasarray([distr.pdf(x) for distr in self.distr_list])
        return np.dot(self.probs,pdf)

