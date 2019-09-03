# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 07:52:43 2019

@author: Praveer Nidamaluri

Module for analyzing Reingforced Concrete sections. Primary aim
is to make interaction diagrams.
"""


# %% Import Necessary Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# %% Define overall section class


class RcSection():
    """General concrete section class.
    """

    def __init__(self, width, thk):
        self.width = width
        self.thk = thk
        self.rebars = []

    def get_extents(self):
        print("Section Size : ({}, {})".format(
                self.width, self.thk))
        return self.width, self.thk

    def plot(self):
        xy = (-self.width/2, 0)
        rectangle = plt.Rectangle(xy, self.width, self.thk)
        fig, axis = plt.subplots()
        axis.add_patch(rectangle)
        axis.set_xlim((-self.width/2*1.1, self.width/2*1.1))
        axis.set_ylim((-self.thk/2*0.1, self.thk*1.05))
        return fig, axis


# %% Define RC Section children
# Define child classes for various types of beam shapes/continuous slabs

class Slab(RcSection):
    pass


class RectangularBeam(RcSection):
    pass


class WBeam(RcSection):
    pass


class CustomBeam(RcSection):
    pass


# %% Define reinforcement class


class Rebar():
    pass