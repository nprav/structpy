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
    rebar_details = ['x', 'y', 'D']
    rectangle_kwargs = {'hatch': '/', 'color': 'b'}
    circle_kwargs = {'color': 'k'}

    def __init__(self, width=200, thk=200):
        """Instantiate concrete section with a width (x) and thickness (y).
        """
        self.width = width
        self.thk = thk
        self.rebars = pd.DataFrame(columns=RcSection.rebar_details)
        self.num_rebars = len(self.rebars)

    def get_extents(self):
        """Return the boundaries of the defined section.
        """
        print("Section Size : ({}, {})".format(
                self.width, self.thk))
        return self.width, self.thk

    def add_rebar(self, D=10, x=0, y=175):
        """Add a single rebar to the section, defined by diameter,
        x position, and y position.
        """
        self.rebars.loc[self.num_rebars] = [x, y, D]
        self.num_rebars = len(self.rebars)
        print(
            "Rebar added; pos = ({}, {}), od = {}".format(
                *self.rebars.iloc[-1]
                )
            )

    def plot(self):
        """Plot the defined section.
        """
        fig, axis = plt.subplots()
        xy = (-self.width/2, 0)
        rectangle = plt.Rectangle(xy, self.width, self.thk,
                                  **RcSection.rectangle_kwargs)
        axis.add_patch(rectangle)

        for (x, y, D) in self.rebars.values:
            circle = plt.Circle((x, y), D/2, **RcSection.circle_kwargs)
            axis.add_patch(circle)

        axis.set_xlim((-self.width/2*1.1, self.width/2*1.1))
        axis.set_ylim((-self.thk/2*0.1, self.thk*1.05))
        plt.axis('equal')
        plt.show()

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
