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

# %% Define global properties

default_materials = {
    'concrete': {'fc': 35, 'e_fc': 0.003},
    'Gr60': {'sy': 413.685, 'Es': 2000, 'compression': False},
    '500': {'sy': 500, 'Es': 2000, 'compression': False},
    }

# %% Define overall section class


class RcSection(object):
    """General concrete section class.
    """
    rebar_column_labels = ['x', 'y', 'D', 'sy', 'Es', 'compression']
    rectangle_kwargs = {'color': 'c'}
    circle_kwargs = {'color': 'k'}

    def __init__(self, width=200, thk=200, fc=35, e_fc=0.003):
        """Instantiate concrete section with a width (x) and thickness (y).
        """
        self.width = width
        self.thk = thk
        self.conc_matprops = {'fc': fc, 'e_fc': e_fc}
        self.rebars = pd.DataFrame(columns=RcSection.rebar_column_labels)
        self.num_rebars = len(self.rebars)

    def get_extents(self):
        """Return the boundaries of the defined section.
        """
        print("Section Size : ({}, {})".format(
                self.width, self.thk))
        return self.width, self.thk

    def add_rebar(self, D=10, x=0, y=175, sy=500,
                  Es=2000, compression=False):
        """Add a single rebar to the section, defined by diameter,
        x position, and y position.
        """
        self.rebars.loc[self.num_rebars] = [x, y, D, sy, Es, compression]
        self.num_rebars = len(self.rebars)
        print(
            "Rebar added; pos = ({}, {}), od = {}".format(
                *self.rebars.iloc[-1]
                )
            )

    def get_mat_props(self):
        """Return material properties of concrete section and all
        defined rebars.
        """
        mat_props = {'concrete': self.conc_matprops}
        view = self.rebars[['sy', 'Es']]
        for i in range(len(view)):
            mat_props['rebar{}'.format(i)] = dict(zip(view.columns,
                                                      view.iloc[i]))
        for mat, props in mat_props.items():
            print(
                "{}: ".format(mat) +
                ", ".join([
                    "{}={}".format(prop, value)
                    for prop, value
                    in props.items()
                    ]))

        return mat_props

    def plot(self):
        """Plot the defined section.
        """
        fig, axis = plt.subplots()
        xy = (-self.width/2, 0)
        rectangle = plt.Rectangle(xy, self.width, self.thk,
                                  **RcSection.rectangle_kwargs)
        axis.add_patch(rectangle)

        for (x, y, D, *args) in self.rebars.values:
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
