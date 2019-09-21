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
    rebar_column_labels = ['x', 'y', 'D', 'area', 'sy',
                           'Es', 'e_y', 'compression',
                           ]
    id_column_labels = ['P', 'M', 'e_top', 'e_bot']
    rectangle_kwargs = {'color': 'c'}
    circle_kwargs = {'color': 'k'}

    def __init__(self, width=100, thk=200, fc=35, e_fc=0.003):
        """Instantiate concrete section with a width (x) and thickness (y).
        """
        self.width = width
        self.thk = thk
        self.conc_matprops = {'fc': fc, 'e_fc': e_fc}
        self.rebars = pd.DataFrame(columns=RcSection.rebar_column_labels)
        self.num_rebars = len(self.rebars)
        self.id = pd.DataFrame(columns=RcSection.id_column_labels)
        self.beta_1 = get_beta_1(fc)

    def get_extents(self):
        """Return the boundaries of the defined section.
        """
        print("Section Size : ({}, {})\n".format(
            self.width, self.thk))
        return self.width, self.thk

    def add_rebar(self, D=10, x=0, y=175, sy=500,
                  Es=2000, compression=False):
        """Add a single rebar to the section, defined by diameter,
        x position, and y position.
        """
        area = np.pi / 4 * D ** 2
        e_y = sy / Es
        self.rebars.loc[self.num_rebars] = [x, y, D, area, sy,
                                            Es, e_y, compression,
                                            ]
        self.num_rebars = len(self.rebars)
        print(
            "Rebar added; pos = ({}, {}), od = {}\n".format(
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
                    "{} = {}".format(prop, value)
                    for prop, value
                    in props.items()
                ]))
        print("\n")

        return mat_props

    def plot_section(self):
        """Plot the defined section.
        """
        fig, axis = plt.subplots()
        xy = (-self.width / 2, 0)
        rectangle = plt.Rectangle(xy, self.width, self.thk,
                                  **RcSection.rectangle_kwargs)
        axis.add_patch(rectangle)

        for (x, y, D, *args) in self.rebars.values:
            circle = plt.Circle((x, y), D / 2, **RcSection.circle_kwargs)
            axis.add_patch(circle)

        axis.set_xlim((-self.width / 2 * 1.1, self.width / 2 * 1.1))
        axis.set_ylim((-self.thk / 2 * 0.1, self.thk * 1.05))
        plt.axis('equal')
        plt.show()

        return fig, axis

    def generate_interaction_diagram(self, npts=50):
        max_tension = -np.sum(self.rebars['area'] * self.rebars['sy'])

        compr_rebars = self.rebars[self.rebars['compression']]

        max_conc_compression = 0.85 * (self.thk * self.width -
                                       np.sum(self.rebars['area'])) * \
                               self.conc_matprops['fc'] + \
                               np.sum(compr_rebars['area'] * compr_rebars['sy'])

        ety = -self.rebars['e_y'].max()
        efc = self.conc_matprops['e_fc']

        self.id.loc[0] = [max_conc_compression, 0,
                          efc, efc,
                          ]
        self.id.loc[npts] = [max_tension, 0,
                             ety, ety,
                             ]

        return self.id

    def get_P(self, e_top, e_bot):
        rebar_P = self.rebars.apply(
            rebar_force, axis=1, args=(self.thk, e_top, e_bot)
        )



# %% Define RC Section childrenS
# Define child classes for various types of beam shapes/continuous slabs


class Slab(RcSection):
    pass

class RectangularBeam(RcSection):
    pass

class WBeam(RcSection):
    pass

class CustomBeam(RcSection):
    pass

# %% Define miscellaneous utility functions

def circle_area(diam):
    return np.pi / 4 * diam ** 2


def get_beta_1(fc):
    """Get the beta_1 parameter for the Whitney Stress Block
        simplification (ACI-318 code section 10.2.7)
    :param fc: concrete compressive strength in MPa
    :return: beta_1 parameter
    """
    if fc <= 28:
        beta_1 = 0.85
    elif 28 < fc <= 56:
        beta_1 = 0.85 - 0.05 * (fc - 28) / 7
    else:
        beta_1 = 0.65
    return beta_1


def rebar_force(rebar, thk, e_top, e_bot):
    strain = (e_top - e_bot) / thk * rebar['y']
    stress = strain * rebar['Es']
    stress = max(min(rebar['sy'], stress), -rebar['sy'])
    if (stress > 0) and not rebar['compression']:
        stress = 0
    force = stress * rebar['area']
    return force


def conc_force(thk, width, e_top, e_bot, beta_1, fc=35, e_fc=0.003, rebars=None):

    try:
        # Setup direction factor so we can assume e_top > e_bot
        if e_top > e_bot:
            direction_factor = 1
        elif e_top < e_bot:
            direction_factor = -1
            e_top, e_bot = e_bot, e_top

        c_from_bot = thk / (e_top - e_bot) * (0 - e_bot)

        if c_from_bot >= thk:
            # Neutral axis is above cross-section; i.e. entire
            # section is in tension
            conc_P = 0
            conc_P_centroid = 0

        else:
            c_from_top = thk - c_from_bot
            a_from_top = beta_1 * c_from_top
            conc_P_centroid = direction_factor * (thk/2 - a_from_top/2)
            conc_P = a_from_top * 0.85 * fc * width

        return conc_P, conc_P_centroid

    except ZeroDivisionError:
        assert e_top == e_bot
        if e_top < 0:
            return 0, 0
        else:
            conc_stress = min(e_top, e_fc) * fc / e_fc
            conc_P = 0.85 * conc_stress * width * thk
            return conc_P, 0