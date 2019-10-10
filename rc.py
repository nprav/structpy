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
    'Gr60': {'sy': 413.685, 'Es': 200000, 'compression': False},
    '500': {'sy': 500, 'Es': 200000, 'compression': False},
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
                  Es=200000, compression=False):
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
        max_tension, ety = self.get_max_tension_P()
        max_compression, efc = self.get_max_compression_P()

        id_pos = pd.DataFrame(columns=RcSection.id_column_labels)
        id_neg = pd.DataFrame(columns=RcSection.id_column_labels)

        for id in [id_pos, id_neg]:
            id.loc[0] = [max_compression, 0,
                            efc, efc,
                            ]
            id.loc[npts-1] = [max_tension, 0,
                                  ety, ety,
                                  ]

        for i in range(1, npts-1):
            force = max_compression - i*(max_compression - max_tension)/(npts-1)
            constraints = {'type': 'eq', 'fun': self.get_P}
            res_pos = minimize(lambda x: -self.get_M(x), (0, 0), constraints=constraints)
            id_pos.loc[i] = [force, -res_pos['fun'], *res_pos['x']]
            res_neg = minimize(self.get_M, (0, 0), constraints=constraints)
            id_neg.loc[i] = [force, res_neg['fun'], *res_neg['x']]

        self.id = pd.concat([id_pos, id_neg])
        return self.id

    def get_P(self, strains):
        e_top, e_bot = strains
        if not self.rebars.empty:
            rebar_P = self.rebars.apply(
                rebar_force, axis=1, args=(self.thk, e_top, e_bot),
                **self.conc_matprops,
            )
            P = np.sum(rebar_P)
        else:
            P = 0
        conc_P, conc_cent = conc_force(self.thk, self.width,
                                       e_top, e_bot, self.beta_1,
                                       **self.conc_matprops)
        P = P + conc_P
        return P

    def get_M(self, strains):
        e_top, e_bot = strains
        if not self.rebars.empty:
            rebar_P = self.rebars.apply(
                rebar_force, axis=1, args=(self.thk, e_top, e_bot),
                **self.conc_matprops,
            )
            rebar_cent = self.rebars['y'] - self.thk/2
            rebar_M = rebar_P * rebar_cent
            M = np.sum(rebar_M)
        else:
            M = 0
        conc_P, conc_cent = conc_force(self.thk, self.width,
                                       e_top, e_bot, self.beta_1,
                                       **self.conc_matprops)
        M = M + conc_P * conc_cent
        return M

    def get_max_tension_P(self):
        max_tension = -np.sum(self.rebars['area'] * self.rebars['sy'])
        ety = -self.rebars['e_y'].max()
        return max_tension, ety

    def get_max_compression_P(self):
        compr_rebars = self.rebars[self.rebars['compression']]
        max_conc_compression = 0.85 * (self.thk * self.width -
                                       np.sum(self.rebars['area'])) * \
                               self.conc_matprops['fc'] + \
                               np.sum(compr_rebars['area'] * compr_rebars['sy'])
        if not compr_rebars.empty:
            efc = max(compr_rebars['e_y'].max(), self.conc_matprops['e_fc'])
        else:
            efc = self.conc_matprops['e_fc']
        return max_conc_compression, efc

    def get_strain_limits(self):
        pass

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


def rebar_force(rebar, thk, e_top, e_bot, fc=35, e_fc=0.003):
    strain = (e_top - e_bot) / thk * rebar['y'] + e_bot
    stress = strain * rebar['Es']
    stress = max(min(rebar['sy'], stress), -rebar['sy'])
    if strain > 0:
        conc_stress = 0.85 * fc / e_fc * min(e_fc, max(e_top, e_bot))
        if rebar['compression']:
            stress = stress - conc_stress
        else:
            stress = -conc_stress
    force = stress * rebar['area']
    return force


def conc_force(thk, width, e_top, e_bot, beta_1, fc=35, e_fc=0.003):

    if e_top != e_bot:
        # Setup direction factor so we can assume e_top > e_bot
        direction_factor = 1
        if e_top < e_bot:
            direction_factor = -1
            e_top, e_bot = e_bot, e_top

        c_from_bot = thk / (e_top - e_bot) * (0 - e_bot)

        if c_from_bot >= thk:
            # Neutral axis is above cross-section; i.e. entire
            # section is in tension
            conc_P = 0
            conc_P_centroid = 0

        else:
            max_conc_stress = fc * min(e_top, e_fc)/e_fc
            abs_c_from_top = thk - c_from_bot
            abs_a_from_top = min(beta_1 * abs_c_from_top, thk)
            conc_P_centroid = direction_factor * (thk/2 - abs_a_from_top/2)
            conc_P = abs_a_from_top * 0.85 * max_conc_stress * width

        return conc_P, conc_P_centroid

    else:
        if e_top < 0:
            return 0, 0
        else:
            conc_stress = min(e_top, e_fc) * fc / e_fc
            conc_P = 0.85 * conc_stress * width * thk
            return conc_P, 0
