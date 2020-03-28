# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 07:52:43 2019

@author: Praveer Nidamaluri

Module for analyzing Reinforced Concrete sections. Primary aim
is to make interaction diagrams.
"""

# %% Import Necessary Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    # Define common class properties such as dataframe input labels \
    # and general plotting settings settings.
    rebar_column_labels = ['x', 'y', 'D', 'area', 'sy',
                           'Es', 'e_y', 'compression',
                           ]
    id_column_labels = ['P', 'M', 'e_top', 'e_bot']
    rectangle_kwargs = {'color': 'c'}
    circle_kwargs = {'color': 'k'}

    def __init__(self, width=100, thk=200, fc=35, e_fc=0.003):
        """Instantiate concrete section with a width (x), thickness (y),
        concrete compressive strength, and failure strain.
        Make sure units are consistent. Defaults to 100mm wide, 200mm thick,
        35MPa concrete compressive strength, 0.003 mm/mm failure strain.
        """

        self.width = width
        self.thk = thk
        self.conc_matprops = {'fc': fc, 'e_fc': e_fc}

        # Define empty dataframe for rebars
        self.rebars = pd.DataFrame(columns=RcSection.rebar_column_labels)
        self.num_rebars = len(self.rebars)

        # Define empty dataframe for interaction diagram
        self.id = pd.DataFrame(columns=RcSection.id_column_labels)
        self.beta_1 = get_beta_1(fc)

    def get_extents(self):
        """Return the dimensions of the defined section.
        Assumes a simple rectangular cross-section.
        """
        print("Section Size : ({}, {})\n".format(
            self.width, self.thk))
        return self.width, self.thk

    def add_rebar(self, D=10, x=0, y=175, sy=500,
                  Es=200000, compression=False):
        """Add a single rebar to the section, defined by diameter,
        x position, y position, yield strength, Young's modulus,
        and a boolean, `compression`, that defines if the rebar is
        active in compression or not.

        Make sure units are consistent with concrete section material
        property inputs.
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
        """Plot the defined section (concrete and rebars).
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
        """Generate the interaction diagram for the defined
        reinforced concrete section. Defaults to a minimum of
        50 points.
        """
        npts = max(50, npts)

        id_pos = pd.DataFrame(columns=RcSection.id_column_labels)
        id_neg = pd.DataFrame(columns=RcSection.id_column_labels)

        top_str_limits, bot_str_limits = self.get_strain_limits()

        # Set up negative side of interaction diagram
        bot_str = bot_str_limits[1]
        alpha = self.thk / self.beta_1
        # Get the first top strain value that will give a moment other
        # than the maximum moment
        start_top_str = top_str_limits[1] / alpha * (alpha - self.thk)
        # setup spacing of points on the interaction diagram
        raw_spacing = (np.geomspace(1, 101, npts - 5) - 1) / 100
        spacing = raw_spacing * (top_str_limits[0] - start_top_str) + start_top_str
        # Iterate through the points and generate the interaction diagram
        for top_str in [top_str_limits[1], *spacing]:
            P = self.get_P((top_str, bot_str))
            M = self.get_M((top_str, bot_str))
            id_neg.loc[len(id_neg)] = [P, M, top_str, bot_str]

        # The worst case angle of section has been reached: concrete failure strain
        # on one side, and tensile limit strain on the other. Now generate the final
        # points by 'pushing out' the cross-section plane so that the concrete strain
        # reaches 0, while the tensile limit strain is constant.
        top_str = top_str_limits[0]
        for bot_str in np.linspace(0, bot_str_limits[1], 5, endpoint=False)[::-1]:
            P = self.get_P((top_str, bot_str))
            M = self.get_M((top_str, bot_str))
            id_neg.loc[len(id_neg)] = [P, M, top_str, bot_str]

        # Setup postiive side of interaction diagram
        top_str = top_str_limits[1]
        start_bot_str = bot_str_limits[1] / alpha * (alpha - self.thk)
        spacing = raw_spacing * (bot_str_limits[0] - start_bot_str) + start_bot_str
        # Iterate through the points and generate the interaction diagram
        for bot_str in [bot_str_limits[1], *spacing]:
            P = self.get_P((top_str, bot_str))
            M = self.get_M((top_str, bot_str))
            id_pos.loc[len(id_pos)] = [P, M, top_str, bot_str]

        # The worst case angle of section has been reached: concrete failure strain
        # on one side, and tensile limit strain on the other. Now generate the final
        # points by 'pushing out' the cross-section plane so that the concrete strain
        # reaches 0, while the tensile limit strain is constant.
        bot_str = bot_str_limits[0]
        for top_str in np.linspace(0, top_str_limits[1], 5, endpoint=False)[::-1]:
            P = self.get_P((top_str, bot_str))
            M = self.get_M((top_str, bot_str))
            id_pos.loc[len(id_pos)] = [P, M, top_str, bot_str]

        # Combine, and plot, and return the interaction diagram
        self.id = id_pos.append(id_neg[::-1])
        plt.plot(self.id.M, self.id.P, 'x-')

        return self.id

    def get_P(self, strains):
        """Get the force for a given tuple of strains: (top strain, bottom strain)
        """
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
        """Get the moment for a given tuple of strains: (top strain, bottom strain)
        """
        e_top, e_bot = strains
        if not self.rebars.empty:
            rebar_P = self.rebars.apply(
                rebar_force, axis=1, args=(self.thk, e_top, e_bot),
                **self.conc_matprops,
            )
            rebar_cent = self.rebars['y'] - self.thk / 2
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
        top_str_limits, bot_str_limits = self.get_strain_limits()
        ety = min(top_str_limits[0], bot_str_limits[0])
        max_tension = self.get_P((ety, ety))
        return max_tension, ety

    def get_max_compression_P(self):
        top_str_limits, bot_str_limits = self.get_strain_limits()
        efc = min(top_str_limits[-1], bot_str_limits[-1])
        max_compression = self.get_P((efc, efc))
        return max_compression, efc

    def get_strain_limits(self):
        if len(self.rebars) > 0:
            min_top_str = (-self.rebars['sy'] / self.rebars['Es'] - self.conc_matprops['e_fc']) \
                / self.rebars['y'] * self.thk
            min_bot_str = (-self.rebars['sy'] / self.rebars['Es'] - self.conc_matprops['e_fc']) \
                / (self.rebars['y'] - self.thk) * (-self.thk) + self.conc_matprops['e_fc']
            top_str_limits = (min_top_str.min(), self.conc_matprops['e_fc'])
            bot_str_limits = (min_bot_str.min(), self.conc_matprops['e_fc'])

        else:
            top_str_limits = (0, self.conc_matprops['e_fc'])
            bot_str_limits = (0, self.conc_matprops['e_fc'])

        return top_str_limits, bot_str_limits


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
    """ Get the force associated with a rebar in a given reinforced
    concrete section.
    :param rebar: Dataframe row with rebar infomation. Should include the column
                    labels from the RcSection class.
    :param thk: thickness of reinforced concrete section. Units should be consistent
                with the rebar input dimensions.
    :param e_top: Strain at the top of the RC section.
    :param e_bot: Strain at the bottom of the RC section.
    :param fc: Concrete compressive strength.
    :param e_fc: Concrete failure strain.
    :return: Force associated with rebar.
    """
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
    """ Get the concrete force from given top and bottom strains.
    :param thk: RC section thickness.
    :param width: RC section width.
    :param e_top: Strain at the top of the RC section.
    :param e_bot: Strain at the bottom of the RC section.
    :param beta_1: beta parameter (Whitney Stress block formulation)
    :param fc: Concrete compressive strength
    :param e_fc: Concrete failure strain
    :return: Concrete force, and force centroid with respect to the centroid
             of the RC section.
    """
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
            max_conc_stress = fc * min(e_top, e_fc) / e_fc
            abs_c_from_top = thk - c_from_bot
            abs_a_from_top = min(beta_1 * abs_c_from_top, thk)
            conc_P_centroid = direction_factor * (thk / 2 - abs_a_from_top / 2)
            conc_P = abs_a_from_top * 0.85 * max_conc_stress * width

        return conc_P, conc_P_centroid

    else:
        if e_top < 0:
            return 0, 0
        else:
            conc_stress = min(e_top, e_fc) * fc / e_fc
            conc_P = 0.85 * conc_stress * width * thk
            return conc_P, 0
