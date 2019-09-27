# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 07:11:59 2019

@author: Praveer Nidamaluri

Script for testing other scripts in the module.
"""

# %% Import Necessary Modules

import unittest
from unittest.mock import patch
import numpy as np
from rc import RcSection, get_beta_1, rebar_force, conc_force

# %% Global variables
steel_sy = 500
steel_Es = 200000


# %% Testcases for rc.py


class TestRC(unittest.TestCase):
    """Unittest Testcase to test the rc.py module that contains
    utility functions for reinforced concrete sections.
    """

    def setUp(self):
        """Initial definitions to set up subsequent tests.
        """
        self.width = 200
        self.thk = 1750
        self.fc = 40
        self.beta_1 = get_beta_1(self.fc)
        inputs = {
            'width': self.width,
            'thk': self.thk,
            'fc': self.fc,
        }
        self.rc = RcSection(**inputs)
        self.rebar_od = 32
        self.cover = 165
        self.rebar_pos_y1 = self.thk - self.cover - self.rebar_od / 2
        self.rebar_pos_y2 = self.cover + self.rebar_od / 2
        self.rc.add_rebar(self.rebar_od, 0, self.rebar_pos_y1)
        self.rc.add_rebar(self.rebar_od, 0, self.rebar_pos_y2,
                          compression=True)

    def test_simple(self):
        """Test instantiation.
        """
        self.assertEqual(type(self.rc), RcSection)

    def test_size_output(self):
        """Test the get_extents() method of the RcSection class.
        """
        test_output = (self.width, self.thk)
        self.assertEqual(test_output, self.rc.get_extents())

    @patch("rc.plt.show")
    def test_plot(self, mock_show):
        """Test the plotting method of the RcSection class.
        """
        mock_show.return_value = None
        fig, axis = self.rc.plot_section()
        self.assertNotEqual(fig.get_axes(), [])
        self.assertEqual(len(axis.patches), len(self.rc.rebars) + 1)

    def test_mat_props(self):
        """Test material property input.
        """
        mat_props = self.rc.get_mat_props()
        self.assertIsInstance(mat_props, dict)
        self.assertEqual(mat_props['concrete']['fc'], self.fc)
        self.assertEqual(mat_props['rebar0']['Es'], 200000)

    def test_max_tension(self):
        test_max_tension, ety = self.rc.get_max_tension_P()
        rebar_area = np.pi / 4 * self.rebar_od ** 2
        max_tension = -2 * steel_sy * rebar_area
        self.assertEqual(max_tension, test_max_tension)

    def test_max_compression(self):
        test_max_compression, efc = self.rc.get_max_compression_P()
        rebar_area = np.pi / 4 * self.rebar_od ** 2
        max_compression = steel_sy * rebar_area + \
                          0.85 * self.fc * (self.width * self.thk - 2 * rebar_area)
        self.assertEqual(max_compression, test_max_compression)

    def test_get_P(self):
        rebar_area = np.pi / 4 * self.rebar_od ** 2
        max_compression = (self.width * self.thk - 2 * rebar_area) * 0.85 * self.fc + \
                          rebar_area * steel_sy
        max_tension = -2 * rebar_area * steel_sy
        test_e_top = -0.005
        test_e_bot = 0.003
        c = -0.003 / (test_e_top - test_e_bot) * self.thk
        a = self.beta_1 * c
        test_e_rebar1 = test_e_bot + \
                        self.rebar_pos_y1 / self.thk * (test_e_top - test_e_bot)
        test_e_rebar2 = test_e_bot + \
                        self.rebar_pos_y2 / self.thk * (test_e_top - test_e_bot)
        test_P_rebar2 = rebar_area * (min(test_e_rebar2 * steel_Es, steel_sy) - 0.85 * self.fc)
        test_P = 0.85 * a * self.fc * self.width + test_P_rebar2 + \
                 max(test_e_rebar1 * steel_Es, -steel_sy) * rebar_area
        test_cases = [(0.003, 0.003, max_compression),
                      (-0.003, -0.003, max_tension),
                      (test_e_top, test_e_bot, test_P),
                      ]
        for e_top, e_bot, P in test_cases:
            msg_string = "e_top = {}, e_bot = {}, P = {}".format(e_top,
                                                                 e_bot,
                                                                 P)
            self.assertEqual(P, self.rc.get_P((e_top, e_bot)), msg=msg_string)

    def test_no_rebar_cases(self):
        rc = RcSection()
        strains = (-0.003, -0.003)
        self.assertEqual(0, rc.get_P(strains))
        self.assertEqual(0, rc.get_M(strains))

    def test_get_beta_1(self):
        self.assertEqual(0.65, get_beta_1(100))
        self.assertEqual(0.85, get_beta_1(10))
        self.assertEqual(0.75, get_beta_1(42))

    def test_rebar_force(self):
        rebar = {'area': 10, 'y': 2, 'Es': 1,
                 'sy': 0.7, 'e_y': 0.7, 'compression': False}
        thk = 10
        e_top = -1
        e_bot = 0
        test_f = rebar_force(rebar, thk, e_top, e_bot)
        self.assertEqual(test_f, -2)

        e_top = 1
        e_bot = 0
        test_f = rebar_force(rebar, thk, e_top, e_bot, fc=1, e_fc=0.2)
        self.assertAlmostEqual(test_f, -8.5)

        rebar['compression'] = True
        test_f = rebar_force(rebar, thk, e_top, e_bot, fc=1, e_fc=0.2)
        self.assertAlmostEqual(test_f, 2 - 8.5)

    def test_conc_force(self):
        thk = 10
        width = 1
        beta_1 = 1
        e_top = 0.003
        e_bot = 0.003
        test_c, y_c = conc_force(thk, width, e_top, e_bot,
                                 beta_1, fc=10, e_fc=0.003)
        self.assertRaises(ZeroDivisionError)
        self.assertEqual(85, test_c)
        self.assertEqual(0, y_c)

        e_top = 0
        e_bot = -0.003
        test_c, y_c = conc_force(thk, width, e_top, e_bot, beta_1,
                                 fc=10, e_fc=0.003)
        self.assertEqual(0, test_c)
        self.assertEqual(0, y_c)

        e_top = -0.003
        e_bot = 0.003
        test_c, y_c = conc_force(thk, width, e_top, e_bot, beta_1,
                                 fc=10, e_fc=0.003)
        self.assertEqual(test_c, 0.85 * 5 * 10)
        self.assertEqual(y_c, -2.5)

        e_top = 0.003
        e_bot = -0.003
        test_c, y_c = conc_force(thk, width, e_top, e_bot, beta_1,
                                 fc=10, e_fc=0.003)
        self.assertEqual(test_c, 0.85 * 5 * 10)
        self.assertEqual(y_c, 5 - (5 / 2))

    def test_get_M(self):
        rebar_area = np.pi / 4 * self.rebar_od ** 2
        self.rc.rebars['compression'] = False
        test_cases = []

        # Pure tension case
        test_e_top = -0.003
        test_e_bot = -0.003
        test_M = 0
        test_cases.append((test_e_top, test_e_bot, test_M))

        # Pure compression case
        test_e_top2 = 0.003
        test_e_bot2 = 0.003
        test_M2 = 0
        test_cases.append((test_e_top2, test_e_bot2, test_M2))

        # Bending case 1
        test_e_top3 = -0.003
        test_e_bot3 = 0.003
        abs_c_from_bot = (0 - test_e_bot3) / (test_e_top3 - test_e_bot3) * self.thk
        abs_a_from_bot = self.beta_1 * abs_c_from_bot
        conc_P = abs_a_from_bot * 0.85 * self.fc * self.width
        conc_centroid = -(self.thk / 2 - abs_a_from_bot / 2)
        tens_rebar_strain = self.rebar_pos_y1 / self.thk * (test_e_top3 - test_e_bot3) + \
                            test_e_bot3
        tens_rebar_P = max(tens_rebar_strain, -steel_sy / steel_Es) * steel_Es * rebar_area
        tens_rebar_centroid = self.rebar_pos_y1 - self.thk / 2
        comp_rebar_P = -0.85 * self.fc * rebar_area
        comp_rebar_centroid = self.rebar_pos_y2 - self.thk/2
        test_M3 = conc_P * conc_centroid + tens_rebar_P * tens_rebar_centroid + \
            comp_rebar_P * comp_rebar_centroid
        test_cases.append((test_e_top3, test_e_bot3, test_M3))

        # Bending case 2
        test_e_top4 = 0.003
        test_e_bot4 = -0.003
        abs_c_from_top = (test_e_top4 - 0) / (test_e_top4 - test_e_bot4) * self.thk
        abs_a_from_top = self.beta_1 * abs_c_from_top
        conc_P = abs_a_from_top * 0.85 * self.fc * self.width
        conc_centroid = self.thk / 2 - abs_a_from_top / 2
        tens_rebar_strain = self.rebar_pos_y2 / self.thk * (test_e_top4 - test_e_bot4) + \
                            test_e_bot4
        tens_rebar_P = max(tens_rebar_strain, -steel_sy / steel_Es) * steel_Es * rebar_area
        tens_rebar_centroid = self.rebar_pos_y2 - self.thk / 2
        comp_rebar_P = -0.85 * self.fc * rebar_area
        comp_rebar_centroid = self.rebar_pos_y1 - self.thk / 2
        test_M4 = conc_P * conc_centroid + tens_rebar_P * tens_rebar_centroid + \
            comp_rebar_P * comp_rebar_centroid
        test_cases.append((test_e_top4, test_e_bot4, test_M4))

        for e_top, e_bot, M in test_cases:
            msg_string = "e_top = {}, e_bot = {}, M = {}".format(e_top,
                                                                 e_bot,
                                                                 M)
            self.assertEqual(M, self.rc.get_M((e_top, e_bot)), msg=msg_string)


# %% Testcases for resp_spect.py


class TestResponseSpectrum(unittest.TestCase):
    pass


class TestBroadbanding(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
