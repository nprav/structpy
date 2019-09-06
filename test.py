# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 07:11:59 2019

@author: Praveer Nidamaluri

Script for testing other scripts in the module.
"""

# %% Import Necessary Modules

import unittest
from unittest.mock import patch
from rc import RcSection

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
        self.fc=40
        inputs = {
            'width': self.width,
            'thk': self.thk,
            'fc': self.fc,
            }
        self.rc = RcSection(**inputs)
        self.rebar_od = 32
        self.cover = 165
        self.rebar_pos_y = self.thk - self.cover - self.rebar_od/2
        self.rc.add_rebar(self.rebar_od, 0, self.rebar_pos_y)

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
        fig, axis = self.rc.plot()
        self.assertNotEqual(fig.get_axes(), [])
        self.assertEqual(len(axis.patches), len(self.rc.rebars)+1)

    def test_mat_props(self):
        """Test material property input.
        """
        mat_props = self.rc.get_mat_props()
        self.assertIsInstance(mat_props, dict)
        self.assertEqual(mat_props['concrete']['fc'], self.fc)
        self.assertEqual(mat_props['rebar0']['Es'], 2000)



# %% Testcases for resp_spect.py


class TestResponseSpectrum(unittest.TestCase):
    pass


class TestBroadbanding(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
