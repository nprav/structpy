# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 07:11:59 2019

@author: Praveer Nidamaluri

Script for testing other scripts in the module.
"""

# %% Import Necessary Modules

import unittest
from rc import RcSection, Slab, RectangularBeam

# %% Testcases for rc.py

class TestRC(unittest.TestCase):

    def setUp(self):
        self.width = 1000
        self.thk = 1750
        self.rc = RcSection(self.width, self.thk)

    def test_simple(self):
        self.assertEqual(type(self.rc), RcSection)

    def test_size_output(self):
        test_output = (self.width, self.thk)
        self.assertEqual(test_output, self.rc.get_extents())

    def test_plot(self):
        fig, axis = self.rc.plot()
        self.assertNotEqual(fig.get_axes(), [])

#    def test_rebar(self):

# %% Testcases for resp_spect.py

class TestResponseSpectrum(unittest.TestCase):
    pass

class TestBroadbanding(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
