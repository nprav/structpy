"""
Scratchpad script for testing out implementations before adding them
to the actual script files.
"""

# %% Make and test an RC section

from rc import RcSection, rebar_force, get_beta_1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_rc = RcSection()
test_rc.add_rebar()
test_rc.add_rebar(y=25, compression=True)
print(test_rc.get_extents())
print(test_rc.get_mat_props())
# test_rc.plot_section()

e_top = 0.003
e_bot = 0.003
thk = test_rc.thk
rebar_P = test_rc.rebars.apply(
    rebar_force, axis=1, args=(test_rc.thk, e_top, e_bot)
)
print(rebar_P)

# Testing get_P testing function in test.py
steel_sy = 500
steel_Es = 200000
width = 200
thk = 1750
fc = 40
inputs = {
    'width': width,
    'thk': thk,
    'fc': fc,
}
rc = RcSection(**inputs)
rebar_od = 32
cover = 165
rebar_pos_y1 = thk - cover - rebar_od / 2
rebar_pos_y2 = cover + rebar_od / 2
rc.add_rebar(rebar_od, 0, rebar_pos_y1)
rc.add_rebar(rebar_od, 0, rebar_pos_y2,
             compression=True)

rebar_area = np.pi/4 * rebar_od**2
test_e_top = -0.005
test_e_bot = 0.003
c = -0.003 / (test_e_top - test_e_bot) * thk
beta_1 = get_beta_1(fc)
a = beta_1 * c
test_e_rebar1 = test_e_bot + \
                rebar_pos_y1 / thk * (test_e_top - test_e_bot)
test_P_rebar1 = max(test_e_rebar1*steel_Es, -steel_sy)*rebar_area
test_e_rebar2 = test_e_bot + \
                rebar_pos_y2 / thk * (test_e_top - test_e_bot)
test_P_rebar2 = rebar_area*(min(test_e_rebar2 * steel_Es, steel_sy) - 0.85 * fc)
test_P = 0.85 * a * fc * width + test_P_rebar2 + test_P_rebar1