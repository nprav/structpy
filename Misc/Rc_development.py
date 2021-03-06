"""
Scratchpad script for testing out implementations before adding them
to the actual script files.
"""

# %% Make and test an RC section

from rc import RcSection, rebar_force, get_beta_1, conc_force
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brute
from matplotlib import cm
from mpl_toolkits import mplot3d




# Test basic plotting
test_rc = RcSection()
test_rc.add_rebar()
test_rc.add_rebar(y=25, compression=True)
print(test_rc.get_extents())
print(test_rc.get_mat_props())
# test_rc.plot_section()


# Test rebar stress evaluations
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

# Plot get_P with a simple section
x = np.linspace(-0.005, 0.003, 50, endpoint=True)
y = np.linspace(-0.005, 0.003, 50, endpoint=True)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(rc.get_P)(X, Y)

fig = plt.Figure()
ax = plt.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
fig.colorbar(surf)
plt.show()


# Testing get_M testing function in test.py
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
             compression=False)

rebar_area = np.pi/4 * rebar_od**2
test_e_top = 0
test_e_bot = 0

rebar_P = rc.rebars.apply(
            rebar_force, axis=1, args=(rc.thk, test_e_top, test_e_bot),
            **rc.conc_matprops,
        )
rebar_cent = rc.rebars['y'] - rc.thk/2

conc_P, conc_cent = conc_force(rc.thk, rc.width,
                                       test_e_top, test_e_bot, rc.beta_1,
                                       **rc.conc_matprops)

x = np.linspace(-0.005, 0.003, 50, endpoint=True)
y = np.linspace(-0.005, 0.003, 50, endpoint=True)
X, Y = np.meshgrid(x, y)
Z = np.vectorize(lambda x, y: rc.get_M((x, y)))(X, Y)

fig = plt.Figure()
ax = plt.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
fig.colorbar(surf)
plt.show()

# Testing max_compression function in rc.py
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

compr_rebars = rc.rebars[rc.rebars['compression']]
max_conc_compression = 0.85 * (thk * width -
                                       np.sum(rc.rebars['area'])) * \
                               rc.conc_matprops['fc'] + \
                               np.sum(compr_rebars['area'] * compr_rebars['sy'])
if not compr_rebars.empty:
    efc = max(compr_rebars['e_y'].max(), rc.conc_matprops['e_fc'])
else:
    efc = rc.conc_matprops['e_fc']



# Debug nan issues
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
             compression=False)
test_strains = np.zeros(2)
test_e_top, test_e_bot = test_strains

if not rc.rebars.empty:
    rebar_P = rc.rebars.apply(
        rebar_force, axis=1, args=(rc.thk, test_e_top, test_e_bot),
        **rc.conc_matprops,
    )
    P = np.sum(rebar_P)
else:
    P = 0
conc_P, conc_cent = conc_force(rc.thk, rc.width,
                               test_e_top, test_e_bot, rc.beta_1,
                               **rc.conc_matprops)
P = P + conc_P




# Testing get M from P using minimize from scipy
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
rc.add_rebar(rebar_od, 0, rebar_pos_y2)

npts = 50
max_tension = rc.get_max_tension_P()[0]
max_compression = rc.get_max_compression_P()[0]
tol = (max_compression - max_tension)/npts * 0.5
test_P = max_compression*0.7
constraints = [
    # {'type': 'ineq', 'fun': lambda x: tol - abs(rc.get_P(x) - test_P)},
    {'type': 'eq', 'fun': lambda x: rc.get_P(x) - test_P},
]
bounds = [(-0.005, 0.003)]*2
x0 = (0.002, 0.0021)
options = {'disp': True, 'maxiter': 1000}

res = minimize(rc.get_M, x0=x0,
               constraints=constraints, bounds=bounds,
               options=options)



# Testing minmax strain limits
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
rc.add_rebar(rebar_od, 0, rebar_pos_y2)

min_top_str = (-rc.rebars['sy']/rc.rebars['Es'] - rc.conc_matprops['e_fc']) \
              / rc.rebars['y'] * rc.thk
max_top_str = rc.conc_matprops['e_fc']

min_bot_str = (-rc.rebars['sy']/rc.rebars['Es'] - rc.conc_matprops['e_fc']) \
              / (rc.rebars['y'] - rc.thk) * (-rc.thk) + rc.conc_matprops['e_fc']

max_bot_str = rc.conc_matprops['e_fc']



# Test classic interaction diagram method
steel_sy = 72500  # psi
steel_Es = 29e6   # psi
width = 200/25.4    # in
thk = 1750/25.4   # in
fc = 4512   # psi
inputs = {
    'width': width,
    'thk': thk,
    'fc': fc,
}
rc = RcSection(**inputs)
rebar_od = 1.27     # in
cover = 6.5     # in
rebar_pos_y1 = thk - cover - rebar_od / 2
rebar_pos_y2 = cover + rebar_od / 2

rc.add_rebar(rebar_od, 0, rebar_pos_y1, compression=True,
             sy=steel_sy, Es=steel_Es)
rc.add_rebar(rebar_od, 0, rebar_pos_y2,
             sy=steel_sy, Es=steel_Es)
rc.add_rebar(rebar_od, 0, rebar_pos_y2,
             sy=steel_sy, Es=steel_Es)
rc.add_rebar(rebar_od, 0, rebar_pos_y2,
             sy=steel_sy, Es=steel_Es)


top_str_limits, bot_str_limits = rc.get_strain_limits()

npts = 50
id_neg = pd.DataFrame(columns=rc.id_column_labels)
bot_str = bot_str_limits[1]
alpha = thk/get_beta_1(fc)
start_top_str = top_str_limits[1]/alpha * (alpha-thk)
raw_spacing = (np.geomspace(1, 101, npts-5) - 1)/100
spacing = raw_spacing * (top_str_limits[0] - start_top_str) + start_top_str
for top_str in spacing:
    P = rc.get_P((top_str, bot_str))
    M = rc.get_M((top_str, bot_str))
    id_neg.loc[len(id_neg)] = [P, M, top_str, bot_str]

top_str = top_str_limits[0]
for bot_str in np.linspace(0, bot_str_limits[1], 5, endpoint=False)[::-1]:
    P = rc.get_P((top_str, bot_str))
    M = rc.get_M((top_str, bot_str))
    id_neg.loc[len(id_neg)] = [P, M, top_str, bot_str]

plt.plot(id_neg.M, id_neg.P, 'x-')


npts = 50
id_pos = pd.DataFrame(columns=rc.id_column_labels)
top_str = top_str_limits[1]
alpha = thk/get_beta_1(fc)
start_bot_str = bot_str_limits[1]/alpha * (alpha-thk)
raw_spacing = (np.geomspace(1, 101, npts-5) - 1)/100
spacing = raw_spacing * (bot_str_limits[0] - start_bot_str) + start_bot_str
for bot_str in spacing:
    P = rc.get_P((top_str, bot_str))
    M = rc.get_M((top_str, bot_str))
    id_pos.loc[len(id_pos)] = [P, M, top_str, bot_str]

bot_str = bot_str_limits[0]
for top_str in np.linspace(0, top_str_limits[1], 5, endpoint=False)[::-1]:
    P = rc.get_P((top_str, bot_str))
    M = rc.get_M((top_str, bot_str))
    id_pos.loc[len(id_pos)] = [P, M, top_str, bot_str]

plt.plot(id_pos.M, id_pos.P, 'x-')

interaction_diagram = id_pos.append(id_neg[::-1])

plt.plot(interaction_diagram.M, interaction_diagram.P)