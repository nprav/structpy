"""
Scratchpad script for testing out implementations before adding them
to the actual script files.
"""

class testing(object):

    def __init__(self, var=dict()):
        self._var = var

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, value):
        print("value set")
        self._var = value

test_obj = testing()
test_obj.var['hello'] = 10
print(test_obj.var)



# %% Make and test an RC section

from rc import RcSection

test_rc = RcSection()
test_rc.add_rebar()
test_rc.add_rebar(y=25)
print(test_rc.get_extents())
print(test_rc.get_mat_props())
test_rc.plot()