# structpy
Python based tools for structural analyses

StructPy is intended as a Python-based package for use by mechanical/structural engineers. 
The package currently contains the following modules:

- structpy.resp_spect:

The resp_spect module contains two different implementations of an acceleration response spectrum generator.

- structpy.asme:

The asme module is used to filter through material properties from the ASME Boiler and Pressure Vessel Code, Section II Part D 
material property tables. The tables are assumed to be available as excel/csv files. The module functions can further interpolate
and extrapolate temperature dependent material properties.

The package requires the following modules to run:
- numpy
- scipy
- matplotlib


The Misc. folder contains other structural engineering related tools. The tools may 
eventually be converted to new modules/functions in the structpy package.

Misc. tools (as of 11/11/2018):
- Generate a Force/Moment interaction diagram for a doubly-reinforced reinforced concrete section (Jupyter Notebook)