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
