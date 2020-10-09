# structpy
Python based tools for structural analyses

StructPy is intended as a Python-based package for use by mechanical/structural engineers. 
The package currently contains the following modules:

- structpy.resp_spect:

The resp_spect module contains two different implementations of an acceleration response spectrum generator.

- structpy.broadband:

Used to broadband (per ASCE 4-16) and simplify response spectra.

- structpy.rc:

Generate `RcSection` objects representing Reinforced Concrete beams. Sections and rebar patterns can be visualized, and Force/Moment Interaction diagrams can be generated.

- structpy.asme:

The asme module is used to filter through material properties from the ASME Boiler and Pressure Vessel Code, Section II Part D 
material property tables. The tables are assumed to be available as excel/csv files. The module functions can further interpolate
and extrapolate temperature dependent material properties.

- structpy.timehistory

Time history post-processing tools including filtering and animation functions.

The package requires the following modules to run:
- numpy
- scipy
- pandas
- matplotlib
