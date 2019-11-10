'''
Created: Nov 2017
Latest update:  Aug 2019
@author: Praveer Nidamaluri

Module that provides file read/write functions for
shake/dmod/lsdyna/csv inputs and outputs.
'''

# %% Import required modules
import numpy as np
import re


# %% Read functions

def read_shk_ahl(filename, header=3):
    '''Read SHAKE .ahl time history output files.

    Parameters
    ----------
    filename : str
        Address of .ahl file with filename.

    header : int>=0, optional
        Number of lines to skip at the start of the file.
        Should be an integer greater than 0. Defaults to 3.

    Returns
    -------
    ahl : 1D list of floats
        List of acceleration values from the .ahl file.
    '''

    file = open(filename)
    ahl = []
    file.seek(0, 0)
    s = False
    for i, line in enumerate(file):
        if i >= header:
            s = True
        if s:
            ahl += [float(k) for k in line.split()]

    file.close()

    return ahl


def read_dmd_acc(filename):
    '''Read surface and base time histories from D-MOD .acc file.

    Parameters
    ----------
    filename : str
        Address of .acc file with filename.

    Returns
    -------
    acc_surf : 1D list fo floats
        List of acceleration values of the surface (1st layer) time history.

    acc_base : 1D list of floats
        List of acceleration values of the base (last layer) time history.
    '''

    file = open(filename, 'r')
    acc_surf = []
    acc_base = []
    file.seek(0, 0)
    s = False
    tstep = 100
    for i, line in enumerate(file):
        if not s and i == 5:
            tstep = line.split()[-1]
        if not s and i > 5:
            if line.split()[0] == tstep:
                s = True
        if s:
            if line.split()[0] == "0":
                file.close()
                break
            acc_surf.append(float(line.split()[1]))
            acc_base.append(float(line.split()[-1]))
    return acc_surf, acc_base


def read_fort_txt(filename, header=8, cols=8, dgts=9):
    '''Extract time histories from text files in fortran format.

    Defaults to '8F9.6' format.

    Parameters
    ----------
    filename : str
        Address of file with filename and extension.

    header : int>=0, optional
        Number of lines to skip at the start of the file.
        Should be an integer greater than 0. Defaults to 8.

    cols : int>=0, optional
        Number of columns in fortran format.
        Should be an integer greater than 0. Defaults to 8.

    dgts : int>=0, optional
        Number of digits per value in fortran format.
        Should be an integer greater than 0. Defaults to 9.

    Returns
    -------
    acc : 1D list of floats
        List of values from the file.
    '''

    file = open(filename)
    acc = []
    file.seek(0, 0)
    s = False
    for i, line in enumerate(file):
        if i >= header:
            s = True
        if s:
            while line != "\n":
                acc.append(float(line[:dgts]))
                line = line[dgts:]

    file.close()

    return acc


def read_TH_inp(filename):
    '''Read default time history input files. Assumes a format of
    r"time   value \n" on each line, with no header lines.

    Output = 2 lists of values as floats, first is time, second is acc value
    Extract time histories.

    Parameters
    ----------
    filename : str
        Address of file with filename and extension.

    Returns
    -------
    tm : 1D list of floats
        List of abscissa values from the input file.

    acc : 1D list of floats
        List of ordinate values from the input file.

    '''

    file = open(filename)
    acc = []
    tm = []
    file.seek(0, 0)
    for line in file:
        tm.append(float(line.split()[0]))
        acc.append(float(line.split()[-1]))

    file.close()

    return tm, acc


def read_csv(filename, header=1):
    '''Reads .csv files with just two columns.

    Parameters
    ----------
    filename : str
        Address of .csv file with filename.

    header : int>=0, optional
        Number of lines to skip at the start of the file.
        Should be an integer greater than 0. Defaults to 2.

    Returns
    -------
    tm : 1D list of floats
        List of abscissa values from the input file.

    acc : 1D list of floats
        List of ordinate values from the input file.
    '''

    file = open(filename, 'r')

    acc = []
    tm = []
    file.seek(0, 0)
    s = False
    for i, line in enumerate(file):
        if i >= header:
            s = True
        if s:
            tm.append(float(line.split(',')[0]))
            acc.append(float(line.split(',')[1]))

    file.close()

    return tm, acc


def read_csv_multi(filename, header=1):
    '''Reads .csv files with just multiple columns.

    Parameters
    ----------
    filename : str
        Address of .csv file with filename.

    header : int>=0, optional
        Number of lines to skip at the start of the file.
        Should be an integer greater than 0. Defaults to 2.

    Returns
    -------
    tm : 1D list of floats
        List of abscissa values from the file.

    acc : 2D list of floats
        List of ordinate values from the file. Values from each
        abscissa column of the csv file are contained in a separate list
        within `acc`.
    '''

    file = open(filename, 'r')

    file.seek(0, 0)
    s = False
    for i, line in enumerate(file):
        if i >= header:
            s = True
        if s:
            num_cols = len(line.split(',')) - 2
            break

    acc = []
    for i in range(0, num_cols):
        acc.append([])

    tm = []
    file.seek(0, 0)
    s = False
    for i, line in enumerate(file):
        if i >= header:
            s = True
        if s:
            tm.append(float(line.split(',')[0]))
            for j in range(0, num_cols):
                acc[j].append(float(line.split(',')[j+1]))

    file.close()

    return tm, acc


def read_rsp(filename):
    '''Read SHAKE/Rspmatch .spc response spectrum output file.

    Input = filename with Rspmatch spc file
    Output = list with time period, list with abs. acc response in g's

    Parameters
    ----------
    filename : str
        Address of .spc file with filename.

    Returns
    -------
    tp : 1D list fo floats
        List of time period values.

    rs : 1D list of floats
        List of spectral acceleration values.
    '''

    file = open(filename, 'r')
    tp = []
    rs = []
    file.seek(0, 0)
    s = False
    for i, line in enumerate(file):
        if i >= 14:
            s = True
        if s and line != "\n":
            tp.append(float(line.split()[0]))
            rs.append(float(line.split()[-2]))
    file.close()

    return tp, rs


def read_peer_record(filename):
    """Read time history file from PEER Strong Motion Record
    Database.

    :param filename: string with file location of record.
    :return: dictionary with time history record and peroperties
    """

    eq_record = {}

    with open(filename, 'r') as file:
        # Skip the first header line
        file.readline()

        line2 = file.readline()
        props2 = line2.split(',')
        eq_record.update(
            dict(zip(['name', 'date', 'array', 'direction'], props2))
        )
        # remove '\n' from end of direction string
        eq_record['direction'] = eq_record['direction'].split()[0]

        line3 = file.readline()
        props3 = line3.split()
        eq_record['type'] = props3[0].lower()
        eq_record['units'] = props3[-1].lower()

        line4 = file.readline()
        eq_record['npts'] = int(re.findall('(?<=NPTS=)[\d ]+(?=,)', line4)[0])
        eq_record['dt'] = float(re.findall('(?<=DT=).+(?=SEC)', line4)[0])

        # Read in time history
        th = []
        for line in file:
            th += list(map(float, line.split()))
        eq_record['th'] = th

    return eq_record


def read_curve(filename, header=8):
    '''Reads LS-DYNA curve files with x, y data.

    Parameters
    ----------
    filename : str
        Address of .ahl file with filename.

    header : int>=0, optional
        Number of lines to skip at the start of the file.
        Should be an integer greater than 0. Defaults to 8.

    Returns
    -------
    tm : 1D list of floats
        List of abscissa values from the input file.

    acc : 1D list of floats
        List of ordinate values from the input file.
    '''

    file = open(filename, 'r')

    acc = []
    tm = []
    file.seek(0, 0)
    s = False
    for i, line in enumerate(file):
        if i >= header:
            s = True
        if s:
            try:
                tm.append(float(line.split()[0]))
                acc.append(float(line.split()[1]))
            except Exception:
                if line.find("end") >= 0:
                    break
    file.close()

    return tm, acc


# %% Write functions

def write_csv(filename, x, y, title="", txt="", col1="x", col2="y"):
    '''Write a .csv file with x and y data, with descriptive text in 3
    header lines.

    Parameters
    ----------
    filename : str
        Address of target file with a filename and extension.

    x : 1D array_like
        Abscissa, x, values.

    y : 1D array_like
        Ordinate, y, values.

    title : str, optional
        Text that will be printed in the first line of the final csv.
        Defaults to "", i.e. nothing.

    txt : st, optional
        Text that will be printed on the second line of the final csv.
        Defaults to "", i.e. nothing.

    col1 : str, optional
        The abscissa column name. Defaults to 'x'.

    col2 : str, optional
        The ordinate column name. Defaults to 'y'.

    Returns
    -------
    None
    '''

    file = open(filename, 'w')

    file.write(title+"\n")
    file.write(txt+"\n")
    file.write(col1+", "+col2+"\n")

    length = min(len(x), len(y))
    for i in range(0, length):
        file.write("%s, %s \n" % (x[i], y[i]))

    file.close()


def write_acc(acc, filename, header=8, txt=''):
    '''Write a time history file in 8F9.6 Fortran format.

    Input = ( list/array of acc files, filename, header = 8, txt = '')
    header = number of lines to skip at the start of the file (default = 8, min of 1)
    txt = text in the first line
    Output = Nothing, a file with the data in fortran 8f9.6 format is created

    Parameters
    ----------
    acc : 1D array_like
        Input time history to be written into fortran format.

    filename : str
        Address of target file with a filename and extension.

    header : int>=1, optional
        Number of lines to skip at the start of the file.
        Should be an integer greater than 1. Defaults to 8.
	
    txt : str
        String input in first header line.	

    Returns
    -------
    None
    '''

    file = open(filename, 'w')
    header = max(1, header)

    file = open(filename, 'w')
    file.seek(0, 0)
    file.write(txt + '\n')
    file.write("\n"*(header - 1))

    if len(acc) % 8 > 0:
        acc = np.append(acc, [0]*(8-len(acc) % 8))

    for i, num in enumerate(acc):
        file.write("%9.6f" % (num))
        if (i+1) % 8 == 0:
            file.write("\n")

    file.close()


def write_lcid(acc, filename, tstep=0.005, gap=2):
    '''Writes a time history csv file with the option to pad
    the first `gap` seconds with 0s.

    Parameters
    ----------
    acc : 1D array_like
        Input time history to be written.

    filename : str
        Address of target file with a filename and extension (.csv).

    tstep : float, optional
        Timestep for the time values in the final time history.
        The time vector is assumed to be in equal increments of `tstep`.
        Defaults to 0.005s, a common value for earthquake records.

    gap : numeric, optional
        The first `gap` seconds are padded with 0s before the actual
        acceleration values are printed. Defaults to 2s.

    Returns
    -------
    None
    '''

    file = open(filename, 'w')

    if gap > 0:
        file.write("%10f,%10f \n" % (0, 0))
        file.write("%10f,%10f \n" % (gap, 0))

    for i, a in enumerate(acc):
        t = tstep*(i+1)+gap
        file.write("%10f,%10f \n" % (t, a))

    file.close()
