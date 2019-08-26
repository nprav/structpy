'''
Created: Nov 2017
Latest update:  Sep 2018
@author: Praveer Nidamaluri

Module that provides file read/write functions for shake/dmod outputs
'''


#

def read_shk_ahl(filename,header = 3):
    '''
    Input = ( filename, header = 3)
    header = number of lines to skip at the start of the file (default = 0)
    Output = list of values as floats
    '''

    file = open(filename)
    ahl = []
    file.seek(0,0)
    s = False
    for i,line in enumerate(file):
        if i >= header:
            s = True
        if s:
            ahl += [float(k) for k in line.split()]

    file.close()

    return ahl


def write_acc(acc,filename,header = 8):
    '''
    Input = ( list/array of acc files, filename, header = 8)
    header = number of lines to skip at the start of the file (default = 0)
    Output = Nothing, a file with the data in fortran 8f9.6 format is created
    '''

    import numpy as np

    file = open(filename,'w')
    file.seek(0,0)
    file.write("\n"*header)

    if len(acc)%8>0:
        acc = np.append(acc,[0]*(8-len(acc)%8))

    for i,num in enumerate(acc):
        file.write("%9.6f" %(num))
        if (i+1) %8 == 0:
            file.write("\n")

    file.close()


def read_dmd_acc(filename):
    '''
    Input = ( dmod acc output filename)
    Output = 2 lists. 1st list = list of floats with surface TH, 2nd list = list of floats with base TH
    '''
    file = open(filename,'r')
    acc_surf = []
    acc_base = []
    file.seek(0,0)
    s = False
    tstep = 100
    for i,line in enumerate(file):
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


def read_fort_txt(filename,header = 8, cols = 8, dgts = 9):
    '''
    Input = ( filename, #header lines,#data columns, #digits per column)
    Output = list of values as floats
    '''

    file = open(filename)
    acc = []
    file.seek(0,0)
    s = False
    for i,line in enumerate(file):
        if i >= header:
            s = True
        if s:
            while line != "\n":
                acc.append(float(line[:dgts]))
                line = line[dgts:]

    file.close()

    return acc

def read_TH_inp(filename):
    '''
    Input = filename of Time History file (assumes a format of "time   value \\n"), with no header lines
    Output = 2 lists of values as floats, first is time, second is acc value
    '''

    file = open(filename)
    acc = []
    tm = []
    file.seek(0,0)
    for line in file:
        tm.append(float(line.split()[0]))
        acc.append(float(line.split()[-1]))

    file.close()

    return tm,acc


def write_lcid(acc,filename,tstep = 0.005,gap = 2):
    '''
    Writes xy data files for LS-DYNA lcid input.
    Input: list of accelerations, name of xydata file (should have .csv at the end), timestep for time values, gap = # of seconds to pad 0s
    to pad acceleration vector
    Output: Nothing, writes the file
    '''

    file = open(filename,'w')

    if gap > 0:
        file.write("%10f,%10f \n" %(0,0))
        file.write("%10f,%10f \n" %(gap,0))

    for i, a in enumerate(acc):
        t = tstep*(i+1)+gap
        file.write("%10f,%10f \n" %(t,a))

    file.close()

def read_csv(filename,header = 1):
    '''
    Reads .csv files with x, y data
    Input: filename, header lines to skip (assume = 1)
    Output: time list, acc. list
    '''

    file = open(filename,'r')

    acc = []
    tm = []
    file.seek(0,0)
    s = False
    for i,line in enumerate(file):
        if i >= header:
            s = True
        if s:
            tm.append(float(line.split(',')[0]))
            acc.append(float(line.split(',')[1]))

    file.close()

    return tm, acc


def read_csv_multi(filename,header = 1):
    '''
    Reads .csv files with one x column, and multiple y columns
    Input: filename, header lines to skip (assume = 1)
    Output: time list, acc. list
    '''

    file = open(filename,'r')

    file.seek(0,0)
    s = False
    for i,line in enumerate(file):
        if i >= header:
            s = True
        if s:
            num_cols = len(line.split(',')) - 2
            break

    acc = []
    for i in range(0,num_cols):
        acc.append([])

    tm = []
    file.seek(0,0)
    s = False
    for i,line in enumerate(file):
        if i >= header:
            s = True
        if s:
            tm.append(float(line.split(',')[0]))
            for j in range(0,num_cols):
                acc[j].append(float(line.split(',')[j+1]))

    file.close()

    return tm, acc


def write_csv(filename,x,y,title = "",txt="",col1="x",col2="y"):
    '''
    Writes a file with x, y data; delimited by commas; will automatically have 3 header lines.
    Input: filename string = must have the full directory with the filename and extension (.txt or .csv, etc.) with forward
    slashes only; x = list like object with x data; y = list like object with y data; title string = string that goes in the first line;
    txt = string going in the second line; col1 string = name of
    column 1 (x data); col2 string = name of column 2 (y data)
    Output: None, file is created
    '''

    file = open(filename,'w')

    file.write(title+"\n")
    file.write(txt+"\n")
    file.write(col1+", "+col2+"\n")

    length = min(len(x),len(y))
    for i in range(0,length):
        file.write("%s, %s \n" %(x[i],y[i]))

    file.close()


def read_rsp(filename):
    '''
    Input = filename with Rspmatch spc file
    Output = list with time period, list with abs. acc response in g's
    '''

    file = open(filename,'r')
    tp = []
    rs = []
    file.seek(0,0)
    s = False
    for i,line in enumerate(file):
        if i >= 14:
            s = True
        if s and line != "\n":
            tp.append(float(line.split()[0]))
            rs.append(float(line.split()[-2]))
    file.close()

    return tp, rs


def read_curve(filename,header = 8):
    '''
    Reads LS-DYNA curve files with x, y data
    Input: filename with extension, header lines to skip (assume = 8)
    Output: time list, acc. list
    '''

    file = open(filename,'r')

    acc = []
    tm = []
    file.seek(0,0)
    s = False
    for i,line in enumerate(file):
        if i >= header:
            s = True
        if s:
            try:
                tm.append(float(line.split()[0]))
                acc.append(float(line.split()[1]))
            except:
                if line.find("end") >= 0:
                    break
    file.close()

    return tm, acc