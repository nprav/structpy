'''
Created: oct 2018
Latest update:  Oct 2018
@author: Praveer Nidamaluri

Module for extracting material property data from ASME BPVC Section II excel tables
'''

import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import os
import re

module_path = os.path.dirname(__file__)
asme_path = os.path.join(module_path, 'ASME')
yld_file = [file for file in os.listdir(asme_path) if 'yield' in file.lower()][0]
ult_file = [file for file in os.listdir(asme_path) if 'tensile' in file.lower()][0]
yld_file_path = os.path.join(asme_path, yld_file)
ult_file_path = os.path.join(asme_path, ult_file)


def sy(crit_temps = [], condensed = False, imp_cols = [], **kwargs):
    '''
    Function to filter and interpolate/extrapolate ASME yield strength data. The function assumes 
    that the ASME material property information is available in an excel file with the path stored 
    in the global variable, yld_file_path, defined under the ASME submodule of structpy.
    
    The function takes the following optional inputs:
    
    crit_temps = a list of temperatures to extract strength data. If data at the temperature does not exist, 
    it will be interpolated/extrapolated
    
    condensed = a boolean (default = False). If True, only the Spec_No, strength vs temperature, imp_cols, and kwargs 
    filter keys will be presented. All other columns available in the ASME tables will not be returned.
    
    imp_cols = a list of columns to include in the final condensed dataframe. This is an optional parameter 
    and only required if condensed = True.
    
    kwargs = keyword filters to apply to the data. Valid keywords are all the ASME informational columns (see list below).
    Unrecognized columns are ignored.
    
    Valid kwargs:
    Spec_No
    Nominal_Composition
    Product_Form
    Type_Grade
    Alloy_Desig_UNS_No
    Class_Cond_Temper
    Size_thickness
    Min_Tensile_Strength
    Min_Yield_Strength
    Notes
    page_no
    line_no
    row_no
    
    eg input: sy(crit_temps = [300], Spec_No = 'SA-517', Type_Grade = 'P', condensed=True, imp_cols=['Size_Thickness'])
    '''
    
    # Extract yld strength dataframe and drop Addenda
    yld_data = pd.read_excel(yld_file_path,header=3)
    yld_data.drop('Addenda',axis=1,inplace=True)
    
    # Filter dataframe based on kwargs:
    data_filter = np.array([True]*yld_data.shape[0])
    for key, value in kwargs.items():
        try:
            data_filter = data_filter & (yld_data[key] == value)
            imp_cols.append(key)
        except:
            print('Unrecognized keyword : ', key)
            continue
    
    if condensed:
        useful_cols = list(set(['Spec_No'] + imp_cols))
    
    filtered_yld_data = yld_data[data_filter]
    
    # Preprocess temperature columns in dataframe
    temp_cols = list(filter(lambda x: re.search('^d[0-9]+$',x),yld_data.columns))
    temp_cols_dict = {col: int(col.strip('d')) for col in temp_cols}
    temp_cols_dict.update({'d_20_to_100':100})
    
    temp_list = list(temp_cols_dict.values())
    temp_list.sort()
    
    filtered_yld_data = filtered_yld_data.rename(columns=temp_cols_dict)

    # Remove any temps in crit_temps for which data is available in the ASME data, temp_list
    crit_temps = list(filter(lambda x: x not in temp_list, crit_temps))
    
    # If there are no temps to extrapolated/interpolate, return the filtered data
    if not crit_temps:
        return filtered_yld_data
       
    # Generate interpolation functions
    y = filtered_yld_data[temp_list].dropna(axis=1)
    X = y.columns
    
    interp_fn = interp1d(X,y,fill_value='extrapolate')
    
    # Generate linearly interpolated/extrapolated data
    interp_data = pd.DataFrame(interp_fn(crit_temps),columns=crit_temps,index=y.index)
    
    # Concatenate new data to the old temp. columns and resort
    new_temp_data = pd.concat([y,interp_data],axis=1)
    new_temp_data.sort_index(axis=1,inplace=True)
    
    # Generate the final dataframe by putting in all the non-temp data:
    if condensed:
        final_df = pd.concat([filtered_yld_data[useful_cols],new_temp_data],axis=1)
    else:
        final_df = pd.concat([filtered_yld_data.drop(temp_list,axis=1),new_temp_data],axis=1)

    return final_df


def su(crit_temps = [], condensed = False, imp_cols = [], **kwargs):
    '''
    Function to filter and interpolate/extrapolate ASME ultimate strength data. The function assumes 
    that the ASME material property information is available in an excel file with the path stored 
    in the global variable, ult_file_path, defined under the ASME submodule of structpy.
    
    The function takes the following optional inputs:
    
    crit_temps = a list of temperatures to extract strength data. If data at the temperature does not exist, 
    it will be interpolated/extrapolated
    
    condensed = a boolean (default = False). If True, only the Spec_No, strength vs temperature, imp_cols, and kwargs 
    filter keys will be presented. All other columns available in the ASME tables will not be returned.
    
    imp_cols = a list of columns to include in the final condensed dataframe. This is an optional parameter 
    and only required if condensed = True.
    
    kwargs = keyword filters to apply to the data. Valid keywords are all the ASME informational columns (see list below).
    Unrecognized columns are ignored.
    
    Valid kwargs:
    Spec_No
    Nominal_Composition
    Product_Form
    Type_Grade
    Alloy_Desig_UNS_No
    Class_Cond_Temper
    Size_thickness
    Min_Tensile_Strength
    Min_Yield_Strength
    Notes
    page_no
    line_no
    row_no
    
    eg input: su(crit_temps = [300], Spec_No = 'SA-517', Type_Grade = 'P', condensed=True, imp_cols=['Size_Thickness'])
    '''
    
    # Extract ult strength dataframe and drop Addenda
    ult_data = pd.read_excel(ult_file_path,header=3)
    ult_data.drop('Addenda',axis=1,inplace=True)
    
    # Convert columns names to only have letters, numbers, or the underscore
    ult_data.columns = list(map(lambda x: re.sub('[^\w]','_',x),ult_data.columns))
    
    # Filter dataframe based on kwargs:
    data_filter = np.array([True]*ult_data.shape[0])
    for key, value in kwargs.items():
        try:
            data_filter = data_filter & (ult_data[key] == value)
            imp_cols.append(key)
        except:
            print('Unrecognized keyword : ', key)
            continue
    
    if condensed:
        useful_cols = list(set(['Spec_No'] + imp_cols))
    
    filtered_ult_data = ult_data[data_filter]
    
    # Preprocess temperature columns in dataframe
    temp_cols = list(filter(lambda x: re.search('^d[0-9]+$',x),ult_data.columns))
    temp_cols_dict = {col: int(col.strip('d')) for col in temp_cols}
    temp_cols_dict.update({'d_20_to_100':100})
    
    temp_list = list(temp_cols_dict.values())
    temp_list.sort()
    
    filtered_ult_data = filtered_ult_data.rename(columns=temp_cols_dict)
    
    # Remove any temps in crit_temps for which data is available in the ASME data, temp_list
    crit_temps = list(filter(lambda x: x not in temp_list, crit_temps))
    
    # If there are no temps to extrapolated/interpolate, return the filtered data
    if not crit_temps:
        return filtered_ult_data
    
   
    # Generate interpolation functions
    y = filtered_ult_data[temp_list].dropna(axis=1)
    X = y.columns
    
    interp_fn = interp1d(X,y,fill_value='extrapolate')
    
    # Generate linearly interpolated/extrapolated data
    interp_data = pd.DataFrame(interp_fn(crit_temps),columns=crit_temps,index=y.index)
    
    # Concatenate new data to the old temp. columns and resort
    new_temp_data = pd.concat([y,interp_data],axis=1)
    new_temp_data.sort_index(axis=1,inplace=True)
    
    # Generate the final dataframe by putting in all the non-temp data:
    if condensed:
        final_df = pd.concat([filtered_ult_data[useful_cols],new_temp_data],axis=1)
    else:
        final_df = pd.concat([filtered_ult_data.drop(temp_list,axis=1),new_temp_data],axis=1)

    return final_df