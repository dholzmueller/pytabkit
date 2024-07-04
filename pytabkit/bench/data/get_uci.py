#!/usr/bin/python3
import os
import shutil
import ssl

import pandas

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.uci_file_ops import prepare_new_data_set_group_id, download_and_save, replace_chars_in_file, \
    load_raw_data, remove_columns, save_data_to_file, unzip_raw_data, concat_files, remove_files, UCIVars, \
    move_label_in_front, remove_rows_with_label, ungz_raw_data, load_mixed_raw_data, \
    auto_replace_categories_in_mixed_data, write_mixed_raw_data, replace_ordinals_in_mixed_data, \
    replace_isodate_by_day_in_mixed_data, replace_circulars_in_mixed_data, get_categories_in_mixed_data, \
    replace_time_by_seconds_in_mixed_data, unrar_raw_data, unarff_raw_data, un_z_raw_data, untar_raw_data, \
    replace_categories_in_mixed_data, replace_bin_cats_in_mixed_data, auto_replace_missing_in_mixed_data, \
    replace_manual_in_mixed_data
from pytabkit.models import utils
import numpy
import sklearn.datasets as datasets
import re as re



#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------



def get_skill_craft():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00272/SkillCraft1_Dataset.csv', 'skill_craft.data')

    replace_chars_in_file('skill_craft.data', '"', '')

    data = load_raw_data('skill_craft.data', sep = ',')
    data = remove_columns(data, [0])
    save_data_to_file(data, 'skill_craft', is_classification = True)
    


#---------------------------------------------------------------------------------------------------



def get_cargo_2000():
    
    prepare_new_data_set_group_id()
    print("Cargo 2000 data set is currently not processed since:")
    print("  - from the description it is completely unclear how this data set can be used")


#---------------------------------------------------------------------------------------------------



def get_KDC_4007():
    
    prepare_new_data_set_group_id()
    print("KDC 4007 data set is currently not processed since:")
    print("  - from the description it is completely unclear how this data set can be used")
    
    
#---------------------------------------------------------------------------------------------------



def get_sml2010():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00274/NEW-DATA.zip', 'sml2010.zip')

    unzip_raw_data('sml2010.zip')    
    concat_files(UCIVars.raw_data_folder + 'NEW-DATA*.txt', UCIVars.raw_data_folder + 'sml2010.data')
    remove_files(UCIVars.raw_data_folder, 'NEW-DATA*.txt')

    replace_chars_in_file('sml2010.data', '#', '')
    data = load_raw_data('sml2010.data', sep = ' ', description_columns = 2)
    
    data_dining = remove_columns(data, [1]) 
    save_data_to_file(data_dining, 'sml2010_dining', is_classification = False)

    data_room = remove_columns(data, [0]) 
    save_data_to_file(data_room, 'sml2010_room', is_classification = False)
    
    



#---------------------------------------------------------------------------------------------------



def get_wine_quality():


    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', 'wine_quality_red.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', 'wine_quality_white.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names', 'wine_quality.description')
    
    
    # The first task is to create data sets in which the quality is the label.
    # To this end, we add a column at the right, which indicates whether the wine is white or read.
    
    data_white = load_raw_data('wine_quality_white.data', sep = ';', header = True)
    data_white = move_label_in_front(data_white, 11)
    white_label = numpy.ones((numpy.shape(data_white)[0], 1))
    data_white = numpy.concatenate((data_white, white_label), axis = 1)
    
    save_data_to_file(data_white, 'wine_quality_white', is_classification = True)

    
    data_red = load_raw_data('wine_quality_red.data', sep = ';', header = True)
    data_red = move_label_in_front(data_red, 11)
    red_label = numpy.zeros((numpy.shape(data_red)[0], 1))
    data_red = numpy.concatenate((data_red, red_label), axis = 1)
    
    data_all = numpy.concatenate((data_red, data_white), axis = 0)
    save_data_to_file(data_all, 'wine_quality_all', is_classification = True)
    

    # The next task is to combine the white and red wine data set and 
    # to add a label describing the color of the wine. We further remove
    # the quality of the wine, since this may give too much information
    # about the color.

    data_white = load_raw_data('wine_quality_white.data', sep = ';', header = True)
    data_white = remove_columns(data_white, [11])
    white_label = numpy.ones((numpy.shape(data_white)[0], 1))
    data_white = numpy.concatenate((white_label, data_white), axis = 1)
    

    data_red = load_raw_data('wine_quality_red.data', sep = ';', header = True)
    data_red = remove_columns(data_red, [11])
    red_label = numpy.zeros((numpy.shape(data_red)[0], 1))
    data_red = numpy.concatenate((red_label, data_red), axis = 1)
    
    data_all = numpy.concatenate((data_red, data_white), axis = 0)
    save_data_to_file(data_all, 'wine_quality_type', is_classification = True, is_regression = False)      
    

  

#---------------------------------------------------------------------------------------------------


def get_parkinson():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data', 'parkinson_updrs.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.names', 'parkinson_updrs.description')

    data = load_raw_data('parkinson_updrs.data', sep = ',', description_columns = 1)


    # The data has two variables that can be predicted, namely updrs_motor and updrs_total. 
    # For both prediction tasks, the other target variable needs to be removed from the data

    data_motor = remove_columns(data, [4])
    data_motor = move_label_in_front(data_motor, 3)
    save_data_to_file(data_motor, 'parkinson_motor', is_classification = False)
    
    data_total = remove_columns(data, [3])
    data_total = move_label_in_front(data_total, 3)
    save_data_to_file(data_total, 'parkinson_total', is_classification = False)



#---------------------------------------------------------------------------------------------------


def get_insurance_benchmark():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt', 'insurance_benchmark.train.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt', 'insurance_benchmark.test.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt', 'insurance_benchmark.test.labels.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/TicDataDescr.txt', 'insurance_benchmark.description')

    train_data = load_raw_data('insurance_benchmark.train.data', sep = '\t')
    test_data = load_raw_data('insurance_benchmark.test.data', sep = '\t')
    test_label = load_raw_data('insurance_benchmark.test.labels.data', sep = '\t')
    
    test_data = numpy.concatenate((test_data, test_label), axis = 1)
    data = numpy.concatenate((train_data, test_data), axis = 0)

    data = move_label_in_front(data, 85)
    save_data_to_file(data, 'insurance_benchmark', is_classification = True)
    
    
    
#---------------------------------------------------------------------------------------------------



def get_EEG_steady_state():
    
    prepare_new_data_set_group_id()
    print("EEG Steady State Visual data set is currently not processed since:")
    print("  - the description indicates that it is time series data")

    
   
    
#---------------------------------------------------------------------------------------------------


def get_air_quality():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip', 'air_quality.zip')
    unzip_raw_data('air_quality.zip')
    
    
    os.rename(UCIVars.raw_data_folder + 'AirQualityUCI.csv', UCIVars.raw_data_folder + 'air_quality.data')
    os.remove(UCIVars.raw_data_folder + 'AirQualityUCI.xlsx')
    
    data = load_raw_data('air_quality.data', sep = ';', date_column = 0, date_sep = '/', date_order = 'dmY', time_column = 1, time_sep = '.', german_decimal = True)


    # The data has five variables that can be predicted, 
    # namely those in columns 2, 4, 5, 7, and 9 (C++ like).
    # For these prediction tasks, the other target variables 
    # need to be removed from the data.

    data_co2 = remove_columns(data, [4, 5, 7, 9])
    data_co2 = move_label_in_front(data_co2, 2)
    data_co2 = remove_rows_with_label(data_co2, -200.0)
    save_data_to_file(data_co2, 'air_quality_co2', is_classification = False)
    
    # The hydrocarbon reference measurements have only been taken 914 times
    # For this reason, they are not included in the constructed data sets.
    
    data_bc = remove_columns(data, [2, 4, 7, 9])
    data_bc = move_label_in_front(data_bc, 3)
    data_bc = remove_rows_with_label(data_bc, -200.0)
    save_data_to_file(data_bc, 'air_quality_bc', is_classification = False)
    
    data_nox = remove_columns(data, [2, 4, 5, 9])
    data_nox = move_label_in_front(data_nox, 4)
    data_nox = remove_rows_with_label(data_nox, -200.0)
    save_data_to_file(data_nox, 'air_quality_nox', is_classification = False)
    
    data_no2 = remove_columns(data, [2, 4, 5, 7])
    data_no2 = move_label_in_front(data_no2, 5)
    data_no2 = remove_rows_with_label(data_no2, -200.0)
    save_data_to_file(data_no2, 'air_quality_no2', is_classification = False)


#---------------------------------------------------------------------------------------------------


def get_cycle_power_plant():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip', 'cycle_power_plant.zip')
    unzip_raw_data('cycle_power_plant.zip')
    
    # The zip file contains some junk and in addition, the data is in EXCEL format. This is addressed now:

    excel_data = pandas.read_excel(UCIVars.raw_data_folder + 'CCPP/Folds5x2_pp.xlsx', engine = 'openpyxl')
    excel_data.to_csv(UCIVars.raw_data_folder + 'cycle_power_plant.data')
    shutil.rmtree(UCIVars.raw_data_folder + 'CCPP')
    
    
    # The response variable is in the last column
    
    data = load_raw_data('cycle_power_plant.data', sep = ',', description_columns = 1)
    data = move_label_in_front(data, 4)
    save_data_to_file(data, 'cycle_power_plant', is_classification = False)


#---------------------------------------------------------------------------------------------------


def get_carbon_nanotubes():
    prepare_new_data_set_group_id()
    
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00448/carbon_nanotubes.csv', 'carbon_nanotubes.data')
    
    data = load_raw_data('carbon_nanotubes.data', sep = ';', german_decimal = True)
    
    data_u = remove_columns(data, [6, 7])
    data_u = move_label_in_front(data_u, 5)
    save_data_to_file(data_u, 'carbon_nanotubes_u', is_classification = False)
    
    data_v = remove_columns(data, [5, 7])
    data_v = move_label_in_front(data_v, 5)
    save_data_to_file(data_v, 'carbon_nanotubes_v', is_classification = False)
    
    data_w = remove_columns(data, [5, 6])
    data_w = move_label_in_front(data_w, 5)
    save_data_to_file(data_w, 'carbon_nanotubes_w', is_classification = False)
    

#---------------------------------------------------------------------------------------------------



def get_naval_propulsion():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip', 'naval_propulsion.zip')
    unzip_raw_data('naval_propulsion.zip')
    
    # The zip file contains quite a bit of junk, which is removed in the following
    
    
    shutil.copy(UCIVars.raw_data_folder + 'UCI CBM Dataset/data.txt', UCIVars.raw_data_folder + 'naval_propulsion.data')
    shutil.copy(UCIVars.raw_data_folder + 'UCI CBM Dataset/Features.txt', UCIVars.raw_data_folder + 'naval_propulsion.features.txt')
    shutil.copy(UCIVars.raw_data_folder + 'UCI CBM Dataset/README.txt', UCIVars.raw_data_folder + 'naval_propulsion.description')    
    
    shutil.rmtree(UCIVars.raw_data_folder + 'UCI CBM Dataset/')
    shutil.rmtree(UCIVars.raw_data_folder + '__MACOSX')
    
    
    data = load_raw_data('naval_propulsion.data', sep = '   ')
    
    
    # The data has actually three response variables, but one of those, namely the ship speed
    # is affine linear in the lever position, which is also recorded in the data. For this
    # reason, only the other two response variables are considered.
    
    data_comp = remove_columns(data, [17])
    data_comp = move_label_in_front(data_comp, 16)
    save_data_to_file(data_comp, 'naval_propulsion_comp', is_classification = False)
    
    data_turb = remove_columns(data, [16])
    data_turb = move_label_in_front(data_turb, 16)
    save_data_to_file(data_turb, 'naval_propulsion_turb', is_classification = False)


#---------------------------------------------------------------------------------------------------
    

def get_blood_pressure():
    
    prepare_new_data_set_group_id()
    print("Cuff-Less Blood pressure Estimation is currently not processed since:")
    print("  - the zip file is about 3.1GB large")
    print("  - the description indicates that each of the three features is actually a times series")
    print("  - the file is in matlab format")
    
    #print('The following download may take a while, since the .zip file is about 3.1GB large.')
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00340/data.zip', 'blood_pressure.zip')
    #unzip_raw_data('blood_pressure.zip')


#---------------------------------------------------------------------------------------------------


def get_gas_sensor_drift():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00270/driftdataset.zip', 'gas_sensor_drift.zip')
    unzip_raw_data('gas_sensor_drift.zip')
    
    concat_files(UCIVars.raw_data_folder + 'batch*.dat', UCIVars.raw_data_folder + 'gas_sensor_drift.data')
    remove_files(UCIVars.raw_data_folder, 'batch*.dat')
    
    
    # Next we need to replace ; by , in .data file, since otherwise the routines for libsvm-like formats won't work.
    # Also, the first label is multiplied by 10000 since the routine for libsvm-like formats seem to sort the 
    # labels. By multiplying the label by 10000, we actually can guarantee that the first label is always the larger
    # one, so that the routine places it at the second position in the list of labels.
    # Then we read a libsvm like file with multiple labels and convert it from Compressed Sparse Row format to normal format
    
    replace_chars_in_file('gas_sensor_drift.data', ';', '0000,')
    data = datasets.load_svmlight_file(UCIVars.raw_data_folder + 'gas_sensor_drift.data', multilabel = True)

    x_data = data[0].toarray()
    all_labels = numpy.reshape(data[1], newshape = (-1, 2))
    
    ## The data has two response variables, one indicating which chemical is measured 
    ## and one reporting its concentration. We simply take both as being of interest ...
    
    class_labels = numpy.reshape(all_labels[ :, 1], newshape = (-1, 1)) / 10000.0
    data_class = numpy.concatenate((class_labels, x_data), axis = 1)
    save_data_to_file(data_class, 'gas_sensor_drift_class', is_classification = True)

    conc_labels = numpy.reshape(all_labels[ :, 0], newshape = (-1, 1))
    data_conc = numpy.concatenate((conc_labels, x_data), axis = 1)
    save_data_to_file(data_conc, 'gas_sensor_drift_conc', is_classification = False)
    


#---------------------------------------------------------------------------------------------------


def get_bike_sharing():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip', 'bike_sharing.zip')
    unzip_raw_data('bike_sharing.zip')
    
    os.remove(UCIVars.raw_data_folder + 'day.csv')
    os.rename(UCIVars.raw_data_folder + 'hour.csv', UCIVars.raw_data_folder + 'bike_sharing.data')
    os.rename(UCIVars.raw_data_folder + 'Readme.txt', UCIVars.raw_data_folder + 'bike_sharing.description')
    
    data = load_raw_data('bike_sharing.data', sep = ',', description_columns = 2)
    
    data_casual = remove_columns(data, [13, 14])
    data_casual = move_label_in_front(data_casual, 12)
    save_data_to_file(data_casual, 'bike_sharing_casual', is_classification = False)

    data_total = remove_columns(data, [12, 13])
    data_total = move_label_in_front(data_total, 12)
    save_data_to_file(data_total, 'bike_sharing_total', is_classification = False)
    
    
    
#---------------------------------------------------------------------------------------------------


def get_appliances_energy():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv', 'appliances_energy.data')

    # The data entries are saved as strings, that is as "...". In addition, date and time are not separated by commas.
    # The following lines cure this.

    replace_chars_in_file('appliances_energy.data', '"', '')
    replace_chars_in_file('appliances_energy.data', ',  ', ',')
    replace_chars_in_file('appliances_energy.data', ', ', ',')
    replace_chars_in_file('appliances_energy.data', ' ', ',')
    
    data = load_raw_data('appliances_energy.data', sep = ',', date_column = 0, date_sep = '-', date_order = 'Ymd', time_column = 1, time_sep = ':')
    
    data = move_label_in_front(data, 2)
    save_data_to_file(data, 'appliances_energy', is_classification = False)


    
    
    
#---------------------------------------------------------------------------------------------------


def get_indoor_loc():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip', 'indoor_loc.zip')
    unzip_raw_data('indoor_loc.zip')
    
 
    os.rename(UCIVars.raw_data_folder + 'UJIndoorLoc/trainingData.csv', UCIVars.raw_data_folder + 'indoor_loc.train.csv')
    os.rename(UCIVars.raw_data_folder + 'UJIndoorLoc/validationData.csv', UCIVars.raw_data_folder + 'indoor_loc.val.csv')
    shutil.rmtree(UCIVars.raw_data_folder + 'UJIndoorLoc')
    
    concat_files(UCIVars.raw_data_folder + 'indoor*.csv', UCIVars.raw_data_folder + 'indoor_loc.data')
    remove_files(UCIVars.raw_data_folder, 'indoor*.csv') 


# --- Regression part ------

    data = load_raw_data('indoor_loc.data', sep = ',')
    data = remove_columns(data, range(523, 529))
    
    data_long = remove_columns(data, [521, 522])
    data_long = move_label_in_front(data_long, 520)
    save_data_to_file(data_long, 'indoor_loc_long', is_classification = False)
    
    data_lat = remove_columns(data, [520, 522])
    data_lat = move_label_in_front(data_lat, 520)
    save_data_to_file(data_lat, 'indoor_loc_lat', is_classification = False)
    
    data_alt = remove_columns(data, [520, 521])
    data_alt = move_label_in_front(data_alt, 520)
    save_data_to_file(data_alt, 'indoor_loc_alt', is_classification = False)
    
    
# --- Classification part -----
    
    data = load_raw_data('indoor_loc.data', sep = ',')
    data = remove_columns(data, range(526, 529))

    data_relative = move_label_in_front(data, 525)
    data_relative = remove_columns(data_relative, range(521, 526))
    save_data_to_file(data_relative, 'indoor_loc_relative', is_classification = True, is_regression = False)

    data_building = move_label_in_front(data, 523)
    data_building = remove_columns(data_building, range(521, 526))
    save_data_to_file(data_building, 'indoor_loc_building', is_classification = True, is_regression = False)
   


    
   


#---------------------------------------------------------------------------------------------------



def get_online_news_popularity():

    prepare_new_data_set_group_id()
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip', 'online_news_popularity.zip')
    unzip_raw_data('online_news_popularity.zip')

    os.rename(UCIVars.raw_data_folder + 'OnlineNewsPopularity/OnlineNewsPopularity.csv', UCIVars.raw_data_folder + 'online_news_popularity.data')
    os.rename(UCIVars.raw_data_folder + 'OnlineNewsPopularity/OnlineNewsPopularity.names', UCIVars.raw_data_folder + 'online_news_popularity.description')
    shutil.rmtree(UCIVars.raw_data_folder + 'OnlineNewsPopularity')
    
    data = load_raw_data('online_news_popularity.data', sep = ', ', description_columns = 2)
    data = move_label_in_front(data, 58)
    save_data_to_file(data, 'online_news_popularity', is_classification = False)


#---------------------------------------------------------------------------------------------------


def get_facebook_comment_volume():

    prepare_new_data_set_group_id()
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip', 'facebook_comment_volume.zip')
    unzip_raw_data('facebook_comment_volume.zip')

    os.rename(UCIVars.raw_data_folder + 'Dataset/Training/Features_Variant_1.csv', UCIVars.raw_data_folder + 'facebook_comment_volume.data')
    
    shutil.rmtree(UCIVars.raw_data_folder + 'Dataset')
    shutil.rmtree(UCIVars.raw_data_folder + '__MACOSX')
    

    data = load_raw_data('facebook_comment_volume.data', sep = ',')
    data = move_label_in_front(data, 53)
    save_data_to_file(data, 'facebook_comment_volume', is_classification = False)



#---------------------------------------------------------------------------------------------------


def get_bejing_pm25():

    prepare_new_data_set_group_id()
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv', 'bejing_pm25.data')


    replace_chars_in_file('bejing_pm25.data', 'cv', '0,0')
    replace_chars_in_file('bejing_pm25.data', 'NW', '1,2')
    replace_chars_in_file('bejing_pm25.data', 'NE', '1,1')
    replace_chars_in_file('bejing_pm25.data', 'SE', '2,1')
    replace_chars_in_file('bejing_pm25.data', 'SW', '2,2')
    
    data = load_raw_data('bejing_pm25.data', sep = ',', description_columns = 1)
    data = move_label_in_front(data, 4)
    save_data_to_file(data, 'bejing_pm25', is_classification = False)
    


#---------------------------------------------------------------------------------------------------
    
    
def get_protein_tertiary_structure():    
    
    prepare_new_data_set_group_id()
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv', 'protein_tertiary_structure.data')
    
    data = load_raw_data('protein_tertiary_structure.data', sep = ',')
    save_data_to_file(data, 'protein_tertiary_structure', is_classification = False)



#---------------------------------------------------------------------------------------------------



def get_tamilnadu_electricity():
    
    prepare_new_data_set_group_id()
    print("Tamilnadu Electricity data set is currently not processed since:")
    print("  - from the description it is completely unclear how this data set can be used")


#---------------------------------------------------------------------------------------------------


def get_metro_interstate_traffic_volume():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz', 'metro_interstate_traffic_volume.zip')

    ungz_raw_data('metro_interstate_traffic_volume.zip')
    os.rename(UCIVars.raw_data_folder + 'metro_interstate_traffic_volume.zip.data', UCIVars.raw_data_folder + 'metro_interstate_traffic_volume.data')

    data = load_mixed_raw_data('metro_interstate_traffic_volume.data', sep = ',', header = True)
    size = data.shape[0]
    data[0:size, 7] = [date_and_time.replace(' ', ',') for date_and_time in data[0:size, 7]]
    
    
    # Deal with the holidays: we put all holidays in one category, and all non-holidays in the other. 
    # There are 11 holidays and 'None'. The latter receives the value 0, while all holidays receive the
    # value 1. The following code is based on string replacement and the particular form of the entries.
    
    data[0:size, 0] = [re.sub(r" ", '', holiday) for holiday in data[0:size, 0]]
    data[0:size, 0] = [re.sub(r"None", '0', holiday) for holiday in data[0:size, 0]]
    data[0:size, 0] = [re.sub(r"D", '1', holiday) for holiday in data[0:size, 0]]
    data[0:size, 0] = [re.sub(r"WashingtonsBirthday", '1', holiday) for holiday in data[0:size, 0]]
    data[0:size, 0] = [re.sub(r"StateFair", '1', holiday) for holiday in data[0:size, 0]]
    data[0:size, 0] = [re.sub(r"[a-zA-Z]", '', holiday) for holiday in data[0:size, 0]]


    # The weather is briefly described in column 5 and in more detail in column 6. 
    # We create two data sets, one for each type of description. 
    
    data_short = auto_replace_categories_in_mixed_data(data, 5, ',')
    data_short = remove_columns(data_short, 6)
    write_mixed_raw_data(UCIVars.raw_data_folder + 'metro_interstate_traffic_volume_short.data', data_short, sep = ",")
    
    data_long = auto_replace_categories_in_mixed_data(data, 6, ',')
    data_long = remove_columns(data_long, 5)
    write_mixed_raw_data(UCIVars.raw_data_folder + 'metro_interstate_traffic_volume_long.data', data_long, sep = ",")    

    write_mixed_raw_data(UCIVars.raw_data_folder + 'metro_interstate_traffic_volume.data', data, sep = ",")
    replace_chars_in_file('metro_interstate_traffic_volume.data', '  ', ' ')
    
    
    # Now we are in the position ot read the data, convert the time and date, and movel the labels
    
    
    data = load_raw_data('metro_interstate_traffic_volume_short.data', ',', description_columns = 0, date_column = 16, date_sep = '-', date_order = 'Ymd', time_column = 17, time_sep = ':')
    data = move_label_in_front(data, 18)
    save_data_to_file(data, 'metro_interstate_traffic_volume_short', is_classification = False, is_regression = True)
    


    data = load_raw_data('metro_interstate_traffic_volume_long.data', ',', description_columns = 0, date_column = 43, date_sep = '-', date_order = 'Ymd', time_column = 44, time_sep = ':')
    data = move_label_in_front(data, 45)
    save_data_to_file(data, 'metro_interstate_traffic_volume_long', is_classification = False, is_regression = True)



#---------------------------------------------------------------------------------------------------


def get_facebook_live_sellers_thailand():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00488/Live_20210128.csv', 'facebook_live_sellers_thailand.data')

    data = load_mixed_raw_data('facebook_live_sellers_thailand.data', sep = ",", header = True)
        
    # Columns 0 and 2 contain id and time information. These are deleted. The last 4 columns are empty,
    # and thus deleted, too.
        
    data = remove_columns(data, [0, 2, 12, 13, 14, 15])
    
    # Next we replace the status_type by some numbers
    
    categories = [u'link', u'photo', u'status', u'video']
    data = replace_ordinals_in_mixed_data(data, categories, 0, separator = ',')  
    write_mixed_raw_data(UCIVars.raw_data_folder + 'facebook_live_sellers_thailand.data', data, sep = ",")
    

    data = load_raw_data('facebook_live_sellers_thailand.data', ',')
    
    # The classes 1 and 3 contain 63 and 365 samples, only. We remove them for the classification data set
    
    data_class = remove_rows_with_label(data, 1)
    data_class = remove_rows_with_label(data_class, 3)
    save_data_to_file(data_class, 'facebook_live_sellers_thailand_status', is_classification = True, is_regression = False)
    
    
    # For the regression data set, we pick the 'shares' column as label 
    
    data_regr = move_label_in_front(data, 3)
    save_data_to_file(data_regr, 'facebook_live_sellers_thailand_shares', is_classification = False, is_regression = True)
    


#---------------------------------------------------------------------------------------------------


def get_parking_birmingham():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00482/dataset.zip', 'parking_birmingham.zip')
    unzip_raw_data('parking_birmingham.zip')
    os.rename(UCIVars.raw_data_folder + 'dataset.csv', UCIVars.raw_data_folder + 'parking_birmingham.data')
    
    
    # One could also convert the name of the parking spot into a binary vector. However, this vector is 
    # of dimension 30 and therefore it would dominate the remaining features. We thus use a one dimensional
    # representation, instead.

    data = load_mixed_raw_data('parking_birmingham.data', sep = ',', header = True)
    categories = ['BHMEURBRD01', 'BHMEURBRD02', 'Bull Ring', 'BHMBRCBRG02', 'BHMBRCBRG03', 'BHMBRCBRG01', 'Shopping', 'BHMNCPLDH01', 'BHMBCCSNH01', 'BHMNCPRAN01', 'BHMBCCPST01', 'Others-CCCPS133', 'BHMBRTARC01', 'Others-CCCPS98', 'NIA North', 'BHMNCPHST01', 'BHMNCPNST01', 'BHMNCPNHS01', 'BHMBCCTHL01', 'Others-CCCPS119a', 'Others-CCCPS8', 'Others-CCCPS105a', 'Broad Street', 'NIA South', 'NIA Car Parks', 'BHMBCCMKT01', 'BHMMBMMBX01', 'Others-CCCPS202', 'Others-CCCPS135a', 'BHMNCPPLS01']

    data = replace_ordinals_in_mixed_data(data, categories, 0, separator = ',')  
    write_mixed_raw_data(UCIVars.raw_data_folder + 'parking_birmingham.data', data, sep = ",")
    
    
    # Next we split date-time into two features
    
    replace_chars_in_file('parking_birmingham.data', '  ', ',')
    

    # Now, we convert the date into a weekday and then into a point on the circle
    # Furthermore, we create a second data set with rounded times fur possible future time series 
    # treatment.

    data = load_mixed_raw_data('parking_birmingham.data', sep = ",", header = False)

    data = replace_isodate_by_day_in_mixed_data(data, 3)
    data = replace_circulars_in_mixed_data(data, get_categories_in_mixed_data(data, 3), 3, ",")
    
    write_mixed_raw_data(UCIVars.raw_data_folder + 'parking_birmingham.data', data, sep = ",")

    data = replace_time_by_seconds_in_mixed_data(data, 4, sep = ':', rounded = 1800)
    write_mixed_raw_data(UCIVars.raw_data_folder + 'parking_birmingham.rounded.data', data, sep = ",")
    
    
    # Now we compute the relative occupancy and use it as label
    # Note that we keep both the parking spot number and its capacity
    
    data = load_raw_data('parking_birmingham.data', ',', time_column = 5, time_sep = ':')
    data[:, 2] = data[:, 2] / data[:, 1]
    data = move_label_in_front(data, 2)
    
    save_data_to_file(data, 'parking_birmingham', is_classification = False, is_regression = True)
    
    
    
    
#---------------------------------------------------------------------------------------------------


def get_tarvel_review_ratings():
    
    prepare_new_data_set_group_id()
    
    
    # Download the data and correct the mispelling of its name
    
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00485/google_review_ratings.csv', 'travel_review_ratings.data')
    
    
    # Remove the commas at the end of each row and clean a few messy lines
    
    replace_chars_in_file('travel_review_ratings.data', ',\r', '\r')
    replace_chars_in_file('travel_review_ratings.data', '"', '')
    replace_chars_in_file('travel_review_ratings.data', ',,', ',')
    replace_chars_in_file('travel_review_ratings.data', '\t', '')
    
    data = load_raw_data('travel_review_ratings.data', ',', description_columns = 1, header = True)
    
    
    # Determine the first column that contains the most ratings, use it as label, and remove possible rows 
    # with label = 0
    
    ratings_counts = data.astype(bool).sum(axis=0)
    most_rated_column = numpy.argmax(ratings_counts)
    data = move_label_in_front(data, most_rated_column)
    remove_rows_with_label(data, 0.0)
    
    save_data_to_file(data, 'travel_review_ratings', is_classification = False, is_regression = True)
    


    
#---------------------------------------------------------------------------------------------------


def get_superconductivity():

    prepare_new_data_set_group_id()
    
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip', 'superconductivity.zip')
    unzip_raw_data('superconductivity.zip')
    
    os.rename(UCIVars.raw_data_folder + 'train.csv', UCIVars.raw_data_folder + 'superconductivity.data')
    os.remove(UCIVars.raw_data_folder + 'unique_m.csv')
    
    
    data = load_raw_data('superconductivity.data', ',', header = True)
    data_regr = move_label_in_front(data, 81)

    save_data_to_file(data_regr, 'superconductivity', is_classification = False, is_regression = True)


    # We also create a classification daat set, in which we try to identify materials with critical temperature above 77K.
    # We refer to https://en.wikipedia.org/wiki/Superconductivity   for the importance of this threhsod in view of liquid nitrogen.

    data_class = move_label_in_front(data, 81)

    temperature_above_77K = data_class[:,0] > 77
    data_class[:,0] = temperature_above_77K.astype(float)

    save_data_to_file(data_class, 'superconductivity_class', is_classification = True, is_regression = False)




#---------------------------------------------------------------------------------------------------


def get_gnfuv_unmanned_surface_vehicles():
    
    prepare_new_data_set_group_id()


    print("GNFUV Unmanned Surface Vehicles is currently not processed since:")
    print("  - the description indicates that it is actually very complicated times series data")


#---------------------------------------------------------------------------------------------------


def get_five_cities_pm25():
    
    prepare_new_data_set_group_id()
    
    print("PM2.5 of Five Chinese Cities is used since:")
    print("  - it actually contains 5 data sets of around 20.000 samples, each.")
    
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00394/FiveCitiePMData.rar', 'five_cities_pm25.rar')
    unrar_raw_data('five_cities_pm25.rar')
    
    cities = {}
    pm_locs = {}

    cities[0] = 'ShenyangPM20100101_20151231.csv'
    cities[1] = 'ChengduPM20100101_20151231.csv'
    cities[2] = 'BeijingPM20100101_20151231.csv'
    cities[3] = 'GuangzhouPM20100101_20151231.csv'
    cities[4] = 'ShanghaiPM20100101_20151231.csv'
    
    pm_locs[0] = (5,6,7)
    pm_locs[1] = (5,6,7)
    pm_locs[2] = (5,6,7,8)
    pm_locs[3] = (5,6,7)
    pm_locs[4] = (5,6,7)
    

    for i in range(0, 5):
        new_city_name = 'five_cities_' + cities[i][:-23].lower() + '_pm25.data'
        os.rename(UCIVars.raw_data_folder + cities[i], UCIVars.raw_data_folder + new_city_name)
        cities[i] = new_city_name
        
        replace_chars_in_file(cities[i], 'cv', '0,0')
        replace_chars_in_file(cities[i], 'NW', '1,2')
        replace_chars_in_file(cities[i], 'NE', '1,1')
        replace_chars_in_file(cities[i], 'SE', '2,1')
        replace_chars_in_file(cities[i], 'SW', '2,2')
        
        data = load_raw_data(cities[i], sep = ',', description_columns = 1)
        
        number_of_rows = numpy.shape(data)[0]
        pm_concs = data[0:number_of_rows, pm_locs[i]]
        pm_concs = numpy.mean(pm_concs, axis = 1)
        pm_concs = numpy.reshape(pm_concs, newshape = (number_of_rows, 1))
    
        data = remove_columns(data, pm_locs[i])
        data = numpy.concatenate((pm_concs, data), axis = 1)
        save_data_to_file(data, new_city_name[:-5], is_classification = False)

    
    
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------





def get_phishing():

    prepare_new_data_set_group_id()
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff', 'phishing.arff') 
    replace_chars_in_file('phishing.arff', ' -1', '-1')
    replace_chars_in_file('phishing.arff', ' 1', '1')
    replace_chars_in_file('phishing.arff', '1 ', '1')
    replace_chars_in_file('phishing.arff', '-1 ', '-1')
    replace_chars_in_file('phishing.arff', '0 ', '0')
    replace_chars_in_file('phishing.arff', ' 0', '0')
    unarff_raw_data('phishing')

    data = load_raw_data('phishing.data', sep = ',', description_columns = 0)
    data = move_label_in_front(data, 30)
    save_data_to_file(data, 'phishing', is_classification = True, is_regression = False)




#---------------------------------------------------------------------------------------------------


def get_ozone_level():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/eighthr.data', 'ozone_level_8hr.data')  
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/eighthr.names', 'ozone_level_8hr.description')
    
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.data', 'ozone_level_1hr.data') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/onehr.names', 'ozone_level_1hr.description')
  

    data = load_raw_data('ozone_level_8hr.data', sep = ',', description_columns = 1, na_string = '?')
    data = move_label_in_front(data, 72)
    save_data_to_file(data, 'ozone_level_8hr', is_classification = True, is_regression = False)
    
    data = load_raw_data('ozone_level_1hr.data', sep = ',', description_columns = 1, na_string = '?')
    data = move_label_in_front(data, 72)
    save_data_to_file(data, 'ozone_level_1hr', is_classification = True, is_regression = False)


#---------------------------------------------------------------------------------------------------

  
def get_opportunity_activity():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip', 'opportunity_activity.zip')
    #unzip_raw_data('opportunity_activity.zip')
    
    
    print("Opportunity Activity Recognition is currently not processed since:")
    print("  - the zip file is about 292MB large")
    print("  - the description indicates that it is actually times series data")


#---------------------------------------------------------------------------------------------------

  
def get_australian_sign_language():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/auslan2-mld/tctodd.tar.gz', 'australian_sign_language.tar.gz')
    
    
    print("Australian Sign Language is currently not processed since:")
    print("  - each sign only has 27 samples")


#---------------------------------------------------------------------------------------------------

  
def get_seismic_bumps():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff', 'seismic_bumps.arff')
    
    unarff_raw_data('seismic_bumps')
    replace_chars_in_file('seismic_bumps.data', 'a', '1')
    replace_chars_in_file('seismic_bumps.data', 'b', '2')
    replace_chars_in_file('seismic_bumps.data', 'c', '3')
    replace_chars_in_file('seismic_bumps.data', 'd', '4')
    
    replace_chars_in_file('seismic_bumps.data', 'N', '0')
    replace_chars_in_file('seismic_bumps.data', 'W', '1')
    
    data = load_raw_data('seismic_bumps.data', sep = ',')
    data = move_label_in_front(data, 18)
    save_data_to_file(data, 'seismic_bumps', is_classification = True, is_regression = False)



#---------------------------------------------------------------------------------------------------

  
def get_meu_mobile_ksd():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00399/MEU-Mobile%20KSD%202016.xlsx', 'meu_mobile_ksd.xlsx')


    print("MEU-Mobile KSD is currently not processed since:")
    print("  - according to the description it seems to be a anomaly detection data set")


#---------------------------------------------------------------------------------------------------

  
def get_character_trajectories():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/character-trajectories/mixoutALL_shifted.mat', 'character_trajectories.mat')


    print("Character Trajectories is currently not processed since:")
    print("  - according to the description it seems to be a time series data set")
    print("  - the file is in matlab format")



#---------------------------------------------------------------------------------------------------

  
def get_vicon_physical_action():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00214/Vicon%20Physical%20Action%20Data%20Set.rar', 'vicon_physical_action.rar')
    
    print("Vicon Physical Action is currently not processed since:")
    print("  - according to the description and an follow-up inspection it seems to be a time series data set")


#---------------------------------------------------------------------------------------------------

  
def get_simulated_falls():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00455/Tests.rar', 'simulated_falls.rar')
    print("Simulated Falls and Daily Living Activities is currently not processed since:")
    print("  - according to the description it seems to be a time series data set")
    print("  - the data set size is 1.2GB")



#---------------------------------------------------------------------------------------------------

  
def get_chess():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data', 'chess.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.names', 'chess.description')

    
    replace_chars_in_file('chess.data', 'nowin', '-1')
    replace_chars_in_file('chess.data', 'won', '1')
    replace_chars_in_file('chess.data', 'b', '0')
    replace_chars_in_file('chess.data', 'f', '1')
    replace_chars_in_file('chess.data', 'g', '2')
    replace_chars_in_file('chess.data', 'l', '3')
    replace_chars_in_file('chess.data', 'n', '4')
    replace_chars_in_file('chess.data', 't', '5')
    replace_chars_in_file('chess.data', 'w', '6')
    
    
    data = load_raw_data('chess.data', sep = ',')
    data = move_label_in_front(data, 36)
    save_data_to_file(data, 'chess', is_classification = True, is_regression = False)
    
    
    
#---------------------------------------------------------------------------------------------------

  
def get_abalone():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', 'abalone.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names', 'abalone.description')
    
    replace_chars_in_file('abalone.data', 'F', '-1')
    replace_chars_in_file('abalone.data', 'I', '0')
    replace_chars_in_file('abalone.data', 'M', '1')
    
    data = load_raw_data('abalone.data', sep = ',')
    save_data_to_file(data, 'abalone', is_classification = True, is_regression = False)
    
    
    
#---------------------------------------------------------------------------------------------------

  
def get_madelon():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data', 'madelon.train.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels', 'madelon.train.labels.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data', 'madelon.valid.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels', 'madelon.valid.labels.data')
    
    
    # I could not find the test labels, so the test data set is not included. LIBSVM's data set does not contain the test part, either
    
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data', 'madelon.test.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/Dataset.pdf', 'madelon.description.pdf')
    
    
    
    train_data = load_raw_data('madelon.train.data', sep = ' ')
    train_label = load_raw_data('madelon.train.labels.data', sep = ' ')
    train_data = numpy.concatenate((train_label, train_data), axis = 1)
    
    valid_data = load_raw_data('madelon.valid.data', sep = ' ')
    valid_label = load_raw_data('madelon.valid.labels.data', sep = ' ')
    valid_data = numpy.concatenate((valid_label, valid_data), axis = 1)
    
    data = numpy.concatenate((train_data, valid_data), axis = 0)
    
    save_data_to_file(data, 'madelon', is_classification = True, is_regression = False)
    
    
    
    
#---------------------------------------------------------------------------------------------------

  
def get_spambase():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.zip', 'spambase.zip')
    unzip_raw_data('spambase.zip')
    
    os.rename(UCIVars.raw_data_folder + 'spambase.names', UCIVars.raw_data_folder + 'spambase.feature.txt')
    os.rename(UCIVars.raw_data_folder + 'spambase.DOCUMENTATION', UCIVars.raw_data_folder + 'spambase.description')
    
    data = load_raw_data('spambase.data', sep = ',')
    data = move_label_in_front(data, 57)
    save_data_to_file(data, 'spambase', is_classification = True, is_regression = False)
    
    
#---------------------------------------------------------------------------------------------------

  
def get_wilt():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00285/wilt.zip', 'wilt.zip') 
    unzip_raw_data('wilt.zip')
    
    os.rename(UCIVars.raw_data_folder + 'training.csv', UCIVars.raw_data_folder + 'wilt.train.data')
    os.rename(UCIVars.raw_data_folder + 'testing.csv', UCIVars.raw_data_folder + 'wilt.test.data')
    
    concat_files(UCIVars.raw_data_folder + 'wilt.t*.data', UCIVars.raw_data_folder + 'wilt.data')
    
    replace_chars_in_file('wilt.data', 'n', '-1')
    replace_chars_in_file('wilt.data', 'w', '1')
    
        
    data = load_raw_data('wilt.data', sep = ',')
    save_data_to_file(data, 'wilt', is_classification = True, is_regression = False) 
    
    

    
    
    
#---------------------------------------------------------------------------------------------------



def get_waveform():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/waveform/waveform.data.Z', 'waveform.Z')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/waveform/waveform.names', 'waveform.description')

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/waveform/waveform-+noise.data.Z', 'waveform_noise.Z')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/waveform/waveform-+noise.names', 'waveform_noise.description')
    
    success = un_z_raw_data('waveform.Z')
    if success == True:
        os.rename(UCIVars.raw_data_folder + 'waveform', UCIVars.raw_data_folder + 'waveform.data')
        data = load_raw_data('waveform.data', sep = ',')
        data = move_label_in_front(data, 21)
        save_data_to_file(data, 'waveform', is_classification = True, is_regression = False)
    else:
        print("The waveform data set could not be built.")
    
    
    success = un_z_raw_data('waveform_noise.Z')
    if success == True:
        os.rename(UCIVars.raw_data_folder + 'waveform_noise', UCIVars.raw_data_folder + 'waveform_noise.data')
        data = load_raw_data('waveform_noise.data', sep = ',')
        data = move_label_in_front(data, 40)
        save_data_to_file(data, 'waveform_noise', is_classification = True, is_regression = False) 
    else:
        print("The waveform_noise data set could not be built.")    
    
    
#---------------------------------------------------------------------------------------------------

  
def get_wall_following_robot():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00194/AllData.zip', 'wall_follow_robot.zip')
    unzip_raw_data('wall_follow_robot.zip')
    
    os.rename(UCIVars.raw_data_folder + 'Wall-following.names', UCIVars.raw_data_folder + 'wall_follow_robot.description')
    
    os.rename(UCIVars.raw_data_folder + 'sensor_readings_2.data', UCIVars.raw_data_folder + 'wall_follow_robot_2.data')
    os.rename(UCIVars.raw_data_folder + 'sensor_readings_4.data', UCIVars.raw_data_folder + 'wall_follow_robot_4.data')
    os.rename(UCIVars.raw_data_folder + 'sensor_readings_24.data', UCIVars.raw_data_folder + 'wall_follow_robot_24.data')
    
    categories = ['Slight-Left-Turn', 'Move-Forward', 'Slight-Right-Turn', 'Sharp-Right-Turn']
    
    data = load_mixed_raw_data('wall_follow_robot_2.data', sep = ',', header = False)
    data = replace_ordinals_in_mixed_data(data, categories, 2, ',', unknown_string = '')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'wall_follow_robot_2.trafo.data', data, sep = ',')
        
    data = load_mixed_raw_data('wall_follow_robot_4.data', sep = ',', header = False)
    data = replace_ordinals_in_mixed_data(data, categories, 4, ',', unknown_string = '')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'wall_follow_robot_4.trafo.data', data, sep = ',')
    
    data = load_mixed_raw_data('wall_follow_robot_24.data', sep = ',', header = False)
    data = replace_ordinals_in_mixed_data(data, categories, 24, ',', unknown_string = '')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'wall_follow_robot_24.trafo.data', data, sep = ',')
    
    data = load_raw_data('wall_follow_robot_2.trafo.data', sep = ',')
    data = move_label_in_front(data, 2)
    save_data_to_file(data, 'wall_follow_robot_2', is_classification = True, is_regression = True)
    
    data = load_raw_data('wall_follow_robot_4.trafo.data', sep = ',')
    data = move_label_in_front(data, 4)
    save_data_to_file(data, 'wall_follow_robot_4', is_classification = True, is_regression = True)
    
    data = load_raw_data('wall_follow_robot_24.trafo.data', sep = ',')
    data = move_label_in_front(data, 24)
    save_data_to_file(data, 'wall_follow_robot_24', is_classification = True, is_regression = True)
    
  
  
#---------------------------------------------------------------------------------------------------

  
def get_page_blocks():
   
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/page-blocks/page-blocks.data.Z', 'page_blocks.Z')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/page-blocks/page-blocks.names', 'page_blocks.description')
    
    success = un_z_raw_data('page_blocks.Z')
    
    if success == True:
        os.rename(UCIVars.raw_data_folder + 'page_blocks', UCIVars.raw_data_folder + 'page_blocks.data')
        
        replace_chars_in_file('page_blocks.data', '      ', ' ')
        replace_chars_in_file('page_blocks.data', '     ', ' ')
        replace_chars_in_file('page_blocks.data', '    ', ' ')
        replace_chars_in_file('page_blocks.data', '   ', ' ')
        replace_chars_in_file('page_blocks.data', '  ', ' ')
        replace_chars_in_file('page_blocks.data', ' ', ',')
        
        data = load_raw_data('page_blocks.data', sep = ',', description_columns = 1)
        data = move_label_in_front(data, 10)
        save_data_to_file(data, 'page_blocks', is_classification = True, is_regression = False)   
    else:
        print("The waveform data set could not be built.")



#---------------------------------------------------------------------------------------------------

  
def get_optical_recognition_handwritten_digits():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', 'optical_recognition_handwritten_digits.test.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', 'optical_recognition_handwritten_digits.train.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.names', 'optical_recognition_handwritten_digits.description')

    # The additional 'original' data sets contain the bitmaps of the handwritten digits in a strange format.
    # For this reason, they are not further considered.

    concat_files(UCIVars.raw_data_folder + 'optical_recognition_handwritten_digits.*.data', UCIVars.raw_data_folder + 'optical_recognition_handwritten_digits.data')

    data = load_raw_data('optical_recognition_handwritten_digits.data', sep = ',')
    data = move_label_in_front(data, 64)
    save_data_to_file(data, 'optical_recognition_handwritten_digits', is_classification = True, is_regression = False)   
    
    

#---------------------------------------------------------------------------------------------------

  
def get_bach_chorals_harmony():
    
    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00298/jsbach_chorals_harmony.zip', 'bach_chorals_harmony.zip')

    print("Bach Chorals Harmony is currently not processed since:")
    print("  - it contains a lot of classes with a handful of samples, only")



#---------------------------------------------------------------------------------------------------

  
def get_turkiye_student_evaluation():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00262/turkiye-student-evaluation_generic.csv', 'turkiye_student_evaluation.data')

    # Without an explicit target variable, we decided to use the instructor id as target variable

    data = load_raw_data('turkiye_student_evaluation.data', sep = ',')
    save_data_to_file(data, 'turkiye_student_evaluation', is_classification = True, is_regression = False) 





#---------------------------------------------------------------------------------------------------

  
def get_smartphone_human_activity():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00364/dataset_uci.zip', 'smartphone_human_activity.zip')
    unzip_raw_data('smartphone_human_activity.zip')

    os.rename(UCIVars.raw_data_folder + 'dataset_uci/final_X_train.txt', UCIVars.raw_data_folder + 'smartphone_human_activity.train.data')
    os.rename(UCIVars.raw_data_folder + 'dataset_uci/final_X_test.txt', UCIVars.raw_data_folder + 'smartphone_human_activity.test.data')

    os.rename(UCIVars.raw_data_folder + 'dataset_uci/final_y_train.txt', UCIVars.raw_data_folder + 'smartphone_human_activity.train.labels.data')
    os.rename(UCIVars.raw_data_folder + 'dataset_uci/final_y_test.txt', UCIVars.raw_data_folder + 'smartphone_human_activity.test.labels.data')

    os.rename(UCIVars.raw_data_folder + 'dataset_uci/features_info.txt', UCIVars.raw_data_folder + 'smartphone_human_activity.features.txt')
    os.rename(UCIVars.raw_data_folder + 'dataset_uci/README.txt', UCIVars.raw_data_folder + 'smartphone_human_activity.description')

    shutil.rmtree(UCIVars.raw_data_folder + 'dataset_uci')
    
    
    train_data = load_raw_data('smartphone_human_activity.train.data', sep = ',')
    train_label = load_raw_data('smartphone_human_activity.train.labels.data', sep = ',')
    train_data = numpy.concatenate((train_label, train_data), axis = 1)
    
    test_data = load_raw_data('smartphone_human_activity.test.data', sep = ',')
    test_label = load_raw_data('smartphone_human_activity.test.labels.data', sep = ',')
    test_data = numpy.concatenate((test_label, test_data), axis = 1)
    
    data = numpy.concatenate((train_data, test_data), axis = 0)
    save_data_to_file(data, 'smartphone_human_activity', is_classification = True, is_regression = False) 


#---------------------------------------------------------------------------------------------------

  
def get_artificial_characters():
    
    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/artificial-characters/character.tar.Z', 'artficial_characters.tar.Z')
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/artificial-characters/character.names', 'artficial_characters.description')

    print("Artificial Characters is currently not processed since:")
    print("  - the data comes in a rather convoluted form")


#---------------------------------------------------------------------------------------------------

  
def get_first_order_theorem_proving():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00249/ml-prove.tar.gz', 'first_order_theorem_proving.tar.gz')
    untar_raw_data('first_order_theorem_proving.tar.gz')
    
    os.rename(UCIVars.raw_data_folder + 'ml-prove/all-data-raw.csv', UCIVars.raw_data_folder + 'first_order_theorem_proving.data')
    os.rename(UCIVars.raw_data_folder + 'ml-prove/bridge-holden-paulson-details.txt', UCIVars.raw_data_folder + 'first_order_theorem_proving.description')
    shutil.rmtree(UCIVars.raw_data_folder + 'ml-prove')
    
    
    data = load_raw_data('first_order_theorem_proving.data', sep = ',')
    
    rows = numpy.shape(data)[0]
    columns = numpy.shape(data)[1]

    times_of_heuristics = data[0:rows, columns - 5:columns]
    data_features = data[0:rows, 0:columns - 5]

    # Create class labels, where -1 means the "decline" option, that occurs, if none of the 
    # five considered heuristics finished within 100 secs. Also, there are 13 samples, in 
    # which the heuristics appear to have finished instantaneously. These get a positive label.
    # One could also create regression tasks for each of the heuristics, but for now, we
    # don't do this.
    
    class_labels = numpy.reshape(numpy.sign(numpy.amax(times_of_heuristics, axis = 1)), newshape = (rows, 1))
    class_labels[numpy.where(class_labels[0:rows, 0] == 0)] = 1.0
    class_data = numpy.concatenate((class_labels, data_features), axis = 1)

    save_data_to_file(class_data, 'first_order_theorem_proving', is_classification = True, is_regression = False) 



#---------------------------------------------------------------------------------------------------

  
def get_landsat_satimage():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn', 'landsat_satimage.train.data') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst', 'landsat_satimage.test.data') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.doc', 'landsat_satimage.description')
    
    concat_files(UCIVars.raw_data_folder + 'landsat_satimage.*.data', UCIVars.raw_data_folder + 'landsat_satimage.data')


    data = load_raw_data('landsat_satimage.data', sep = ' ')
    data = move_label_in_front(data, 36)
    save_data_to_file(data, 'landsat_satimage', is_classification = True, is_regression = False)   



#---------------------------------------------------------------------------------------------------

  
def get_hiv_1_protease():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00330/newHIV-1_data.zip', 'hiv_1_protease.zip') 
    
    print("HIV-1 protease is currently not processed since:")
    print("  - the 1D data comes in a rather convoluted form")

    

#---------------------------------------------------------------------------------------------------

  
def get_musk():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/musk/clean2.data.Z', 'musk.Z')     
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/musk/clean2.info', 'musk.description')     
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/musk/clean2.names', 'musk.features.txt')     
    
    success = un_z_raw_data('musk.Z')
    
    if success == True:
        os.rename(UCIVars.raw_data_folder + 'musk', UCIVars.raw_data_folder + 'musk.data')
        
        data = load_raw_data('musk.data', description_columns = 2, sep = ',')
        data = move_label_in_front(data, 166)
        save_data_to_file(data, 'musk', is_classification = True, is_regression = False)   
    else:
        print("The musk data set could not be built.")


#---------------------------------------------------------------------------------------------------

  
def get_ble_rssi_indoor_location():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00435/BLE_RSSI_dataset.zip', 'ble_rssi_indoor_location.zip') 
    
    print("BLE RSSI indoor location is currently not processed since:")
    print("  - it only has 1420 labeled samples")


#---------------------------------------------------------------------------------------------------

  
def get_australian_sign_language():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/auslan-mld/allsigns.tar.gz', 'australian_sign_language.zip') 
    
    print("Australian sign language is currently not processed since:")
    print("  - the 1D data comes in a rather convoluted form")
    print("  - it truly seems to be time series data")


#---------------------------------------------------------------------------------------------------

  
def get_anuran_calls():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip', 'anuran_calls.zip') 
    unzip_raw_data('anuran_calls.zip')
    
    os.rename(UCIVars.raw_data_folder + 'Frogs_MFCCs.csv', UCIVars.raw_data_folder + 'anuran_calls.data')
    os.rename(UCIVars.raw_data_folder + 'Readme.txt', UCIVars.raw_data_folder + 'anuran_calls.description')
    
    
    data = load_mixed_raw_data('anuran_calls.data', sep = ',', header = True)

    categories = sorted(get_categories_in_mixed_data(data, 22))
    data = replace_ordinals_in_mixed_data(data, categories, 22, separator = ',', unknown_replacement_value = 0, begin_value = 1)
    
    categories = get_categories_in_mixed_data(data, 23)
    data = replace_ordinals_in_mixed_data(data, categories, 23, separator = ',', unknown_replacement_value = 0, begin_value = 1)
    
    categories = get_categories_in_mixed_data(data, 24)
    data = replace_ordinals_in_mixed_data(data, categories, 24, separator = ',', unknown_replacement_value = 0, begin_value = 1)
    
    write_mixed_raw_data(UCIVars.raw_data_folder + 'anuran_calls.data', data, sep = ',')
    
    
    data = load_raw_data('anuran_calls.data', sep = ',')
    data = remove_columns(data, [25])
    
    
    # There are three different classification problems, each have a few classes
    # with less than 250 samples. The following lines build these three problems
    # and remove the small classes.
    
    data_species = remove_columns(data, [22, 23])
    data_species = move_label_in_front(data_species, 22)     
    rows = numpy.shape(data_species)[0]
    data_species = data_species[numpy.where(data_species[0:rows, 0] != 3)[0], 0:24]
    rows = numpy.shape(data_species)[0]
    data_species = data_species[numpy.where(data_species[0:rows, 0] != 6)[0], 0:24]
    rows = numpy.shape(data_species)[0]
    data_species = data_species[numpy.where(data_species[0:rows, 0] != 10)[0], 0:24]
    save_data_to_file(data_species, 'anuran_calls_species', is_classification = True, is_regression = False)   
    
    data_genus = remove_columns(data, [22, 24])
    data_genus = move_label_in_front(data_genus, 22)
    rows = numpy.shape(data_genus)[0]
    data_genus = data_genus[numpy.where(data_genus[0:rows, 0] != 1)[0], 0:24]
    rows = numpy.shape(data_genus)[0]
    data_genus = data_genus[numpy.where(data_genus[0:rows, 0] != 4)[0], 0:24]
    rows = numpy.shape(data_genus)[0]
    data_genus = data_genus[numpy.where(data_genus[0:rows, 0] != 5)[0], 0:24]
    save_data_to_file(data_genus, 'anuran_calls_genus', is_classification = True, is_regression = False)  
    
    data_families = remove_columns(data, [23, 24])
    data_families = move_label_in_front(data_families, 22)
    rows = numpy.shape(data_families)[0]
    data_families = data_families[numpy.where(data_families[0:rows, 0] != 1)[0], 0:24]
    save_data_to_file(data_families, 'anuran_calls_families', is_classification = True, is_regression = False)  



#---------------------------------------------------------------------------------------------------

  
def get_thyroids():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick-euthyroid.data', 'thyroid_sick_eu.data') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick-euthyroid.names', 'thyroid_sick_eu.description') 

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.data', 'thyroid_sick.train.data') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.test', 'thyroid_sick.test.data') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.names', 'thyroid_sick.description') 

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/dis.data', 'thyroid_dis.train.data') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/dis.test', 'thyroid_dis.test.data') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/dis.names', 'thyroid_dis.description')

    # new-thyroid.data only contains 215 samples and is thus ommitted

    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data', 'thyroid_hypo.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.names', 'thyroid_hypo.description')

    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data', 'thyroid_ann.train.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-test.data', 'thyroid_ann.test.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-thyroid.names', 'thyroid_ann.description')
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-Readme', 'thyroid_ann.more_description')
    
    
    
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allbp.data', 'thyroid_all_bp.train.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allbp.test', 'thyroid_all_bp.test.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allbp.names', 'thyroid_all_bp.description')    
    
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allrep.data', 'thyroid_all_rep.train.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allrep.test', 'thyroid_all_rep.test.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allrep.names', 'thyroid_all_rep.description')    

    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhypo.data', 'thyroid_all_hypo.train.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhypo.test', 'thyroid_all_hypo.test.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhypo.names', 'thyroid_all_hypo.description') 

    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.data', 'thyroid_all_hyper.train.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.test', 'thyroid_all_hyper.test.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.names', 'thyroid_all_hyper.description')    
    

    
    #--------------------------------------------------
    
    data = load_mixed_raw_data('thyroid_sick_eu.data', sep = ',', header = False)
    categories = [u'sick-euthyroid', u'negative']
    data = replace_categories_in_mixed_data(data, categories, 0, separator = ',')
    
    for col in range(2, 15):
        categories = get_categories_in_mixed_data(data, col)
        data = replace_bin_cats_in_mixed_data(data, categories, col, separator = ',')
    
    columns = [16, 18, 20, 22, 24]  
    for col in columns:
        categories = get_categories_in_mixed_data(data, col)
        data = replace_bin_cats_in_mixed_data(data, categories, col, separator = ',')
        
    # The last column is still in bad shape. The next two lines fix this
    # problem by a little dirty trick
        
    write_mixed_raw_data(UCIVars.raw_data_folder + 'thyroid_sick_eu.data', data, sep = ',')
    data = load_mixed_raw_data('thyroid_sick_eu.data', sep = ',', header = False)
    
    data = auto_replace_missing_in_mixed_data(data, unknown_string = '?')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'thyroid_sick_eu.data', data, sep = ',')

    data = load_raw_data('thyroid_sick_eu.data', sep = ',', na_string = '?')
    save_data_to_file(data, 'thyroid_sick_eu', is_classification = True, is_regression = False) 
    
    
    #--------------------------------------------------
    
    concat_files(UCIVars.raw_data_folder + 'thyroid_sick.t*', UCIVars.raw_data_folder + 'thyroid_sick.data')
    replace_chars_in_file('thyroid_sick.data', '.|', ',')
    replace_chars_in_file('thyroid_sick.data', 'F', '0')
    replace_chars_in_file('thyroid_sick.data', 'M', '1')
    replace_chars_in_file('thyroid_sick.data', 'f', '0')
    replace_chars_in_file('thyroid_sick.data', 't', '1')
    replace_chars_in_file('thyroid_sick.data', ',0,?', ',0,0')
    
    data = load_mixed_raw_data('thyroid_sick.data', sep = ',', header = False)
    data = auto_replace_categories_in_mixed_data(data, 28, ',')
    data = auto_replace_categories_in_mixed_data(data, 29, ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'thyroid_sick.data', data, sep = ',')
    
    data = load_raw_data('thyroid_sick.data', sep = ',', na_string = '?')
    data = remove_columns(data, 34)
    data = move_label_in_front(data, 33)
    save_data_to_file(data, 'thyroid_sick', is_classification = True, is_regression = False) 
    

    #--------------------------------------------------

    concat_files(UCIVars.raw_data_folder + 'thyroid_dis.t*', UCIVars.raw_data_folder + 'thyroid_dis.data')
    replace_chars_in_file('thyroid_dis.data', '.|', ',')
    replace_chars_in_file('thyroid_dis.data', 'F', '0')
    replace_chars_in_file('thyroid_dis.data', 'M', '1')
    replace_chars_in_file('thyroid_dis.data', 'f', '0')
    replace_chars_in_file('thyroid_dis.data', 't', '1')
    replace_chars_in_file('thyroid_dis.data', ',0,?', ',0,0')
    
    data = load_mixed_raw_data('thyroid_dis.data', sep = ',', header = False)
    data = auto_replace_categories_in_mixed_data(data, 28, ',')
    data = auto_replace_categories_in_mixed_data(data, 29, ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'thyroid_dis.data', data, sep = ',')
    
    data = load_raw_data('thyroid_dis.data', sep = ',', na_string = '?')
    data = remove_columns(data, 34)
    data = move_label_in_front(data, 33)
    save_data_to_file(data, 'thyroid_dis', is_classification = True, is_regression = False) 
    
    
    #--------------------------------------------------

    replace_chars_in_file('thyroid_hypo.data', 'F', '0')
    replace_chars_in_file('thyroid_hypo.data', 'M', '1')
    replace_chars_in_file('thyroid_hypo.data', 'f', '0')
    replace_chars_in_file('thyroid_hypo.data', 't', '1')
    replace_chars_in_file('thyroid_hypo.data', 'n', '0')
    replace_chars_in_file('thyroid_hypo.data', 'y', '1')   
    replace_chars_in_file('thyroid_hypo.data', ',0,?', ',0,0')
    
    data = load_mixed_raw_data('thyroid_hypo.data', sep = ',', header = False)
    data = auto_replace_categories_in_mixed_data(data, 0, ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'thyroid_hypo.data', data, sep = ',')
        
    data = load_raw_data('thyroid_hypo.data', sep = ',', na_string = '?')
    save_data_to_file(data, 'thyroid_hypo', is_classification = True, is_regression = False) 
    
    
    #--------------------------------------------------
    
    concat_files(UCIVars.raw_data_folder + 'thyroid_ann.t*', UCIVars.raw_data_folder + 'thyroid_ann.data')
    data = load_raw_data('thyroid_ann.data', sep = ' ', na_string = '?')
    data = move_label_in_front(data, 21)
    save_data_to_file(data, 'thyroid_ann', is_classification = True, is_regression = False) 
    
    
    #--------------------------------------------------
    
    concat_files(UCIVars.raw_data_folder + 'thyroid_all_bp.t*', UCIVars.raw_data_folder + 'thyroid_all_bp.data')
    replace_chars_in_file('thyroid_all_bp.data', '.|', ',')
    replace_chars_in_file('thyroid_all_bp.data', 'F', '0')
    replace_chars_in_file('thyroid_all_bp.data', 'M', '1')
    replace_chars_in_file('thyroid_all_bp.data', 'f', '0')
    replace_chars_in_file('thyroid_all_bp.data', 't', '1')
    replace_chars_in_file('thyroid_all_bp.data', ',0,?', ',0,0')
    
    data = load_mixed_raw_data('thyroid_all_bp.data', sep = ',', header = False)
    data = auto_replace_categories_in_mixed_data(data, 28, ',')
    
    # We combine all 2 non-negative classes to one, they are all very small
    
    categories = sorted(get_categories_in_mixed_data(data, 29))
    data = replace_manual_in_mixed_data(data, categories, 29, (1, 1, 2), ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'thyroid_all_bp.data', data, sep = ',')
    
    data = load_raw_data('thyroid_all_bp.data', sep = ',', na_string = '?')
    data = remove_columns(data, 34)
    data = move_label_in_front(data, 33)
    
    save_data_to_file(data, 'thyroid_all_bp', is_classification = True, is_regression = False) 
    
    
    #--------------------------------------------------
    
    concat_files(UCIVars.raw_data_folder + 'thyroid_all_rep.t*', UCIVars.raw_data_folder + 'thyroid_all_rep.data')
    replace_chars_in_file('thyroid_all_rep.data', '.|', ',')
    replace_chars_in_file('thyroid_all_rep.data', 'F', '0')
    replace_chars_in_file('thyroid_all_rep.data', 'M', '1')
    replace_chars_in_file('thyroid_all_rep.data', 'f', '0')
    replace_chars_in_file('thyroid_all_rep.data', 't', '1')
    replace_chars_in_file('thyroid_all_rep.data', ',0,?', ',0,0')
    
    data = load_mixed_raw_data('thyroid_all_rep.data', sep = ',', header = False)
    data = auto_replace_categories_in_mixed_data(data, 28, ',')
    
    # We combine all 3 non-negative classes to one, they are all very small
    
    categories = sorted(get_categories_in_mixed_data(data, 29))
    data = replace_manual_in_mixed_data(data, categories, 29, (1, 2, 2, 2), ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'thyroid_all_rep.data', data, sep = ',')
    
    data = load_raw_data('thyroid_all_rep.data', sep = ',', na_string = '?')
    data = remove_columns(data, 34)
    data = move_label_in_front(data, 33)
    save_data_to_file(data, 'thyroid_all_rep', is_classification = True, is_regression = False) 
    
    
    #--------------------------------------------------
    
    concat_files(UCIVars.raw_data_folder + 'thyroid_all_hypo.t*', UCIVars.raw_data_folder + 'thyroid_all_hypo.data')
    replace_chars_in_file('thyroid_all_hypo.data', '.|', ',')
    replace_chars_in_file('thyroid_all_hypo.data', 'F', '0')
    replace_chars_in_file('thyroid_all_hypo.data', 'M', '1')
    replace_chars_in_file('thyroid_all_hypo.data', 'f', '0')
    replace_chars_in_file('thyroid_all_hypo.data', 't', '1')
    replace_chars_in_file('thyroid_all_hypo.data', ',0,?', ',0,0')
    
    data = load_mixed_raw_data('thyroid_all_hypo.data', sep = ',', header = False)
    data = auto_replace_categories_in_mixed_data(data, 28, ',')
    
    # We combine 'primary' and 'secondary' to a new class since 'secondary' only has 2 samples
    
    categories = sorted(get_categories_in_mixed_data(data, 29))
    data = replace_manual_in_mixed_data(data, categories, 29, (1, 2, 3, 3), ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'thyroid_all_hypo.data', data, sep = ',')
    
    data = load_raw_data('thyroid_all_hypo.data', sep = ',', na_string = '?')
    data = remove_columns(data, 34)
    data = move_label_in_front(data, 33)
    save_data_to_file(data, 'thyroid_all_hypo', is_classification = True, is_regression = False)    
    
    
    #--------------------------------------------------
    
    concat_files(UCIVars.raw_data_folder + 'thyroid_all_hyper.t*', UCIVars.raw_data_folder + 'thyroid_all_hyper.data')
    replace_chars_in_file('thyroid_all_hyper.data', '.|', ',')
    replace_chars_in_file('thyroid_all_hyper.data', 'F', '0')
    replace_chars_in_file('thyroid_all_hyper.data', 'M', '1')
    replace_chars_in_file('thyroid_all_hyper.data', 'f', '0')
    replace_chars_in_file('thyroid_all_hyper.data', 't', '1')
    replace_chars_in_file('thyroid_all_hyper.data', ',0,?', ',0,0')
    
    data = load_mixed_raw_data('thyroid_all_hyper.data', sep = ',', header = False)
    data = auto_replace_categories_in_mixed_data(data, 28, ',')
    
    # We combine all 4 non-negative classes to one, they are all very small
    
    categories = sorted(get_categories_in_mixed_data(data, 29))
    data = replace_manual_in_mixed_data(data, categories, 29, (1, 1, 1, 2, 1), ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'thyroid_all_hyper.data', data, sep = ',')
    
    data = load_raw_data('thyroid_all_hyper.data', sep = ',', na_string = '?')
    data = remove_columns(data, 34)
    data = move_label_in_front(data, 33)
    save_data_to_file(data, 'thyroid_all_hyper', is_classification = True, is_regression = False)    




#---------------------------------------------------------------------------------------------------

  
def get_isolet():

    prepare_new_data_set_group_id()
    #download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z', 'isolet.train.Z') 
    #download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z', 'isolet.test.Z') 
    
    print("ISOLET is currently not processed since:")
    print("  - all classes are rather small (around 300 each)")



#---------------------------------------------------------------------------------------------------

  
def get_mushroom():

    prepare_new_data_set_group_id()
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data', 'mushroom.data') 
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names', 'mushroom.description') 


    data = load_mixed_raw_data('mushroom.data', sep = ',', header = False)
    columns = numpy.shape(data)[1]
    for col in range(0, columns):
        data = auto_replace_categories_in_mixed_data(data, col, ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'mushroom.data', data, sep = ',')
    
    data = load_raw_data('mushroom.data', sep = ',')
    save_data_to_file(data, 'mushroom', is_classification = True, is_regression = False)   

#---------------------------------------------------------------------------------------------------

  
def get_assamese_characters():

    prepare_new_data_set_group_id()
    #download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00208/Online%20Handwritten%20Assamese%20Characters%20Dataset.rar', 'assamese_characters.rar') 
    
    print("Assamese Characters is currently not processed since:")
    print("  - all classes are rather small (around 45 each)")



#---------------------------------------------------------------------------------------------------

  
def get_arabic_digit():

    prepare_new_data_set_group_id()
    #download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00195/Test_Arabic_Digit.txt', 'arabic_digit.test.data') 
    #download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00195/Train_Arabic_Digit.txt', 'arabic_digit.train.data') 
    #download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00195/documentation.html', 'arabic_digit.html') 
    
    print("Arabic Digits is currently not processed since:")
    print("  - I could not find the time to figure out the format")


#---------------------------------------------------------------------------------------------------

  
def get_eeg_steady_state_visual():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00457/BCI-SSVEP_Database_Aceves.zip', 'eeg_steady_state_visual.zip') 
    
    print("EMG Physical Action is currently not processed since:")
    print("  - the data comes in a rather convoluted form")
    print("  - it truly seems to be time series data")



#---------------------------------------------------------------------------------------------------

  
def get_gesture_phase_segmentation():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00302/gesture_phase_dataset.zip', 'gesture_phase_segmentation.zip') 
    unzip_raw_data('gesture_phase_segmentation.zip')

    os.rename(UCIVars.raw_data_folder + 'data_description.txt', UCIVars.raw_data_folder + 'gesture_phase_segmentation.description')

    letters = ['a', 'b', 'c']
    versions = ['raw', 'va3']
    
    for version in versions:
        for letter in letters:
            concat_files(UCIVars.raw_data_folder + letter + '?_' + version + '.csv', UCIVars.raw_data_folder + 'gesture_phase_segmentation.' + letter + version + '.data')
            remove_files(UCIVars.raw_data_folder, letter + '?_' + version + '.csv')
            
        tmp_filename = 'gesture_phase_segmentation.?' + version + '.data'  
        version_filename = 'gesture_phase_segmentation_' + version + '.data'
            
        concat_files(UCIVars.raw_data_folder + tmp_filename, UCIVars.raw_data_folder + version_filename)
        remove_files(UCIVars.raw_data_folder, tmp_filename)
        
        if version == 'raw':
            replace_chars_in_file(version_filename, 'Rest', '1')
            replace_chars_in_file(version_filename, 'Preparation', '2')
            replace_chars_in_file(version_filename, 'Stroke', '3')
            replace_chars_in_file(version_filename, 'Hold', '4')
            replace_chars_in_file(version_filename, 'Retraction', '5')    
        else:
            replace_chars_in_file(version_filename, 'D', '1')
            replace_chars_in_file(version_filename, 'P', '2')
            replace_chars_in_file(version_filename, 'S', '3')
            replace_chars_in_file(version_filename, 'H', '4')
            replace_chars_in_file(version_filename, 'R', '5')    
        
        data = load_raw_data(version_filename, sep = ',')
        columns = numpy.shape(data)[1]
        
        data = move_label_in_front(data, columns - 1)
        save_data_to_file(data, 'gesture_phase_segmentation_' + version, is_classification = True, is_regression = False)   
    

    
    

#---------------------------------------------------------------------------------------------------

  
def get_emg_physical_action():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00213/EMG%20Physical%20Action%20Data%20Set.rar', 'emg_physical_action.rar') 
    #unrar_raw_data('emg_physical_action.rar')
    
    print("EMG Physical Action is currently not processed since:")
    print("  - the data comes in a rather convoluted form")


#---------------------------------------------------------------------------------------------------

  
def get_human_activity_smartphone():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip', 'human_activity_smartphone.zip') 
    unzip_raw_data('human_activity_smartphone.zip')
    
    os.rename(UCIVars.raw_data_folder + 'UCI HAR Dataset/train/X_train.txt', UCIVars.raw_data_folder + 'human_activity_smartphone.train.data')
    os.rename(UCIVars.raw_data_folder + 'UCI HAR Dataset/test/X_test.txt', UCIVars.raw_data_folder + 'human_activity_smartphone.test.data')

    os.rename(UCIVars.raw_data_folder + 'UCI HAR Dataset/train/y_train.txt', UCIVars.raw_data_folder + 'human_activity_smartphone.train.labels.data')
    os.rename(UCIVars.raw_data_folder + 'UCI HAR Dataset/test/y_test.txt', UCIVars.raw_data_folder + 'human_activity_smartphone.test.labels.data')

    os.rename(UCIVars.raw_data_folder + 'UCI HAR Dataset/features_info.txt', UCIVars.raw_data_folder + 'human_activity_smartphone.features.txt')
    os.rename(UCIVars.raw_data_folder + 'UCI HAR Dataset/README.txt', UCIVars.raw_data_folder + 'human_activity_smartphone.description')

    shutil.rmtree(UCIVars.raw_data_folder + 'UCI HAR Dataset')
    shutil.rmtree(UCIVars.raw_data_folder + '__MACOSX')

    replace_chars_in_file('human_activity_smartphone.train.data', '  ', ' ')
    replace_chars_in_file('human_activity_smartphone.test.data', '  ', ' ')
    
    train_data = load_raw_data('human_activity_smartphone.train.data', sep = ' ')
    train_label = load_raw_data('human_activity_smartphone.train.labels.data', sep = ',')
    train_data = numpy.concatenate((train_label, train_data), axis = 1)
    
    test_data = load_raw_data('human_activity_smartphone.test.data', sep = ' ')
    test_label = load_raw_data('human_activity_smartphone.test.labels.data', sep = ',')
    test_data = numpy.concatenate((test_label, test_data), axis = 1)
    
    data = numpy.concatenate((train_data, test_data), axis = 0)
        
    save_data_to_file(data, 'human_activity_smartphone', is_classification = True, is_regression = False) 




#---------------------------------------------------------------------------------------------------

  
def get_polish_companies_bankruptcy():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip', 'polish_companies_bankruptcy.zip')
    unzip_raw_data('polish_companies_bankruptcy.zip')
    

    for i in range(1, 6):
        unarff_raw_data(str(i) + 'year')
        remove_files(UCIVars.raw_data_folder, str(i) + 'year.arff')
        os.rename(UCIVars.raw_data_folder + str(i) + 'year.data', UCIVars.raw_data_folder + 'polish_companies_bankruptcy_' + str(i) + 'year.data')
        replace_chars_in_file('polish_companies_bankruptcy_' + str(i) + 'year.data', 'nan', '?')
        
        data = load_mixed_raw_data('polish_companies_bankruptcy_' + str(i) + 'year.data', sep = ',')
        data = auto_replace_missing_in_mixed_data(data, unknown_string = '?')
        write_mixed_raw_data(UCIVars.raw_data_folder + 'polish_companies_bankruptcy_' + str(i) + 'year.trafo.data', data, sep = ',')
        
        data = load_raw_data('polish_companies_bankruptcy_' + str(i) + 'year.trafo.data', sep = ',')
        data = move_label_in_front(data, 64)
        save_data_to_file(data, 'polish_companies_bankruptcy_' + str(i) + 'year', is_classification = True, is_regression = False)   

    
    
#---------------------------------------------------------------------------------------------------

  
def get_crowd_sourced_mapping():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00400/Crowdsourced%20Mapping.zip', 'crowd_sourced_mapping.zip')
    unzip_raw_data('crowd_sourced_mapping.zip')
    
    os.rename(UCIVars.raw_data_folder + 'training.csv', UCIVars.raw_data_folder + 'crowd_sourced_mapping.train.data')
    os.rename(UCIVars.raw_data_folder + 'testing.csv', UCIVars.raw_data_folder + 'crowd_sourced_mapping.test.data')
    
    
    # Get rid of the headers ...
    
    train_data = load_mixed_raw_data('crowd_sourced_mapping.train.data', sep = ',', header = True)
    write_mixed_raw_data(UCIVars.raw_data_folder + 'crowd_sourced_mapping.train.data', train_data, sep = ',')
    
    test_data = load_mixed_raw_data('crowd_sourced_mapping.test.data', sep = ',', header = True)
    write_mixed_raw_data(UCIVars.raw_data_folder + 'crowd_sourced_mapping.test.data', test_data, sep = ',')
    
    concat_files(UCIVars.raw_data_folder + 'crowd_sourced_mapping.*.data', UCIVars.raw_data_folder + 'crowd_sourced_mapping.data')
    
    
    # The data set actually has the following classes: ['impervious', 'orchard', 'farm', 'water', 'forest', 'grass']
    # However, 'orchard' and 'water' only occur 100 and 250 times, respectively. Ignoring them during the 
    # replacement below leads eventually to a 4-class problem with the remaining classes.
    
    data = load_mixed_raw_data('crowd_sourced_mapping.data', sep = ',', header = True)
    categories = ['impervious', 'farm', 'forest', 'grass']
    data = replace_ordinals_in_mixed_data(data, categories, 0, separator = ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'crowd_sourced_mapping.data', data, sep = ',')
    

    data = load_raw_data('crowd_sourced_mapping.data', sep = ',')
    save_data_to_file(data, 'crowd_sourced_mapping', is_classification = True, is_regression = False) 
    

    
#---------------------------------------------------------------------------------------------------

  
def get_firm_teacher_clave(): 
  
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00324/ClaveVectors_Firm-Teacher_Model.txt', 'firm_teacher_clave.data') 
    
    replace_chars_in_file('firm_teacher_clave.data', ' ', ',')
    replace_chars_in_file('firm_teacher_clave.data', 'error,fixed', '')
    replace_chars_in_file('firm_teacher_clave.data', ',	', '')
 
    
    data = load_mixed_raw_data('firm_teacher_clave.data', sep = ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'firm_teacher_clave.data', data, sep = ',')
    
    data = load_raw_data('firm_teacher_clave.data', sep = ',')
    
    # The data set has four classes, and their labels are stored as a four-dimensional
    # 'categorial'-vector. The following lines convert this format to the usual one.

    rows = numpy.shape(data)[0]
    columns = numpy.shape(data)[1]
    label_vectors = data[0:rows, columns - 4:columns]
    data_features = data[0:rows, 0:columns - 4]    
    
    labels = numpy.zeros(shape = (rows, 1))    
    for i in range(0, 4):
        labels[numpy.where(label_vectors[0:rows, i] == 1)] = i
    
    data = numpy.concatenate((labels, data_features), axis = 1)
    save_data_to_file(data, 'firm_teacher_clave', is_classification = True, is_regression = False) 

    
#---------------------------------------------------------------------------------------------------

  
def get_smartphone_human_activity_postural():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00341/HAPT%20Data%20Set.zip', 'smartphone_human_activity_postural.zip') 
    unzip_raw_data('smartphone_human_activity_postural.zip')
    
    os.rename(UCIVars.raw_data_folder + 'Train/X_train.txt', UCIVars.raw_data_folder + 'smartphone_human_activity_postural.train.data')
    os.rename(UCIVars.raw_data_folder + 'Test/X_test.txt', UCIVars.raw_data_folder + 'smartphone_human_activity_postural.test.data')

    os.rename(UCIVars.raw_data_folder + 'Train/y_train.txt', UCIVars.raw_data_folder + 'smartphone_human_activity_postural.train.labels.data')
    os.rename(UCIVars.raw_data_folder + 'Test/y_test.txt', UCIVars.raw_data_folder + 'smartphone_human_activity_postural.test.labels.data')

    os.rename(UCIVars.raw_data_folder + 'features_info.txt', UCIVars.raw_data_folder + 'smartphone_human_activity_postural.features.txt')
    os.rename(UCIVars.raw_data_folder + 'README.txt', UCIVars.raw_data_folder + 'smartphone_human_activity_postural.description')

    shutil.rmtree(UCIVars.raw_data_folder + 'Train')
    shutil.rmtree(UCIVars.raw_data_folder + 'Test')
    shutil.rmtree(UCIVars.raw_data_folder + 'RawData')
    os.remove(UCIVars.raw_data_folder + 'features.txt')
    os.remove(UCIVars.raw_data_folder + 'activity_labels.txt')
    
    
    train_data = load_raw_data('smartphone_human_activity_postural.train.data', sep = ' ')
    train_label = load_raw_data('smartphone_human_activity_postural.train.labels.data', sep = ',')
    train_data = numpy.concatenate((train_label, train_data), axis = 1)
    
    test_data = load_raw_data('smartphone_human_activity_postural.test.data', sep = ' ')
    test_label = load_raw_data('smartphone_human_activity_postural.test.labels.data', sep = ',')
    test_data = numpy.concatenate((test_label, test_data), axis = 1)
    
    data = numpy.concatenate((train_data, test_data), axis = 0)
    
    # The transitional classes 7 to 12 are very small compared to the first 6 classes. Since
    # we are mostly interested in data sets for which no extra care is needed, we remove these
    # six classes.
    
    rows = numpy.shape(data)[0]
    columns = numpy.shape(data)[1]
    data = data[numpy.where(data[0:rows, 0] <= 6)[0], 0:columns]
    
    save_data_to_file(data, 'smartphone_human_activity_postural', is_classification = True, is_regression = False) 
    
    
#---------------------------------------------------------------------------------------------------

  
def get_pen_recognition_handwritten_characters():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra', 'pen_recognition_handwritten_characters.train.data') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes', 'pen_recognition_handwritten_characters.test.data') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.names', 'pen_recognition_handwritten_characters.description')    
    
    concat_files(UCIVars.raw_data_folder + 'pen_recognition_handwritten_characters.*.data', UCIVars.raw_data_folder + 'pen_recognition_handwritten_characters.data')
    replace_chars_in_file('pen_recognition_handwritten_characters.data', '  ', '')
    replace_chars_in_file('pen_recognition_handwritten_characters.data', ' ', '')
    
    data = load_raw_data('pen_recognition_handwritten_characters.data', sep = ',')

    data = move_label_in_front(data, 16)
    save_data_to_file(data, 'pen_recognition_handwritten_characters', is_classification = True, is_regression = False) 


#---------------------------------------------------------------------------------------------------

  
def get_epileptic_seizure_recognition():

    print("Epileptic seizure recognition is currently not processed since:")
    print("  - it was removed from the UCI repository")
    
    #prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv', 'epileptic_seizure_recognition.data')

    #data = load_raw_data('epileptic_seizure_recognition.data', description_columns = 1, sep = ',')
    #data = move_label_in_front(data, 178)
    #save_data_to_file(data, 'epileptic_seizure_recognition', is_classification = True, is_regression = False)   


#---------------------------------------------------------------------------------------------------

  
def get_nursery():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data', 'nursery.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.names', 'nursery.description')

    data = load_mixed_raw_data('nursery.data', ',')

    categories = [u'usual', u'pretentious', u'great_pret']
    data = replace_ordinals_in_mixed_data(data, categories, 0, separator = ',')
    
    categories = [u'proper', u'less_proper', u'improper', u'critical', u'very_crit']
    data = replace_ordinals_in_mixed_data(data, categories, 1, separator = ',')    
    
    categories = [u'complete', u'completed', u'incomplete', u'foster']
    data = replace_ordinals_in_mixed_data(data, categories, 2, separator = ',')      
    
    categories = [u'1', u'3', u'2', u'more']
    data = replace_ordinals_in_mixed_data(data, categories, 3, separator = ',')      

    categories = [u'convenient', u'less_conv', u'critical']
    data = replace_ordinals_in_mixed_data(data, categories, 4, separator = ',')       
    
    categories = [u'convenient', u'inconv']
    data = replace_ordinals_in_mixed_data(data, categories, 5, separator = ',')         

    categories = [u'nonprob', u'slightly_prob', u'problematic']
    data = replace_ordinals_in_mixed_data(data, categories, 6, separator = ',')        

    categories = [u'not_recom', u'recommended', u'priority']
    data = replace_ordinals_in_mixed_data(data, categories, 7, separator = ',')    
    
    
    # We combine the classes 'not_recom' and 'recommend', since the latter only has two instances
    
    categories = [u'recommend']
    data = replace_ordinals_in_mixed_data(data, categories, 8, separator = ',') 
    categories = [u'not_recom', u'very_recom', u'priority', u'spec_prior']
    data = replace_ordinals_in_mixed_data(data, categories, 8, separator = ',')   
    
    write_mixed_raw_data(UCIVars.raw_data_folder + 'nursery.trafo.data', data, sep = ',')
    
    
    data = load_raw_data('nursery.trafo.data', sep = ',')
    data = move_label_in_front(data, 8)
    save_data_to_file(data, 'nursery', is_classification = True, is_regression = True)   


#---------------------------------------------------------------------------------------------------

  
def get_indoor_user_movement_prediction():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00348/MovementAAL.zip', 'indoor_user_movement_prediction.zip')

    print("Indoor User Movement Prediction is currently not processed since:")
    print("  - according to the description it seems to be a time series data set")
    print("  - the number of time series samples is small, namely a few hundreds")
    
    

#---------------------------------------------------------------------------------------------------

  
def get_eeg_eye_state():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff', 'eeg_eye_state.arff')
    unarff_raw_data('eeg_eye_state')
    
    data = load_raw_data('eeg_eye_state.data', sep = ',')
    data = move_label_in_front(data, 14)
    save_data_to_file(data, 'eeg_eye_state', is_classification = True, is_regression = False)


#---------------------------------------------------------------------------------------------------

  
def get_htru2():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip', 'htru2.zip')
    unzip_raw_data('htru2.zip')

    os.rename(UCIVars.raw_data_folder + 'HTRU_2.csv', UCIVars.raw_data_folder + 'htru2.data')
    os.rename(UCIVars.raw_data_folder + 'Readme.txt', UCIVars.raw_data_folder + 'htru2.description')
    os.remove(UCIVars.raw_data_folder + 'HTRU_2.arff')
    
    
    # Somehow, the original htru2.data file has a strange format, so that all data is 
    # viewed to be as a single row. Probably, the endofline characters are messed up. 
    # In any case, the following two lines cure this.
    
    data = load_mixed_raw_data('htru2.data', ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'htru2.data', data, sep = ',')
    
    data = load_raw_data('htru2.data', ',')
    data = move_label_in_front(data, 8)
    save_data_to_file(data, 'htru2', is_classification = True, is_regression = False) 



#---------------------------------------------------------------------------------------------------

  
def get_magic_gamma_telescope():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data', 'magic_gamma_telescope.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.names', 'magic_gamma_telescope.description')
    
    replace_chars_in_file('magic_gamma_telescope.data', 'g', '1')
    replace_chars_in_file('magic_gamma_telescope.data', 'h', '-1')
    
    data = load_raw_data('magic_gamma_telescope.data', ',')
    data = move_label_in_front(data, 10)
    save_data_to_file(data, 'magic_gamma_telescope', is_classification = True, is_regression = False) 
    


#---------------------------------------------------------------------------------------------------

  
def get_letter_recognition():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data', 'letter_recognition.data')    
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.names', 'letter_recognition.description')    
    
    data = load_mixed_raw_data('letter_recognition.data', sep = ',')
    categories = get_categories_in_mixed_data(data, 0)
    data = replace_ordinals_in_mixed_data(data, sorted(categories), 0, separator = ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'letter_recognition.data', data, sep = ',')
    
    data = load_raw_data('letter_recognition.data', ',')
    save_data_to_file(data, 'letter_recognition', is_classification = True, is_regression = False) 



#---------------------------------------------------------------------------------------------------

  
def get_occupancy_detection():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip', 'occupancy_detection.zip')
    unzip_raw_data('occupancy_detection.zip')
    
    os.rename(UCIVars.raw_data_folder + 'datatraining.txt', UCIVars.raw_data_folder + 'occupancy_detection.train.data')
    os.rename(UCIVars.raw_data_folder + 'datatest.txt', UCIVars.raw_data_folder + 'occupancy_detection.val.data')
    os.rename(UCIVars.raw_data_folder + 'datatest2.txt', UCIVars.raw_data_folder + 'occupancy_detection.test.data')
    concat_files(UCIVars.raw_data_folder + 'occupancy_detection.*.data', UCIVars.raw_data_folder + 'occupancy_detection.data')
    
    replace_chars_in_file('occupancy_detection.data', ' ', ',')
    replace_chars_in_file('occupancy_detection.data', '"', '')
    
    data = load_raw_data('occupancy_detection.data', ',', description_columns = 1, date_column = 1, date_sep = '-', date_order = 'Ymd', time_column = 2, time_sep = ':')
    data = move_label_in_front(data, 7)
    save_data_to_file(data, 'occupancy_detection', is_classification = True, is_regression = False)
    
    
#---------------------------------------------------------------------------------------------------

  
def get_avila():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip', 'avila.zip')
    unzip_raw_data('avila.zip')
    
    os.rename(UCIVars.raw_data_folder + 'avila/avila-description.txt', UCIVars.raw_data_folder + 'avila.description')
    os.rename(UCIVars.raw_data_folder + 'avila/avila-tr.txt', UCIVars.raw_data_folder + 'avila.train.data')
    os.rename(UCIVars.raw_data_folder + 'avila/avila-ts.txt', UCIVars.raw_data_folder + 'avila.test.data')
    shutil.rmtree(UCIVars.raw_data_folder + 'avila')
    
    concat_files(UCIVars.raw_data_folder + 'avila.*.data', UCIVars.raw_data_folder + 'avila.data')
    
    replace_chars_in_file('avila.data', 'A', '1')
    replace_chars_in_file('avila.data', 'B', '2')
    replace_chars_in_file('avila.data', 'C', '3')
    replace_chars_in_file('avila.data', 'D', '4')
    replace_chars_in_file('avila.data', 'E', '5')
    replace_chars_in_file('avila.data', 'F', '6')
    replace_chars_in_file('avila.data', 'G', '7')
    replace_chars_in_file('avila.data', 'H', '8')
    replace_chars_in_file('avila.data', 'I', '9')
    replace_chars_in_file('avila.data', 'W', '10')
    replace_chars_in_file('avila.data', 'X', '11')
    replace_chars_in_file('avila.data', 'Y', '12')
    
    data = load_raw_data('avila.data', ',')
    data = move_label_in_front(data, 10)
    save_data_to_file(data, 'avila', is_classification = True, is_regression = False) 


#---------------------------------------------------------------------------------------------------

  
def get_grammatical_facial_expressions():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00317/grammatical_facial_expression.zip', 'grammatical_facial_expression.zip')

    print("Activity Recognition is currently not processed since:")
    print("  - according to the description it seems to be a time series data set")
    print("  - the number of time series samples is very low, namely 36")

#---------------------------------------------------------------------------------------------------


def get_chess_krvk():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data', 'chess_krvk.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.info', 'chess_krvk.description')

    
    data = load_mixed_raw_data('chess_krvk.data', sep = ',')
        
    data = auto_replace_categories_in_mixed_data(data, 0, separator = ',')
    data = auto_replace_categories_in_mixed_data(data, 2, separator = ',')
    data = auto_replace_categories_in_mixed_data(data, 4, separator = ',')

    categories = ['draw', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen']
    data = replace_ordinals_in_mixed_data(data, categories, 6, separator = ',', begin_value = -1)

    write_mixed_raw_data(UCIVars.raw_data_folder + 'chess_krvk.trafo.data', data, sep = ',')  

    data = load_raw_data('chess_krvk.trafo.data', sep = ',')
    data = move_label_in_front(data, 23)
    save_data_to_file(data, 'chess_krvk', is_classification = True, is_regression = True)



#---------------------------------------------------------------------------------------------------


def get_default_credit_card():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls', 'default_credit_card.xls')

    excel_data = pandas.read_excel(UCIVars.raw_data_folder + 'default_credit_card.xls', engine = 'xlrd')
    excel_data.to_csv(UCIVars.raw_data_folder + 'default_credit_card.data')
    
    
    data = load_raw_data('default_credit_card.data', sep = ',', description_columns = 1)
    data = move_label_in_front(data, 24)
    save_data_to_file(data, 'default_credit_card', is_classification = True, is_regression = False) 



#---------------------------------------------------------------------------------------------------


def get_nomao():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00227/Nomao.zip', 'nomao.zip')
    unzip_raw_data('nomao.zip')
    
    os.rename(UCIVars.raw_data_folder + 'Nomao/Nomao.data', UCIVars.raw_data_folder + 'nomao.data')
    os.rename(UCIVars.raw_data_folder + 'Nomao/Nomao.names', UCIVars.raw_data_folder + 'nomao.description')
    
    shutil.rmtree(UCIVars.raw_data_folder + 'Nomao')
    
    replace_chars_in_file('nomao.data', '#', ',')
    
    data = load_mixed_raw_data('nomao.data', sep = ',', header = False)
    
    categories = ['s', 'm', 'n']
    columns = [8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 64, 65, 72, 73, 80, 81, 88, 89, 93, 97, 101, 105, 109, 113, 117]
    
    for i in range(len(columns)):
        data = replace_ordinals_in_mixed_data(data, categories, columns[i], ',', unknown_string = '')

    data = auto_replace_missing_in_mixed_data(data, unknown_string = '?')

     
    write_mixed_raw_data(UCIVars.raw_data_folder + 'nomao.trafo.data', data, sep = ',')  
    
    data = load_raw_data('nomao.trafo.data', sep = ',')
    data = move_label_in_front(data, 120)
    save_data_to_file(data, 'nomao', is_classification = True, is_regression = False) 


#---------------------------------------------------------------------------------------------------


def get_indoor_loc_mag():

    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00343/UJIIndoorLoc-Mag-forUCI.zip', 'indoor_loc_mag.zip')
    
    print("Indoor Location Mag is currently not processed since:")
    print("  - according to the description it seems to be a time series data set")
    print("  - the number of time series samples is too low")


#---------------------------------------------------------------------------------------------------

  
def get_activity_recognition():
    
    prepare_new_data_set_group_id()
    #download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00366/AReM.zip', 'activity_recognition.zip')


    print("Activity Recognition is currently not processed since:")
    print("  - according to the description it seems to be a time series data set")
    print("  - the number of time series samples is too low")

  
#---------------------------------------------------------------------------------------------------

  
def get_bank_marketing():

    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip', 'bank_marketing.zip') 
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip', 'bank_marketing_additional.zip') 
    
    
    unzip_raw_data('bank_marketing.zip')
    os.rename(UCIVars.raw_data_folder + 'bank-full.csv', UCIVars.raw_data_folder + 'bank_marketing.data')
    os.rename(UCIVars.raw_data_folder + 'bank-names.txt', UCIVars.raw_data_folder + 'bank_marketing.description')
    os.remove(UCIVars.raw_data_folder + 'bank.csv')
    

    replace_chars_in_file('bank_marketing.data', '"', '')
    data = load_mixed_raw_data('bank_marketing.data', sep = ';', header = True)

    categories = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
    data = replace_categories_in_mixed_data(data, categories, 1, ';', unknown_string = 'unknown', unknown_replacement_value = 0)

    categories = ['divorced', 'married', 'single']
    data = replace_categories_in_mixed_data(data, categories, 2, ';', unknown_string = 'unknown', unknown_replacement_value = 0)

    categories = ['primary', 'secondary', 'tertiary']
    data = replace_ordinals_in_mixed_data(data, categories, 3, ';', unknown_string = '')

    categories = ['no', 'yes']
    data = replace_bin_cats_in_mixed_data(data, categories, 4, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    data = replace_bin_cats_in_mixed_data(data, categories, 6, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    data = replace_bin_cats_in_mixed_data(data, categories, 7, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    data = replace_bin_cats_in_mixed_data(data, categories, 16, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    
    categories = ['cellular', 'telephone']
    data = replace_bin_cats_in_mixed_data(data, categories, 8, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    
    categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    data = replace_circulars_in_mixed_data(data, categories, 10, ';', unknown_string = 'unknown')
    
    categories = ['failure', 'success']
    data = replace_bin_cats_in_mixed_data(data, categories, 15, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    
    write_mixed_raw_data(UCIVars.raw_data_folder + 'bank_marketing.trafo.data', data, sep = ';')


    data = load_raw_data('bank_marketing.trafo.data', sep = ';', na_string = 'unknown')
    data = move_label_in_front(data, 29)
    save_data_to_file(data, 'bank_marketing', is_classification = True, is_regression = False) 

    
#------------------------------------------------
    

    unzip_raw_data('bank_marketing_additional.zip')
    shutil.rmtree(UCIVars.raw_data_folder + '__MACOSX')
    os.rename(UCIVars.raw_data_folder + 'bank-additional/bank-additional-full.csv', UCIVars.raw_data_folder + 'bank_marketing_additional.data')
    os.rename(UCIVars.raw_data_folder + 'bank-additional/bank-additional-names.txt', UCIVars.raw_data_folder + 'bank_marketing_additional.description')
    shutil.rmtree(UCIVars.raw_data_folder + 'bank-additional')


    replace_chars_in_file('bank_marketing_additional.data', '"', '')
    data = load_mixed_raw_data('bank_marketing_additional.data', sep = ';', header = True)
       
    categories = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
    data = replace_categories_in_mixed_data(data, categories, 1, ';', unknown_string = 'unknown', unknown_replacement_value = 0)

    categories = ['divorced', 'married', 'single']
    data = replace_categories_in_mixed_data(data, categories, 2, ';', unknown_string = 'unknown', unknown_replacement_value = 0)

    categories = ['illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree']
    data = replace_ordinals_in_mixed_data(data, categories, 3, ';', unknown_string = '')

    categories = ['no', 'yes']
    data = replace_bin_cats_in_mixed_data(data, categories, 4, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    data = replace_bin_cats_in_mixed_data(data, categories, 5, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    data = replace_bin_cats_in_mixed_data(data, categories, 6, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    data = replace_bin_cats_in_mixed_data(data, categories, 20, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    
    categories = ['cellular', 'telephone']
    data = replace_bin_cats_in_mixed_data(data, categories, 7, ';', unknown_string = 'unknown', unknown_replacement_value = 0)
    
    categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    data = replace_circulars_in_mixed_data(data, categories, 8, ';', unknown_string = 'unknown')

    categories = ['mon', 'tue', 'wed', 'thu', 'fri']
    data = replace_circulars_in_mixed_data(data, categories, 9, ';', unknown_string = 'unknown')
    
    categories = ['failure', 'success']
    data = replace_bin_cats_in_mixed_data(data, categories, 14, ';', unknown_string = 'nonexistent', unknown_replacement_value = 0)
    

    write_mixed_raw_data(UCIVars.raw_data_folder + 'bank_marketing_additional.trafo.data', data, sep = ';')

    data = load_raw_data('bank_marketing_additional.trafo.data', sep = ';')
    data = move_label_in_front(data, 34)
    save_data_to_file(data, 'bank_marketing_additional', is_classification = True, is_regression = False) 


    

    
#---------------------------------------------------------------------------------------------------

  
def get_census_income():
    
    prepare_new_data_set_group_id()
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 'adult.train.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', 'adult.test.data')
    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names', 'adult.description')
    if os.path.exists(UCIVars.raw_data_folder + 'adult.trafo.data'):
        os.remove(UCIVars.raw_data_folder + 'adult.trafo.data')

    concat_files(UCIVars.raw_data_folder + 'adult.t*.data', UCIVars.raw_data_folder + 'adult.data')
    replace_chars_in_file('adult.data', '>50K.', '>50K') 
    replace_chars_in_file('adult.data', '<=50K.', '<=50K') 
    replace_chars_in_file('adult.data', '|1x3 Cross validator', '')
    
    replace_chars_in_file('adult.data', ', ', ',')
    
    data = load_mixed_raw_data('adult.data', sep = ',', header = False)
          
    categories = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
    data = replace_categories_in_mixed_data(data, categories, 1, ',', unknown_string = '')
    
    categories = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors', 'Prof-school', 'Masters', 'Doctorate']
    data = replace_ordinals_in_mixed_data(data, categories, 3, ',', unknown_string = '')
    
    categories = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    data = replace_categories_in_mixed_data(data, categories, 5, ',', unknown_string = '')
    
    categories = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
    data = replace_categories_in_mixed_data(data, categories, 6, ',', unknown_string = '')
    
    categories = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    data = replace_categories_in_mixed_data(data, categories, 7, ',', unknown_string = '')
    
    categories = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    data = replace_categories_in_mixed_data(data, categories, 8, ',', unknown_string = '')
    
    categories = ['Female', 'Male']
    data = replace_bin_cats_in_mixed_data(data, categories, 9, ',', unknown_string = '')
    
    categories = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    data = replace_categories_in_mixed_data(data, categories, 13, ',', unknown_string = '')
    
    categories = ['<=50K', '>50K']
    data = replace_bin_cats_in_mixed_data(data, categories, 14, ',', unknown_string = '')
    
    write_mixed_raw_data(UCIVars.raw_data_folder + 'adult.trafo.data', data, sep = ',')
    
    
    data = load_raw_data('adult.trafo.data', sep = ',')
    data = move_label_in_front(data, 89)
    
    save_data_to_file(data, 'adult', is_classification = True, is_regression = False) 






#---------------------------------------------------------------------------------------------------

  
def get_emg_for_gestures():

    prepare_new_data_set_group_id()


    print("EMG for Gestures is currently not processed since:")
    print("  - according to the description it seems to be a time series data set")



#---------------------------------------------------------------------------------------------------

  
def get_indoor_channel_measurements():

    prepare_new_data_set_group_id()


    print("Indoor Channel Measurements is currently not processed since:")
    print("  - according to the description it seems to be a complicated time series data set")
    
    
#---------------------------------------------------------------------------------------------------

  
def get_electrical_grid_stability_simulated():
    
    prepare_new_data_set_group_id()
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv', 'electrical_grid_stability_simulated.data')
    
    data = load_mixed_raw_data('electrical_grid_stability_simulated.data', sep = ',', header = True)
    
    categories = get_categories_in_mixed_data(data, 13)
    data = replace_bin_cats_in_mixed_data(data, categories, 13, ',')
    
    write_mixed_raw_data(UCIVars.raw_data_folder + 'electrical_grid_stability_simulated.data', data, sep = ',')
    
    
    data = load_raw_data('electrical_grid_stability_simulated.data', ',')

    data_class = move_label_in_front(data, 13)
    data_class = remove_columns(data_class, 13)
    save_data_to_file(data_class, 'electrical_grid_stability_simulated', is_classification = True, is_regression = False) 
    
    
    data_regr = move_label_in_front(data, 12)
    data_regr = remove_columns(data_regr, 13)
    save_data_to_file(data_regr, 'electrical_grid_stability_simulated', is_classification = False, is_regression = True) 
    

#---------------------------------------------------------------------------------------------------


def get_online_shoppers_attention():

    prepare_new_data_set_group_id()
    
    download_and_save('http://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv', 'online_shoppers_attention.data')

    data = load_mixed_raw_data('online_shoppers_attention.data', sep = ',', header = True)
    
    data = auto_replace_categories_in_mixed_data(data, 16, ',')
    data = auto_replace_categories_in_mixed_data(data, 17, ',')
    
    categories = get_categories_in_mixed_data(data, 15)
    data = replace_categories_in_mixed_data(data, categories, 15, ',')
    
    categories = [u'Jan', u'Feb', u'Mar', u'Apr', u'May', u'June', u'Jul', u'Aug', u'Sep', u'Oct', u'Nov', u'Dec']
    data = replace_circulars_in_mixed_data(data, categories, 10, ',')
    
    write_mixed_raw_data(UCIVars.raw_data_folder + 'online_shoppers_attention.data', data, sep = ',')
    
    
    data = load_raw_data('online_shoppers_attention.data', ',')
    data = move_label_in_front(data, 20)
    save_data_to_file(data, 'online_shoppers_attention', is_classification = True, is_regression = False) 


#---------------------------------------------------------------------------------------------------


def get_pmu_ud():
    
    prepare_new_data_set_group_id()


    print("PMU-UD is currently not processed since:")
    print("  - the data consists of .jpg images")
    

#---------------------------------------------------------------------------------------------------


def get_seoul_bike_data():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv', 'seoul_bike_data.data')

    # The purpose of the following two lines is to remove the header, which gives an annoying encoding error...
    data = pandas.read_csv(UCIVars.raw_data_folder + 'seoul_bike_data.data', encoding = 'unicode_escape')
    data.to_csv(UCIVars.raw_data_folder + 'seoul_bike_data.data', header = False, index = False)
    
    data = load_mixed_raw_data('seoul_bike_data.data', sep = ',', header = False)

    categories = ['No', 'Yes']
    data = replace_bin_cats_in_mixed_data(data, categories, column = 13, separator = ',')

    categories = ['No Holiday', 'Holiday']
    data = replace_bin_cats_in_mixed_data(data, categories, column = 12, separator = ',')

    categories = ['Winter', 'Spring', 'Summer', 'Autumn']
    data = replace_circulars_in_mixed_data(data, categories, 11, ',')

    write_mixed_raw_data(UCIVars.raw_data_folder + 'seoul_bike_data.data', data, sep = ',')
    data = load_raw_data('seoul_bike_data.data', sep=',', date_column=0, date_sep='/', date_order=['d','m','Y'], header=False)
    data = move_label_in_front(data, 1)
    save_data_to_file(data, 'seoul_bike_data', is_classification = False, is_regression = True)
    

#---------------------------------------------------------------------------------------------------


def get_south_german_credit():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00573/SouthGermanCredit.zip', 'south_german_credit.zip')
    unrar_raw_data('south_german_credit.zip')
    remove_files(UCIVars.raw_data_folder, 'read_SouthGermanCredit.R')
    remove_files(UCIVars.raw_data_folder, 'codetable.txt')
    remove_files(UCIVars.raw_data_folder, 'south_german_credit.zip')
    data = load_raw_data('SouthGermanCredit.asc', sep = ' ', header = True)
    data = move_label_in_front(data, 20)
    save_data_to_file(data, 'south_german_credit', is_classification = True, is_regression = False)


#---------------------------------------------------------------------------------------------------


def get_shill_bidding():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00562/Shill%20Bidding%20Dataset.csv', 'shill_bidding.data')
    data = load_mixed_raw_data('shill_bidding.data', sep = ',', header = True)

    # Remove Record ID, Auction ID, Bidder ID
    data = remove_columns(data, [0, 1, 2])

    write_mixed_raw_data(UCIVars.raw_data_folder + 'shill_bidding.data', data, sep = ',')

    data = load_raw_data('shill_bidding.data', sep = ',')
    data = move_label_in_front(data, 9)
    save_data_to_file(data, 'shill_bidding', is_classification = True, is_regression = False)


#---------------------------------------------------------------------------------------------------


def get_gas_turbine():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00551/pp_gas_emission.zip', 'gas_turbine.zip')
    unzip_raw_data('gas_turbine.zip')
    remove_files(UCIVars.raw_data_folder, 'gas_turbine.zip')

    concat_files(UCIVars.raw_data_folder + 'gt_201*.csv', UCIVars.raw_data_folder + 'gt.data')
    remove_files(UCIVars.raw_data_folder, 'gt_201*.csv')

    data = load_raw_data('gt.data', sep = ',', header = True) # Will report 4 errors because of headers in the middle of the data

    data_co = remove_columns(data, [10])
    data_co = move_label_in_front(data_co, 9)
    save_data_to_file(data_co, 'gas_turbine_co', is_classification=False, is_regression=True)

    data_nox = remove_columns(data, [9])
    data_nox = move_label_in_front(data_nox, 9)
    save_data_to_file(data_nox, 'gas_turbine_nox', is_classification=False, is_regression=True)


#---------------------------------------------------------------------------------------------------


def get_oral_toxicity():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00508/qsar_oral_toxicity.zip', 'oral_toxicity.zip')
    unzip_raw_data('oral_toxicity.zip')
    remove_files(UCIVars.raw_data_folder, 'oral_toxicity.zip')

    data = load_mixed_raw_data('qsar_oral_toxicity.csv', sep = ';', header = False)
    categories = ['negative', 'positive']
    data = replace_bin_cats_in_mixed_data(data, categories, column = 1024, separator = ';')
    
    write_mixed_raw_data(UCIVars.raw_data_folder + 'oral_toxicity.data', data, sep = ',')
    remove_files(UCIVars.raw_data_folder, 'qsar_oral_toxicity.csv')

    data = load_raw_data('oral_toxicity.data', sep = ',', header = False)
    data = move_label_in_front(data, 1024)
    save_data_to_file(data, 'oral_toxicity', is_classification = True, is_regression = False)


#---------------------------------------------------------------------------------------------------


def get_wave_energy():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00494/WECs_DataSet.zip', 'wave_energy.zip')
    unzip_raw_data('wave_energy.zip')
    remove_files(UCIVars.raw_data_folder, 'wave_energy.zip')

    # For each of the 4 data sets, the last column contains the sum of columns 32 to 47.
    # I assume the last column is the label and columns 32 to 47 are intermediate results
    # and that only the first 32 colums should be used as features.
    indices = range(32, 48)

    data_adelaide = load_raw_data('WECs_DataSet/Adelaide_Data.csv', sep=',')
    data_adelaide = remove_columns(data_adelaide, indices)
    data_adelaide = move_label_in_front(data_adelaide, 32)
    save_data_to_file(data_adelaide, 'wave_energy_adelaide', is_classification=False, is_regression=True)

    
    data_perth = load_raw_data('WECs_DataSet/Perth_Data.csv', sep=',')
    data_perth = remove_columns(data_perth, indices)
    data_perth = move_label_in_front(data_perth, 32)
    save_data_to_file(data_perth, 'wave_energy_perth', is_classification=False, is_regression=True)
    
    data_sydney = load_raw_data('WECs_DataSet/Sydney_Data.csv', sep=',')
    data_sydney = remove_columns(data_sydney, indices)
    data_sydney = move_label_in_front(data_sydney, 32)
    save_data_to_file(data_sydney, 'wave_energy_sydney', is_classification=False, is_regression=True)
    
    data_tasmania = load_raw_data('WECs_DataSet/Tasmania_Data.csv', sep=',')
    data_tasmania = remove_columns(data_tasmania, indices)
    data_tasmania = move_label_in_front(data_tasmania, 32)
    save_data_to_file(data_tasmania, 'wave_energy_tasmania', is_classification=False, is_regression=True)


#---------------------------------------------------------------------------------------------------


def get_firewall():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00542/log2.csv', 'firewall.data')

    data = load_mixed_raw_data('firewall.data', sep = ',', header = True)
    categories = ['allow', 'drop', 'deny', 'reset-both']
    data = replace_ordinals_in_mixed_data(data, categories, column = 4, separator = ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'firewall.data', data, sep = ',')

    data = load_raw_data('firewall.data', sep = ',', header = False)
    data = move_label_in_front(data, 4)
    save_data_to_file(data, 'firewall', is_classification = True, is_regression = False)


#---------------------------------------------------------------------------------------------------


def get_real_estate_value():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx', 'real_estate_value.xlsx')
    excel_data = pandas.read_excel(UCIVars.raw_data_folder + 'real_estate_value.xlsx', engine = 'openpyxl')
    excel_data.to_csv(UCIVars.raw_data_folder + 'real_estate_value.data', index = False)
    remove_files(UCIVars.raw_data_folder, 'real_estate_value.xlsx')

    data = load_raw_data('real_estate_value.data', sep = ',', header = True)
    data = remove_columns(data, [0])
    data = move_label_in_front(data, 6)
    save_data_to_file(data, 'real_estate_value', is_classification = False, is_regression = True)


#---------------------------------------------------------------------------------------------------


def get_crop_mapping():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00525/data.zip', 'crop_mapping.zip')
    unzip_raw_data('crop_mapping.zip')
    remove_files(UCIVars.raw_data_folder, 'crop_mapping.zip')
    data = load_raw_data('WinnipegDataset.txt', sep=',', header=True)
    save_data_to_file(data, 'crop_mapping', is_classification=True, is_regression = False)


#---------------------------------------------------------------------------------------------------


def get_bitcoin_heist():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00526/data.zip', 'bitcoin_heist.zip')
    unzip_raw_data('bitcoin_heist.zip')
    remove_files(UCIVars.raw_data_folder, 'bitcoin_heist.zip')

    data = load_mixed_raw_data('BitcoinHeistData.csv', sep = ',', header = True)
    data = remove_columns(data, [0])

    # The labels consist of 28 ransomware types and the labe 'white' for not ransomware.
    # We merge every ransomware type into one class. The resulting data set still only has 1.4% positive labels.
    categories = sorted(get_categories_in_mixed_data(data, 8))
    new_cats = [1]*(len(categories)-1) + [2]
    data = replace_manual_in_mixed_data(data, categories, 8, new_cats, ',')
    write_mixed_raw_data(UCIVars.raw_data_folder + 'bitcoin_heist.data', data, sep = ',')
    remove_files(UCIVars.raw_data_folder, 'BitcoinHeistData.csv')

    data = load_raw_data('bitcoin_heist.data', sep = ',', header = False)
    data = move_label_in_front(data, 8)
    save_data_to_file(data, 'bitcoin_heist', is_classification = True, is_regression = False)
    

#---------------------------------------------------------------------------------------------------


def get_query_analytics():

    prepare_new_data_set_group_id()

    download_and_save('https://archive.ics.uci.edu/ml/machine-learning-databases/00493/datasets.zip', 'query_analytics.zip')
    unzip_raw_data('query_analytics.zip')
    remove_files(UCIVars.raw_data_folder, 'query_analytics.zip')
    remove_files(UCIVars.raw_data_folder + 'Datasets/', 'Radius-Queries.csv')

    data_radius = load_raw_data('Datasets/Radius-Queries-Count.csv', sep = ',', header = False)
    data_radius = move_label_in_front(data_radius, 3)
    save_data_to_file(data_radius, 'radius_query', is_classification = False, is_regression = True)

    data_range = load_raw_data('Datasets/Range-Queries-Aggregates.csv', sep = ',', header = True)
    data_range = remove_columns(data_range, [0])

    data_range_incidents = remove_columns(data_range, [5, 6])
    data_range_incidents = move_label_in_front(data_range_incidents, 4)
    save_data_to_file(data_range_incidents, 'range_query_incidents', is_classification = False, is_regression = True)

    data_range_arrests = remove_columns(data_range, [4, 6])
    data_range_arrests = move_label_in_front(data_range_arrests, 4)
    save_data_to_file(data_range_arrests, 'range_query_arrests', is_classification = False, is_regression = True)

    data_range_beat = remove_columns(data_range, [4, 5])
    data_range_beat = move_label_in_front(data_range_beat, 4)
    save_data_to_file(data_range_beat, 'range_query_beat', is_classification = False, is_regression = True)
    

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def download_all_uci(paths: Paths):
    # preparation
    # code was written with global variables, so we set the global variable values here for the paths
    base_folder = str(paths.uci_download())
    #global data_folder
    #global UCIVars.raw_data_folder
    #global regression_data_folder
    #global binary_classification_data_folder
    #global multiclass_classification_data_folder
    #global statistics_filename

    UCIVars.data_folder = base_folder + '/data/'
    UCIVars.raw_data_folder = base_folder + '/raw_data/'
    UCIVars.regression_data_folder = base_folder + '/regression-data/'
    UCIVars.binary_classification_data_folder = base_folder + '/bin-class-data/'
    UCIVars.multiclass_classification_data_folder = base_folder + '/multi-class-data/'
    UCIVars.statistics_filename = base_folder + '/data_statistics.csv'

    utils.ensureDir(UCIVars.data_folder)
    utils.ensureDir(UCIVars.raw_data_folder)
    utils.ensureDir(UCIVars.regression_data_folder)
    utils.ensureDir(UCIVars.binary_classification_data_folder)
    utils.ensureDir(UCIVars.multiclass_classification_data_folder)

    # this was also a global statement
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context

    #if os.path.exists(statistics_filename):
        #os.remove(statistics_filename)

    #---------------------------------------------------------------------------------------------------

    # Data sets that are (primarily) of regression type

    #---------------------------------------------------------------------------------------------------

    get_skill_craft()
    get_cargo_2000()
    get_KDC_4007()
    get_sml2010()
    get_wine_quality()
    get_parkinson()
    get_insurance_benchmark()
    get_air_quality()
    get_EEG_steady_state()
    get_cycle_power_plant()
    get_carbon_nanotubes()
    get_naval_propulsion()
    get_blood_pressure()
    get_gas_sensor_drift()
    get_bike_sharing()
    get_appliances_energy()
    get_indoor_loc()
    get_online_news_popularity()
    get_facebook_comment_volume()
    get_bejing_pm25()
    get_protein_tertiary_structure()
    get_five_cities_pm25()
    get_tamilnadu_electricity()

    # Additional data sets added after mid 2018

    get_metro_interstate_traffic_volume()
    get_facebook_live_sellers_thailand()
    get_parking_birmingham()
    get_tarvel_review_ratings()
    get_superconductivity()
    get_gnfuv_unmanned_surface_vehicles()


    # Additional data sets added February 2021

    #get_seoul_bike_data()
    #get_gas_turbine()
    #get_wave_energy()
    #get_real_estate_value()
    #get_query_analytics()


    #---------------------------------------------------------------------------------------------------

    # Data sets that are (primarily) of classification type

    #---------------------------------------------------------------------------------------------------


    get_phishing()
    get_ozone_level()
    get_opportunity_activity()
    get_australian_sign_language()
    get_seismic_bumps()
    get_meu_mobile_ksd()
    get_character_trajectories()
    get_vicon_physical_action()
    get_simulated_falls()
    get_chess()
    get_abalone()
    get_madelon()
    get_spambase()
    get_wilt()
    get_waveform()
    get_wall_following_robot()
    get_page_blocks()
    get_optical_recognition_handwritten_digits()
    get_bach_chorals_harmony()
    get_smartphone_human_activity()
    get_turkiye_student_evaluation()
    get_artificial_characters()
    get_first_order_theorem_proving()
    get_landsat_satimage()
    get_hiv_1_protease()
    get_musk()
    get_ble_rssi_indoor_location()
    get_anuran_calls()
    get_thyroids()
    get_isolet()
    get_mushroom()
    get_assamese_characters()
    get_arabic_digit()
    get_eeg_steady_state_visual()
    get_gesture_phase_segmentation()
    get_emg_physical_action()
    get_human_activity_smartphone()
    get_polish_companies_bankruptcy()
    get_crowd_sourced_mapping()
    get_firm_teacher_clave()
    get_smartphone_human_activity_postural()
    get_pen_recognition_handwritten_characters()
    get_epileptic_seizure_recognition()
    get_nursery()
    get_indoor_user_movement_prediction()
    get_eeg_eye_state()
    get_htru2()
    get_magic_gamma_telescope()
    get_letter_recognition()
    get_occupancy_detection()
    get_avila()
    get_grammatical_facial_expressions()
    get_chess_krvk()
    get_default_credit_card()
    get_nomao()
    get_indoor_loc_mag()
    get_activity_recognition()
    get_bank_marketing()
    get_census_income()


    ## Additional data sets added after mid 2018


    get_emg_for_gestures()
    get_indoor_channel_measurements()
    get_electrical_grid_stability_simulated()
    get_online_shoppers_attention()
    get_pmu_ud()


    # Additional data sets added February 2021


    #get_south_german_credit()
    #get_shill_bidding()
    #get_oral_toxicity()
    #get_firewall()
    #get_crop_mapping()
    #get_bitcoin_heist()

