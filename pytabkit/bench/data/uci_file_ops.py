


import os as os
import re as re
import csv as csv
import math as math
from pathlib import Path

import pandas as pandas
import numpy as numpy
import os.path as path
import glob as glob
import shutil as shutil
import zipfile as zipfile
from scipy.io import arff
import patoolib as patoolib
import sklearn.preprocessing as preprocessing
import sklearn.datasets as datasets



import urllib.request as urllib2
import time
import datetime
import codecs
import platform
import tarfile
import gzip
import ssl

from collections import Counter


class UCIVars:
    # formerly global variables, will be re-set by get_uci.download_all_uci()
    data_folder = '../data/'
    raw_data_folder = '../raw-data/'
    regression_data_folder = '../regression-data/'
    binary_classification_data_folder = '../bin-class-data/'
    multiclass_classification_data_folder = '../multi-class-data/'
    statistics_filename = "../data_statistics.csv"

    data_group_id = 0


# if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
#     ssl._create_default_https_context = ssl._create_unverified_context


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------

def prepare_new_data_set_group_id():
    print("\n==================================================================")
    UCIVars.data_group_id = UCIVars.data_group_id + 1
    

#---------------------------------------------------------------------------------------------------



def make_folder(folder):
    if (os.path.exists(folder) == False):
        os.mkdir(folder)
    



#---------------------------------------------------------------------------------------------------



def download_and_save(url, filename):
    
    data_link = urllib2.urlopen(url)
    print('Downloading: ' + filename) 
    
    if os.path.exists(UCIVars.raw_data_folder + filename):
        os.remove(UCIVars.raw_data_folder + filename)
    
    with open(UCIVars.raw_data_folder + filename, 'wb') as output:
        output.write(data_link.read())
    
    
#---------------------------------------------------------------------------------------------------


def unzip_raw_data(filename):
    
    zip_ref = zipfile.ZipFile(UCIVars.raw_data_folder + filename, 'r')
    zip_ref.extractall(UCIVars.raw_data_folder)
    zip_ref.close()


#---------------------------------------------------------------------------------------------------


def unrar_raw_data(filename):
    
    full_filename = UCIVars.raw_data_folder + filename 
    patoolib.extract_archive(full_filename, outdir = UCIVars.raw_data_folder)

    
#---------------------------------------------------------------------------------------------------


def my_decode(x):
    
    if isinstance(x, bytes):
        return x.decode('utf-8')
    else:
        return str(x)


#---------------------------------------------------------------------------------------------------


def unarff_raw_data(filename):
    
    data = arff.loadarff(UCIVars.raw_data_folder + filename + '.arff')[0]
    target_filename = UCIVars.raw_data_folder + filename + '.data'
        
    data_cleaned = []
    for row in data:
        data_cleaned.append([my_decode(entry) for entry in row])

    
    with open(target_filename, "w") as target_file:
        writer = csv.writer(target_file, lineterminator = '\n')
        writer.writerows(data_cleaned)


#---------------------------------------------------------------------------------------------------


def un_z_raw_data(filename):
    
    if platform.system() == "Linux":
        os.system('uncompress -f ' + UCIVars.raw_data_folder + filename)
        return True
    else:
        print("Could not decompress .Z file, since this requires Linux.")
        return False
    
    
    
#---------------------------------------------------------------------------------------------------


def untar_raw_data(filename):
    
    full_filename = UCIVars.raw_data_folder + filename
    tar = tarfile.open(full_filename)
    tar.extractall(UCIVars.raw_data_folder)
    tar.close()
    
    
#---------------------------------------------------------------------------------------------------


def ungz_raw_data(filename):
    
    full_filename = UCIVars.raw_data_folder + filename
    target_filename = UCIVars.raw_data_folder + filename + '.data'
    
    target_file = open(target_filename, "w")
    with gzip.open(full_filename, 'rt') as source_file:
        data = source_file.read()

    target_file.write(data)
    target_file.close()
    
    
    
#---------------------------------------------------------------------------------------------------


def replace_chars_in_file(filename, old_char, new_char):
    
    fr = codecs.open(UCIVars.raw_data_folder + filename, encoding = 'utf-8')
    content = fr.read()
    fr.close()

    newcontent = content.replace(old_char, new_char)
    
    fw = codecs.open(UCIVars.raw_data_folder + filename, 'w', encoding = 'utf-8')
    fw.write(newcontent)
    fw.close()
   
   
#---------------------------------------------------------------------------------------------------
    
    
    
def get_category_replace_string(category_size, position, separator):
    
    string = ''
    for i in range(position):
        string = string + '0' + separator
        
    string = string + '1' + separator    
    
    
    for i in range(position + 1, category_size):
        string = string + '0' + separator
    
    string = string[0:len(string) - len(separator)]
    
    return string



        
#---------------------------------------------------------------------------------------------------

    
    
def replace_categories_in_file(filename, categories, separator):
    
    for i in range(len(categories)):
        replace_chars_in_file(filename, categories[i], get_category_replace_string(len(categories), i, separator))
        
        

#---------------------------------------------------------------------------------------------------

def convert_replace_string_to_vector(string, separator):
    
    string_vector = string.split(separator)
    
    return list(numpy.float_(string_vector))
    



#---------------------------------------------------------------------------------------------------

    
    
def get_categories_in_mixed_data(data, column):

    rows = numpy.shape(data)[0]
    categories = list(set(data[0:rows, column]))
    
    return categories

    
#---------------------------------------------------------------------------------------------------

    
    
def auto_replace_categories_in_mixed_data(data, column, separator, unknown_string = '', unknown_replacement_value = 0):    
    
    categories = get_categories_in_mixed_data(data, column)
    if numpy.shape(categories)[0] == 2:
        new_data = replace_bin_cats_in_mixed_data(data, categories, column, separator, unknown_string = unknown_string, unknown_replacement_value = unknown_replacement_value)
    else:
        new_data = replace_categories_in_mixed_data(data, categories, column, separator, unknown_string = unknown_string, unknown_replacement_value = unknown_replacement_value)
    
    return new_data



#---------------------------------------------------------------------------------------------------

    
    
def auto_replace_missing_in_mixed_data(data, unknown_string = '?'):    
    
    rows = numpy.shape(data)[0]
    dim = numpy.shape(data)[1]    
    
    columns = range(dim)
    for i in range(len(columns)):
        count_entries = Counter(data[0:rows, columns[i]])
        weighted_sum = 0.0
        entries_sum = 0.0
        for key in count_entries:
            if key != unknown_string:
                weighted_sum = weighted_sum + float(key) * count_entries[key]
                entries_sum = entries_sum + count_entries[key]
        average = weighted_sum / float(entries_sum)    
        data = replace_categories_in_mixed_data(data, [], columns[i], ',', unknown_string = '?', unknown_replacement_value = average)
        
        
    return data
    
    
#---------------------------------------------------------------------------------------------------

    
    
def replace_categories_in_mixed_data(data, categories, column, separator, unknown_string = '', unknown_replacement_value = 0):
    
    rows = numpy.shape(data)[0]
    cols = numpy.shape(data)[1]
    
    
    empty_string_length = len(categories) * max(1, len(str(unknown_replacement_value))) + (len(categories) - 1) * len(separator)
    empty_string = ' ' * empty_string_length
    new_column = [empty_string] * rows
    new_column = data[0:rows, column]
    
    for i in range(len(categories)):
        replacement = get_category_replace_string(len(categories), i, separator)
        new_column = [replacement if word == categories[i] else word for word in new_column]
        
    if unknown_string != '':
        replacement = str(unknown_replacement_value)
        for i in range(len(categories) - 1):
            replacement = replacement + separator + str(unknown_replacement_value)
        new_column = [replacement if word == unknown_string else word for word in new_column]
        
    new_column = numpy.reshape(new_column, newshape = (rows, 1))
    new_data = numpy.concatenate((data[0:rows, 0:column], new_column, data[0:rows, column + 1:cols]), axis = 1)    

    return new_data


#---------------------------------------------------------------------------------------------------

    
    
def replace_bin_cats_in_mixed_data(data, categories, column, separator, unknown_string = '', unknown_replacement_value = 0):

    rows = numpy.shape(data)[0]
    cols = numpy.shape(data)[1]
    
    empty_string_length = max(2, len(str(unknown_replacement_value)))
    empty_string = ' ' * empty_string_length
    new_column = [empty_string] * rows
    new_column = data[0:rows, column]

    if unknown_string != '':
        replacement = str(unknown_replacement_value)
        new_column = [replacement if word == unknown_string else word for word in new_column]
        
    for i in range(len(categories)):
        replacement = str(2 * i - 1)
        new_column = [replacement if word == categories[i] else word for word in new_column]

    new_column = numpy.reshape(new_column, newshape = (rows, 1))
    new_data = numpy.concatenate((data[0:rows, 0:column], new_column, data[0:rows, column + 1:cols]), axis = 1)    
        
    return new_data



#---------------------------------------------------------------------------------------------------

    
    
def replace_ordinals_in_mixed_data(data, categories, column, separator, unknown_string = '', unknown_replacement_value = 0, begin_value = 1):

    rows = numpy.shape(data)[0]
    cols = numpy.shape(data)[1]
    
    empty_string_length = max(len(str(unknown_replacement_value)), len(str(len(categories) + 1)))
    empty_string = ' ' * empty_string_length
    new_column = [empty_string] * rows
    new_column = data[0:rows, column]

    for i in range(len(categories)):
        replacement = str(i + begin_value)
        new_column = [replacement if word == categories[i] else word for word in new_column]
        
    if unknown_string != '':
        replacement = str(unknown_replacement_value)
        new_column = [replacement if word == unknown_string else word for word in new_column]

    new_column = numpy.reshape(new_column, newshape = (rows, 1))
    new_data = numpy.concatenate((data[0:rows, 0:column], new_column, data[0:rows, column + 1:cols]), axis = 1)    
        
    return new_data


#---------------------------------------------------------------------------------------------------

    
    
def replace_manual_in_mixed_data(data, categories, column, replacement, separator, unknown_string = '', unknown_replacement_value = 0):

    rows = numpy.shape(data)[0]
    cols = numpy.shape(data)[1]
    
    empty_string_length = max(len(str(unknown_replacement_value)), len(str(len(categories) + 1)))
    empty_string = ' ' * empty_string_length
    new_column = [empty_string] * rows
    new_column = data[0:rows, column]

    for i in range(len(categories)):
        new_column = [str(replacement[i]) if word == categories[i] else word for word in new_column]
        
    if unknown_string != '':
        replacement_tmp = str(unknown_replacement_value)
        new_column = [replacement_tmp if word == unknown_string else word for word in new_column]

    new_column = numpy.reshape(new_column, newshape = (rows, 1))
    new_data = numpy.concatenate((data[0:rows, 0:column], new_column, data[0:rows, column + 1:cols]), axis = 1)    
        
    return new_data


#---------------------------------------------------------------------------------------------------

    
    
def replace_circulars_in_mixed_data(data, categories, column, separator, unknown_string = ''):

    rows = numpy.shape(data)[0]
    cols = numpy.shape(data)[1]
    decimals = 5
    
    
    empty_string_length = 2 * (decimals + 3) + len(separator)
    empty_string = ' ' * empty_string_length
    new_column = [empty_string] * rows
    new_column = data[0:rows, column]

    for i in range(len(categories)):
        radians = float(i) * 2.0 * math.pi / float(len(categories))
        replacement = str(round(math.cos(radians), decimals)) + separator + str(round(math.sin(radians), decimals))
        new_column = [replacement if word == categories[i] else word for word in new_column]
        
    if unknown_string != '':
        replacement = str(0.0) + separator + str(0.0)
        new_column = [replacement if word == unknown_string else word for word in new_column]

    new_column = numpy.reshape(new_column, newshape = (rows, 1))
    new_data = numpy.concatenate((data[0:rows, 0:column], new_column, data[0:rows, column + 1:cols]), axis = 1)    
        
    return new_data




#---------------------------------------------------------------------------------------------------

    
    
def replace_isodate_by_day_in_mixed_data(data, column):

    rows = numpy.shape(data)[0]
    cols = numpy.shape(data)[1]

    old_column = [numpy.datetime64(date) for date in data[0:rows, column]]
    new_column = [str(date.astype(datetime.datetime).isoweekday()) for date in old_column]
    
    new_column = numpy.reshape(new_column, newshape = (rows, 1))
    new_data = numpy.concatenate((data[0:rows, 0:column], new_column, data[0:rows, column + 1:cols]), axis = 1)
    
        
    return new_data


#---------------------------------------------------------------------------------------------------

    
    
def replace_time_by_seconds_in_mixed_data(data, column, sep, rounded = 1):

    rows = numpy.shape(data)[0]
    cols = numpy.shape(data)[1]

    new_column = [str(int(round(float(convert_time_to_seconds(time, sep)) / float(rounded))) * rounded) for time in data[0:rows, column]]
    
    new_column = numpy.reshape(new_column, newshape = (rows, 1))
    new_data = numpy.concatenate((data[0:rows, 0:column], new_column, data[0:rows, column + 1:cols]), axis = 1)
    
        
    return new_data


#---------------------------------------------------------------------------------------------------


def remove_files(folder, filename_pattern):
    
    filenames = glob.glob(folder + filename_pattern)

    for name in filenames:
        os.remove(name)

#---------------------------------------------------------------------------------------------------


def concat_files(source_filename_pattern, target_filename):
    
    filenames = glob.glob(source_filename_pattern)
    
    if os.path.exists(target_filename):
        os.remove(target_filename)
    
    with open(target_filename,'wb') as target_file:
        for name in filenames:
            with open(name,'rb') as source_file:
                shutil.copyfileobj(source_file, target_file, 1024*1024*10)
                
#---------------------------------------------------------------------------------------------------


def load_mixed_raw_data(filename, sep, header = False):
    
    # Some Python versions issue a warning if 'encoding' is not set, while other versions do not know 'encoding'
    # Pick the one you prefer ...
    
    #data = numpy.genfromtxt(UCIVars.raw_data_folder + filename, dtype = None, delimiter = sep)
    data = numpy.genfromtxt(UCIVars.raw_data_folder + filename, dtype = str, delimiter = sep, encoding = None)
    if (header == True):
        data = numpy.delete(data, 0, 0)

    if len(numpy.shape(data)) == 1:
        dim = len(data[0])
        rows = numpy.shape(data)[0]
        new_data = [None] * (dim * rows)
        new_data = numpy.reshape(new_data, newshape = (rows, dim))
        
        for i in range(rows):
            new_data[i] = map(str, data[i]) 
            
        data = new_data

    return data


#---------------------------------------------------------------------------------------------------


def write_mixed_raw_data(filename, data, sep):
    
    with open(filename, mode = 'w') as write_file:
        writer = csv.writer(write_file, delimiter = sep, quotechar = '', quoting = csv.QUOTE_NONE, escapechar = ' ')
        writer.writerows(data)

    # replace_chars_in_file will add the raw_data_path, so we have to remove it from the filename
    replace_chars_in_file(Path(filename).name, ' ' + sep, sep)

    
#---------------------------------------------------------------------------------------------------
    
    
def load_raw_data(filename, sep, description_columns = 0, date_column = -1, date_sep = '', date_order = '', time_column = -1, time_sep = '', german_decimal = False, na_string = '---', show_intermediate = False, header = False):
    
    fp = open(UCIVars.raw_data_folder + filename, 'r')
    
    number_of_rows = 0
    number_of_lines = 0
    max_number_of_columns = 0

    rows_with_na_string = 0
    rows_with_incorrect_date = 0
    rows_with_incorrect_time = 0
    rows_with_incorrect_number_of_columns = 0
    rows_with_odd_error = 0
    
    is_first_line = True
    
    for row in fp:
        if (is_first_line == True) and (header == True):
            is_first_line = False
        else:
            row = row.strip()
            raw_row = row.split(sep)
            
            number_of_columns = numpy.shape(raw_row)[0]
            max_number_of_columns = max(number_of_columns, max_number_of_columns)
            
            number_of_data_columns = number_of_columns - description_columns
            current_row = numpy.zeros(shape = (1, number_of_data_columns))
            number_of_lines = number_of_lines + 1
            
            if ((number_of_lines % 1000 == 0) and (show_intermediate == True)):
                print("Read %d lines" %number_of_lines)


            correct_row = True
            for c in range(description_columns, number_of_columns):
                if (raw_row[c] == na_string):
                    correct_row = False
                    rows_with_na_string = rows_with_na_string + 1
                elif (c == date_column):
                    date = raw_row[c].split(date_sep)
                    if (len(date) != 3):
                        correct_row = False
                        rows_with_incorrect_date = rows_with_incorrect_date + 1
                    else:
                        date_string = date[0] + '-' + date[1] + '-' + date[2] 
                        date_fmt = '%' + date_order[0] + '-%' + date_order[1] + '-%' + date_order[2]
                        date_result = datetime.datetime.strptime(date_string, date_fmt)
                        date_tuple = date_result.timetuple()     
                        current_row[0, c - description_columns] = float(date_tuple.tm_yday)
                elif (c == time_column):
                    time = raw_row[c].split(time_sep)
                    if (len(time) != 3):
                        correct_row = False
                        rows_with_incorrect_time = rows_with_incorrect_time + 1 
                    else:
                        current_row[0, c - description_columns] = 3600.0 * float(time[0]) + 60.0 * float(time[1]) + float(time[2])
                elif (is_number(raw_row[c], german_decimal) == True):
                    if (german_decimal == False):
                        current_row[0, c - description_columns] = float(raw_row[c])
                    else:
                        current_row[0, c - description_columns] = float(raw_row[c].replace(',', '.', 1))
                elif (raw_row[c] == ''):
                    current_row[0, c - description_columns] = 0.0
                else:
                    correct_row = False
                    rows_with_odd_error = rows_with_odd_error + 1
                
                if (number_of_columns != max_number_of_columns):
                    correct_row = False
                    rows_with_incorrect_number_of_columns = rows_with_incorrect_number_of_columns + 1
                
                if (correct_row == False):
                    break

                
            if (correct_row == True):    
                number_of_rows = number_of_rows + 1
                if (number_of_rows == 1):
                    data = numpy.zeros(shape = (0, number_of_data_columns))
                    data_block = current_row
                else:
                    data_block = numpy.concatenate((data_block, current_row), axis = 0)
                    
                if (number_of_rows == 1000):
                    data = data_block
                    data_block = numpy.zeros(shape = (0, number_of_data_columns))
                elif (number_of_rows % 1000 == 0):
                    data = numpy.concatenate((data, data_block), axis = 0)
                    data_block = numpy.zeros(shape = (0, number_of_data_columns))


    # Make sure the last block is added if this has not just happened
    
    if (number_of_rows % 1000 != 0):
        data = numpy.concatenate((data, data_block), axis = 0)
        
    fp.close()
    
    if (number_of_lines - number_of_rows > 0):
        
        if (number_of_rows > 0):
            print("File %s has %d data columns and %d rows with complete data and %d rows with corrupted data" % (filename, numpy.shape(data)[1], number_of_rows, number_of_lines - number_of_rows))
            
            print("Rows with na string: %d" % rows_with_na_string)
            print("Rows with incorrect date: %d" % rows_with_incorrect_date)
            print("Rows with incorrect time: %d" % rows_with_incorrect_time)
            print("Rows with incorrect number of columns: %d" % rows_with_incorrect_number_of_columns)
            print("Rows with odd error: %d" % rows_with_odd_error)
        else:
            print("Could not read a single row!!!\n")
            quit()
        
    else:
        print("File %s has %d data columns and %d rows" % (filename, numpy.shape(data)[1], number_of_rows))
    
    return data


#---------------------------------------------------------------------------------------------------


def remove_rows_with_label(data, label):
    
    bad_rows = numpy.where(data[:, 0] == label)[0]

    if (len(bad_rows) > 0):
        data = numpy.delete(data, bad_rows, axis = 0)
        print('Removing %d rows with label %1.3f' % (len(bad_rows), label))
        
    return data

#---------------------------------------------------------------------------------------------------


def remove_empty_columns(data):
    
    min_values = numpy.min(data, axis = 0)
    max_values = numpy.max(data, axis = 0)
    value_range = max_values - min_values
    
    empty_columns = numpy.where(value_range == 0.0)[0]
    
    if (len(empty_columns) > 0):
        print('Removing %d empty columns' % len(empty_columns))
        data = remove_columns(data, empty_columns)
    
    return data
    
    
#---------------------------------------------------------------------------------------------------


    
def save_data_to_file(data, filename, is_classification, is_regression = True, min_scale = -1.0, max_scale = 1.0):
    
    data_stats = {}
    data_stats['filename'] = filename
    
    data = remove_empty_columns(data)
    
    number_of_rows = numpy.shape(data)[0]
    number_of_columns = numpy.shape(data)[1]
    
    data_stats['rows'] = number_of_rows
    data_stats['columns'] = number_of_columns - 1
    data_stats['binary columns'] = count_bin_columns(data)
    
    
    print("Writing file %s with dim = %d and %d rows" % (filename, number_of_columns - 1, number_of_rows))
    numpy.savetxt(UCIVars.data_folder + filename + '.csv', data, fmt = '%.8e', delimiter = ',', newline = '\n', header = '', footer = '')
    
    min_values = numpy.min(data, axis = 0)
    max_values = numpy.max(data, axis = 0)
    value_range = max_values - min_values
    
    
        
    for c in range(1, number_of_columns):
        m = (max_scale - min_scale) / value_range[c]
        b = min_scale - m * min_values[c]
        data[:, c] = m * data[:, c] + b

    if (is_classification == False):
        min_scale = -1.0
        max_scale = 1.0
        m = (max_scale - min_scale) / value_range[0]
        b = min_scale - m * min_values[0]
        data[:, 0] = m * data[:, 0] + b

    
    
    if (is_regression == True):
        numpy.savetxt(UCIVars.regression_data_folder + filename + '.csv', data, fmt = '%.8e', delimiter = ',', newline = '\n', header = '', footer = '')
        data_stats['classes'] = 0
        data_stats['naive'] = numpy.var(data[:, 0])
        
        save_data_stats(data_stats)
        
        
    if (is_classification == True):
        all_labels = data[:, 0].astype(int)
        labels, label_counts = numpy.unique(all_labels, return_counts = True)
        data_stats['classes'] = len(labels)
        
        highest_frequency = numpy.max(label_counts)
        data_stats['naive'] = float(number_of_rows - highest_frequency) / float(number_of_rows)
        
        if (len(labels) == 2):
            m = 2.0 / (labels[1] - labels[0])
            b = - (labels[1] + labels[0]) / (labels[1] - labels[0])
            data[:, 0] = numpy.floor(m * data[:, 0] + b + 0.5)
            
            numpy.savetxt(UCIVars.binary_classification_data_folder + filename + '.csv', data, fmt = '%.8e', delimiter = ',', newline = '\n', header = '', footer = '')
            save_data_stats(data_stats)
        else:
            numpy.savetxt(UCIVars.multiclass_classification_data_folder + filename + '.csv', data, fmt = '%.8e', delimiter = ',', newline = '\n', header = '', footer = '')
            save_data_stats(data_stats)
            
            second_highest_frequency = numpy.sort(label_counts)[len(labels) - 2]
            if (highest_frequency != second_highest_frequency):
                label_1 = labels[numpy.nonzero(label_counts == highest_frequency)[0][0]]
                label_2 = labels[numpy.nonzero(label_counts == second_highest_frequency)[0][0]]
            else:
                label_1 = labels[numpy.nonzero(label_counts == highest_frequency)[0][0]]
                label_2 = labels[numpy.nonzero(label_counts == highest_frequency)[0][1]]
              
            data_1 = data[data[:, 0] == label_1]
            data_1[:, 0] = -1.0 
            data_2 = data[data[:, 0] == label_2]
            data_2[:, 0] = 1.0 
            data = numpy.concatenate((data_1, data_2), axis = 0)
            
            data_stats['classes'] = 2
            data_stats['rows'] = highest_frequency + second_highest_frequency
            data_stats['naive'] = float(second_highest_frequency) / float(data_stats['rows'])
            
            if (data_stats['rows'] >= 2500):
                numpy.savetxt(UCIVars.binary_classification_data_folder + filename + '.csv', data, fmt = '%.8e', delimiter = ',', newline = '\n', header = '', footer = '')
                save_data_stats(data_stats)
            
        
    
    
    
#---------------------------------------------------------------------------------------------------

def save_data_stats(data_stats): 

    if os.path.exists(UCIVars.statistics_filename):
        string = ''
    else:
        string = 'Name, Rows, Columns, Binary Columns, Classes, Naive Error, Relative Weight\n'
        

    string = string + data_stats['filename'] + ', ' + str(data_stats['rows']) + ', ' + str(data_stats['columns']) + ', ' + str(data_stats['binary columns']) + ', ' + str(data_stats['classes']) + ', ' + str(data_stats['naive']) + ', ' + str(UCIVars.data_group_id) + '\n'
    with open(UCIVars.statistics_filename, "a") as fp:
        fp.write(string)


    
    
#---------------------------------------------------------------------------------------------------

def is_number(string, german_decimal):
    
    # Idea of this code is taken from
    # https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
    
    if (german_decimal == False):
        string = string.replace('.', '', 1)
    else:
        string = string.replace(',', '', 1)
    string = string.replace('e-', '', 1)
    string = string.replace('e+', '', 1)
    string = string.replace('E-', '', 1)
    string = string.replace('E+', '', 1)
    string = string.replace('-', '', 2)
    string = string.replace('+', '', 1)
    
    return string.isdigit()



#---------------------------------------------------------------------------------------------------


def remove_columns(data, columns):
    
    return numpy.delete(data, columns, axis = 1)


#---------------------------------------------------------------------------------------------------


def move_label_in_front(data, label_column):
    
    number_of_rows = numpy.shape(data)[0]
    
    labels = numpy.reshape(data[:, label_column], newshape = (number_of_rows, 1))
    unlabeled_data = remove_columns(data, [label_column])    
    
    data = numpy.concatenate((labels, unlabeled_data), axis = 1)
    
    return data
    
    
#---------------------------------------------------------------------------------------------------

def count_bin_columns(data):

    cols = numpy.shape(data)[1]
	
    count = 0
    for i in range(1, cols):
        if len(set(data[:, i])) == 2:
            count = count + 1

    return count



#---------------------------------------------------------------------------------------------------


def convert_time_to_seconds(time, sep):

    time_tmp = time.split(sep)
    seconds = 3600 * int(time_tmp[0]) + 60 * int(time_tmp[1]) + int(time_tmp[2])

    return seconds

