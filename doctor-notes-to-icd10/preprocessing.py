# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:10:31 2020

@author: Peiyu Wang
"""

from read_data import *

"""
1.Transform all the doctor notes into all lower cases
2. Extract the first three digits out of the icd-10 code associated with the note
3. Shuffle the dataframe
4. Keep only the doctor note column, the icd-10 code column and the first 3 digits of the icd-10 code column
5. Rename the columns as "text", "label" and "icd_10_code_first_digits"
"""    
def process_note(exp,icd_code_type):
    exp.note_text=exp.note_text.str.lower()
    exp['icd_code_first_digits']=exp[icd_code_type].str[:3]
    doctor_notes_exp=exp[['note_text',icd_code_type,'icd_code_first_digits']]
    doctor_notes_exp.rename(columns={'note_text':'text',icd_code_type:'label','icd_code_first_digits':'icd_code_first_digits'},inplace=True)
    doctor_notes_exp = shuffle(doctor_notes_exp,random_state=4)
    return doctor_notes_exp

#pd.set_option('display.max_colwidth', -1)
"""
Group the exp dataframe by the first 3 digits of their icd-10 codes, and store the result counts into a
variable called counts
Keep only the classes/icd-10 codes that have at least 10 samples
use 70% of the counts as the training data and the rest 30% as the testing data, record the number of samples
that should be in the training set by class and store the classes and numbers in dataframe called
training_set_num_by_class
"""
def get_resample_info(doctor_notes_exp):
    counts=doctor_notes_exp.groupby('icd_code_first_digits').text.count()
    counts=counts[counts>=300]
    counts_for_upsampling=counts[counts<=1550]
    counts_for_downsampling=counts[counts>1550]
    return counts


def get_training_num_by_class(doctor_notes_exp,diabetes_list,counts):
    training_set_num_by_class=(0.7*counts).astype(int)
    class_list=training_set_num_by_class.keys().tolist()
    for code in diabetes_list:
        if code in class_list:
            class_list.remove(code)
    return training_set_num_by_class,class_list

"""
get codes associated with diabetes
"""
def get_diabetes_code(doctor_notes_exp):
    diabetes_df=doctor_notes_exp[doctor_notes_exp.icd_code_first_digits.between('E08','E13',inclusive=True)]
    return diabetes_df