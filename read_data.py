# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:07:47 2020

@author: Peiyu Wang
"""

from packages import *
"""
Querying doctor notes from the database
"""
def query_data(icd_code_type):
    q="""
    Insert ORACLE Query here
    """.format(icd_code_type,icd_code_type)
    exp=db.on().read(q)
    return exp

#print(query_data('CURRENT_ICD10_LIST'))
"""
Read the icd-10 code and titles file into a variable named icd_10_data
fit the label encode on all the icd_10 code in icd_10_data
later on if new icd_codes come out and is in the same format as the current icd_10_code data, then can
directly input the new data file directory
"""
def read_icd_10_data_file(file_dir):
    icd_data = pd.read_fwf(file_dir,header=None, names=["icd_code", "icd_title"])
    icd_data['icd_code_first_digits']=icd_data['icd_code'].str[:3]
    num_classes=len(icd_data['icd_code_first_digits'].unique())
    encoder = LabelEncoder()
    encoder.fit(icd_data.icd_code_first_digits)
    return encoder,num_classes

