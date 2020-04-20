# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:20:15 2020

@author: Peiyu Wang
"""
from read_data import *
from preprocessing import *
from all_train_test import *
from all_model import *
"""
query data from database. Set the ICD_code type as CURRENT_ICD10_LIST, later on when the new icd_10_codes
come out, given that they are still in the same table as the current_icd10_list, change the input parameter
could get the new icd_code out
"""

max_words=1000
max_length=20
icd_code_type='current_icd10_list'
exp=query_data(icd_code_type)
file_dir='icd10cm_codes_2020.txt'
encoder,num_classes= read_icd_10_data_file(file_dir)
"""
this block load the pre-trained word2vec model into variable named hier_word2vec_model and load the query
results into variable named exp
"""
hier_word2vec_model = Word2Vec.load("hier_word2vec_model")
print(hier_word2vec_model.wv.vectors.shape)
doctor_notes_exp=process_note(exp,icd_code_type)
print(doctor_notes_exp)
stop = stopwords.words('english')
diabetes_df=get_diabetes_code(doctor_notes_exp)
diabetes_list=diabetes_df.icd_code_first_digits.unique().tolist()
counts=get_resample_info(doctor_notes_exp)
training_set_num_by_class,class_list=get_training_num_by_class(doctor_notes_exp,diabetes_list,counts)
#class_list
