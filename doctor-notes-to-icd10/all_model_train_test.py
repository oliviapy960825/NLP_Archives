# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:37:53 2020

@author: Peiyu Wang
"""

from model_set_up import *
from all_model import *
from model_test import *
training_df,testing_df=get_all_training_testing(doctor_notes_exp,class_list,training_set_num_by_class,diabetes_df,diabetes_list)
training_df=get_all_resampled_training(training_df,class_list,diabetes_list)
t,vocab_size=tokenizing(training_df,max_words)
embedding_matrix=initiate_embedding_layer(t,vocab_size,hier_word2vec_model)
X_train,X_test=text_preprocessing(training_df,testing_df,t,max_length)
Y_train,Y_test=non_diabetes_label_preprocessing(training_df,testing_df,num_classes,encoder)

embedding_dim=hier_word2vec_model.wv.vectors.shape[1]
batch_size = 64
epochs = 5
model=get_compiled_model_all(vocab_size, embedding_dim, embedding_matrix,max_length,num_classes)
print(model.summary())
history = model.fit(X_train, Y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_split=0.1)

metric_name,score=test_model(model,X_test,Y_test)
print(metric_name)
print(score)
