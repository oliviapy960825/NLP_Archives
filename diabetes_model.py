# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:15:46 2020

@author: Peiyu Wang
"""
from read_data import *
from preprocessing import *
from diabetes_train_test import *
"""
Model structure
"""
def get_compiled_model_diabetes(vocab_size, embedding_dim, embedding_matrix,max_length,num_classes=1):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              weights=[embedding_matrix], input_length=max_length, trainable=True),
        tf.keras.layers.ReLU(),
        tf.keras.layers.RNN(tf.keras.layers.GRUCell(300)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes,activation='sigmoid'),
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.TruePositives(name='tp'),
                          tf.keras.metrics.Accuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      #f1
]
                                 )
    return model