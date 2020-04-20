# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:13:21 2020

@author: Peiyu Wang
"""
from read_data import *
from preprocessing import *
stop = stopwords.words('english')
def get_all_training_testing(doctor_notes_exp,class_list,training_set_num_by_class,diabetes_df,diabetes_list):
    training_df=pd.DataFrame()
    testing_df=pd.DataFrame()
    for icd_code in class_list:
        print(icd_code)
        training_set_num=training_set_num_by_class[icd_code]
        print(training_set_num)
        temp_df=doctor_notes_exp[doctor_notes_exp['icd_code_first_digits']==icd_code]
        #print(temp_df)
        training_df=training_df.append(temp_df[:training_set_num])
        testing_df=testing_df.append(temp_df[training_set_num:])
        print(training_df)
    for icd_code in diabetes_list:
        print(icd_code)
        training_set_num=int(len(diabetes_df[diabetes_df.icd_code_first_digits==icd_code])*0.7)
        print(training_set_num)
        temp_df=doctor_notes_exp[doctor_notes_exp.icd_code_first_digits==icd_code]
        print(temp_df)
        training_df=training_df.append(temp_df[:training_set_num])
        testing_df=testing_df.append(temp_df[training_set_num:])
        print(training_df)
    training_df= training_df[training_df.icd_code_first_digits != 'ADM']
    testing_df= testing_df[testing_df.icd_code_first_digits != 'ADM']
    training_df['text']=training_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    testing_df['text']=testing_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    training_df['text']=training_df['text'].str.replace('\d+', '')
    testing_df['text']=testing_df['text'].str.replace('\d+', '')
    return training_df, testing_df

def get_all_resampled_training(training_df,class_list,diabetes_list):
    all_icd_codes=class_list+diabetes_list
    for icd_code in all_icd_codes:
        print(icd_code)
        resampling_df=training_df[training_df.icd_code_first_digits==icd_code]
        if len(resampling_df)>0:
            other_df=training_df[training_df.icd_code_first_digits!=icd_code]
            resampled_df = resample(resampling_df, 
                                replace=True,     # sample with replacement
                                n_samples=1550,    # to match majority class
                                random_state=4)
            training_df=pd.concat([other_df, resampled_df])
    training_df=shuffle(training_df,random_state=4)
    return training_df

def tokenizing(training_df,max_words):
    X_train = training_df['text']
    t= Tokenizer(num_words=max_words, char_level=False)
    t.fit_on_texts(X_train)
    vocab_size = len(t.word_index) + 1
    print(vocab_size)
    return t,vocab_size

def initiate_embedding_layer(tokenizer,vocab_size,word_embedding):
    vocab_size=int(vocab_size)
    print(vocab_size)
    embedding_matrix = zeros((vocab_size, 50))
    for word, i in tokenizer.word_index.items():
        if word in word_embedding:
            embedding_matrix[i] = word_embedding[word]
    return embedding_matrix

def text_preprocessing(training_df,testing_df,tokenizer,max_length):
    X_train = training_df['text']
    X_test=testing_df['text']
    X_train = tokenizer.texts_to_sequences(X_train)
    X_train  = pad_sequences(X_train , maxlen=max_length, padding='post')
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test , maxlen=max_length, padding='post')
    return X_train, X_test


def non_diabetes_label_preprocessing(training_df,testing_df,num_classes,encoder):
    Y_train=training_df['icd_code_first_digits']
    Y_test=testing_df['icd_code_first_digits']
    Y_train = encoder.transform(Y_train)
    Y_test = encoder.transform(Y_test)
    Y_train= utils.to_categorical(Y_train, num_classes)
    Y_test= utils.to_categorical(Y_test, num_classes)
    Y_train=np.array(Y_train)
    Y_test=np.array(Y_test)
    return Y_train, Y_test
