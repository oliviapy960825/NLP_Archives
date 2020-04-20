# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:02:13 2020

@author: Peiyu Wang
"""

"""
import packages
"""
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from numpy import zeros
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from gensim.test.utils import common_texts
from sklearn.utils import shuffle
import random
from scipy import linalg, mat, dot
import scipy.spatial.distance as distance
from sklearn.utils import shuffle
import ast
from tensorflow.keras import utils
from sklearn.model_selection import KFold
import pickle
from sklearn.metrics import roc_auc_score

