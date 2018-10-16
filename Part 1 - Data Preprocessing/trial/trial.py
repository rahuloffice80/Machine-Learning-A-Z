# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 22:15:55 2018

@author: I322919
"""
import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

