# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 20:04:42 2020

@author: Brian Hoepfl
Datathon 2020
Let's predict those divorces!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


def confusion_mat_2_percent_correct(cm):
    return (cm[0][0] + cm[1][1]) / float(np.sum(cm))


#Importing dataset
divorce_data_df = pd.read_csv('D:/Owner/Downloads/divorce/divorce.csv', sep = ';')

#Seperating Data from labels
x_Data = divorce_data_df.drop('Class', axis = 1)
y_Data = divorce_data_df['Class']

print(divorce_data_df.head())

x_Train, x_Test, y_Train, y_Test = train_test_split(x_Data, y_Data, 
                                                    test_size = 0.2,
                                                    random_state = 42)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decision_Tree_Classifier = DecisionTreeClassifier(criterion='entropy', 
                                                  random_state = 42)
decision_Tree_Classifier.fit(x_Train, y_Train)

y_pred_tree = decision_Tree_Classifier.predict(x_Test)

cm_decision_tree = confusion_matrix(y_Test, y_pred_tree)

percent_correct = confusion_mat_2_percent_correct(cm_decision_tree)

print('Percent correct decision tree classifier: ' + str(percent_correct))

#Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
naive_Bayes_Classifier = GaussianNB()
naive_Bayes_Classifier.fit(x_Train, y_Train)
y_Pred_Naive_Bayes = naive_Bayes_Classifier.predict(x_Test)

cm_Naive_Bayes = confusion_matrix(y_Test, y_Pred_Naive_Bayes)

print('Percent correct Naive Bayes classifier: ' + str(confusion_mat_2_percent_correct(cm_Naive_Bayes)))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest_Classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',
                                           random_state = 42)
forest_Classifier.fit(x_Train, y_Train)
y_Pred_Random_Forest = forest_Classifier.predict(x_Test)

print(type(x_Test))

cm_Random_Forest = confusion_matrix(y_Test, y_Pred_Random_Forest)

print('Percent correct Random Forest classifier: ' + str(confusion_mat_2_percent_correct(cm_Random_Forest)))



##Densely Connected Neural Network
#
##Create Feature Columns
#q_1_Answer = tf.feature_column.numeric_column('Atr1')
#q_2_Answer = tf.feature_column.numeric_column('Atr2')
#q_3_Answer = tf.feature_column.numeric_column('Atr3')
#q_4_Answer = tf.feature_column.numeric_column('Atr4')
#q_5_Answer = tf.feature_column.numeric_column('Atr5')
#q_6_Answer = tf.feature_column.numeric_column('Atr6')
#q_7_Answer = tf.feature_column.numeric_column('Atr7')
#q_8_Answer = tf.feature_column.numeric_column('Atr8')
#q_9_Answer = tf.feature_column.numeric_column('Atr9')
#q_10_Answer = tf.feature_column.numeric_column('Atr10')
#q_11_Answer = tf.feature_column.numeric_column('Atr11')
#q_12_Answer = tf.feature_column.numeric_column('Atr12')
#q_13_Answer = tf.feature_column.numeric_column('Atr13')
#q_14_Answer = tf.feature_column.numeric_column('Atr14')
#q_15_Answer = tf.feature_column.numeric_column('Atr15')
#q_16_Answer = tf.feature_column.numeric_column('Atr16')
#q_17_Answer = tf.feature_column.numeric_column('Atr17')
#q_18_Answer = tf.feature_column.numeric_column('Atr18')
#q_19_Answer = tf.feature_column.numeric_column('Atr19')
#q_20_Answer = tf.feature_column.numeric_column('Atr20')
#q_21_Answer = tf.feature_column.numeric_column('Atr21')
#q_22_Answer = tf.feature_column.numeric_column('Atr22')
#q_23_Answer = tf.feature_column.numeric_column('Atr23')
#q_24_Answer = tf.feature_column.numeric_column('Atr24')
#q_25_Answer = tf.feature_column.numeric_column('Atr25')
#q_26_Answer = tf.feature_column.numeric_column('Atr26')
#q_27_Answer = tf.feature_column.numeric_column('Atr27')
#q_28_Answer = tf.feature_column.numeric_column('Atr28')
#q_29_Answer = tf.feature_column.numeric_column('Atr29')
#q_30_Answer = tf.feature_column.numeric_column('Atr30')
#q_31_Answer = tf.feature_column.numeric_column('Atr31')
#q_32_Answer = tf.feature_column.numeric_column('Atr32')
#q_33_Answer = tf.feature_column.numeric_column('Atr33')
#q_34_Answer = tf.feature_column.numeric_column('Atr34')
#q_35_Answer = tf.feature_column.numeric_column('Atr35')
#q_36_Answer = tf.feature_column.numeric_column('Atr36')
#q_37_Answer = tf.feature_column.numeric_column('Atr37')
#q_38_Answer = tf.feature_column.numeric_column('Atr38')
#q_39_Answer = tf.feature_column.numeric_column('Atr39')
#q_40_Answer = tf.feature_column.numeric_column('Atr40')
#q_41_Answer = tf.feature_column.numeric_column('Atr41')
#q_42_Answer = tf.feature_column.numeric_column('Atr42')
#q_43_Answer = tf.feature_column.numeric_column('Atr43')
#q_44_Answer = tf.feature_column.numeric_column('Atr44')
#q_45_Answer = tf.feature_column.numeric_column('Atr45')
#q_46_Answer = tf.feature_column.numeric_column('Atr46')
#q_47_Answer = tf.feature_column.numeric_column('Atr47')
#q_48_Answer = tf.feature_column.numeric_column('Atr48')
#q_49_Answer = tf.feature_column.numeric_column('Atr49')
#q_50_Answer = tf.feature_column.numeric_column('Atr50')
#q_51_Answer = tf.feature_column.numeric_column('Atr51')
#q_52_Answer = tf.feature_column.numeric_column('Atr52')
#q_53_Answer = tf.feature_column.numeric_column('Atr53')
#q_54_Answer = tf.feature_column.numeric_column('Atr54')
#
#
#feature_cols = [q_1_Answer, q_2_Answer, q_3_Answer, q_4_Answer, q_5_Answer,
#                q_6_Answer, q_7_Answer, q_8_Answer, q_9_Answer, q_10_Answer,
#                q_11_Answer, q_12_Answer, q_13_Answer, q_14_Answer, q_15_Answer,
#                q_16_Answer, q_17_Answer, q_18_Answer, q_19_Answer, q_20_Answer,
#                q_21_Answer, q_22_Answer, q_23_Answer, q_24_Answer, q_25_Answer,
#                q_26_Answer, q_27_Answer, q_28_Answer, q_29_Answer, q_30_Answer,
#                q_31_Answer, q_32_Answer, q_33_Answer, q_34_Answer, q_35_Answer,
#                q_36_Answer, q_37_Answer, q_38_Answer, q_39_Answer, q_40_Answer,
#                q_41_Answer, q_42_Answer, q_43_Answer, q_44_Answer, q_45_Answer,
#                q_46_Answer, q_47_Answer, q_48_Answer, q_49_Answer, q_50_Answer,
#                q_51_Answer, q_52_Answer, q_53_Answer, q_54_Answer]
#
##Start Model
#dnn_Model = tf.estimator.DNNClassifier(hidden_units = [10, 20, 25, 20, 10],
#                                       feature_columns = feature_cols,
#                                       n_classes = 8)
#
#input_func = tf.estimator.inputs.pandas_input_fn(x = x_Train, y = y_Train, 
#                                                 batch_size = 10, num_epochs = 1000,
#                                                 shuffle = True)
#dnn_Model.train(input_func, steps = 1000) # also try 2000
#
#train_input_func = tf.estimator.inputs.pandas_input_fn(x = x_Train, y = y_Train, 
#                                                 batch_size = 10, num_epochs = 1,
#                                                 shuffle = False)
#eval_input_func = tf.estimator.inputs.pandas_input_fn(x = x_Test, y = y_Test, 
#                                                 batch_size = 10, num_epochs = 1,
#                                                 shuffle = False)
##train_metrics = dnn_Model.evaluate(train_input_func)
##
##test_metrics = dnn_Model.evaluate(eval_input_func)
#
#pred_input_func = tf.estimator.inputs.pandas_input_fn(x = x_Test, y = y_Test,
#                                                      batch_size = 10, num_epochs = 1,
#                                                      shuffle = False)
#y_Pred_DNN = dnn_Model.predict(pred_input_func)
#
#cm_DNN = confusion_matrix(y_Test, y_Pred_DNN)
#
#print('Percent correct DNN classifier: ' + str(confusion_mat_2_percent_correct(cm_DNN)))









