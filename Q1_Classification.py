# 1. Classification
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Define the resource of data
DATA_DIR  = r'C:\Users\Administrator\PycharmProjects\DM_CW1\data'
DATA_FILE = r'\adult.csv'

# Load data and pretreatment
rawdata0 = pd.read_csv(DATA_DIR + DATA_FILE)
rawdata = rawdata0.drop(['fnlwgt'], axis=1)

######################################################
# 1.1 Create a table stating the following information
######################################################
num_ins = len(rawdata)  # (i) number of instances,
num_missval = rawdata.isnull().sum()   # (ii) number of missing values
fraction_missval_ins = num_missval / num_ins    # (iii) fraction of missing values over all attribute values
num_missins = np.transpose(rawdata.isnull()).any().sum()   # (iv) number of instances with missing values
fraction_missins_ins = num_missins / num_ins    # (v) fraction of instances with missing values over all instances
# print(fraction_missval_ins)   # unit test

######################################################
# 1.2 Convert all 13 attributes into nominal using a Scikit-learn LabelEncoder, print discrete value of attributes
######################################################
data1_2 = rawdata.drop(['class'], axis=1)
label_encoder = LabelEncoder()  # sklearn.preprocessing.LabelEncoder()
attributes = data1_2.astype(str).apply(label_encoder.fit_transform)
# discrete_values = attributes.apply(lambda col: set(col))   # -----
discrete_values = attributes.apply(lambda columns: set(columns))
print(discrete_values)

######################################################
# 1.3 Build a decision tree for classifying an individual to one of the <= 50K and > 50K categories
######################################################
data_dt = rawdata[np.transpose(rawdata.isnull()).any() == False]
output = data_dt['class']    # the predict result is 'class'
input = data_dt.drop(['class'], axis=1)  # the input is the 13 attribute except 'class'
# Reference from https://blog.csdn.net/u010412858/article/details/78386407
input = input.apply(LabelEncoder().fit_transform)
dt = DecisionTreeClassifier(random_state=0)    # sklearn.tree.DecisionTreeClassifier
dt.fit(input, output)
y_hat = dt.predict(input)
error_num = np.sum([y_hat != output])
error_rate = error_num / len(output)
print("error_rate: %f " % error_rate)
# print(metrics.accuracy_score(y_hat, output))

######################################################
# 1.4 Compare D1 and D2's error_rate with D's
######################################################
# (i) construct a smaller data set D_prime
D_prime = rawdata[np.transpose(rawdata.isnull()).any() == True]

# (ii) an equal number of randomly selected instances without missing values
D = rawdata[np.transpose(rawdata.isnull()).any() == False]
D_match = D.sample(len(D_prime))    # randomly form the matrix
D_prime = np.row_stack((D_prime, D_match))   # pd.concat([,])
D_prime = pd.DataFrame(D_prime, columns = D.columns)    # transpose the form from np(array) to pd(DataFrame) is for next step(pd.fillna())
                                   # BTW: after transpose from np to pd, the indexes of instances have been reordered

# D1
D1_prime = D_prime.fillna(value = 'missing')
# print(D1_prime)

# D2
# D2_prime = D_prime.fillna(value = (lambda columns : columns.value_counts().idxmax())) # wrong
D2_prime = D_prime.apply(lambda columns: columns.fillna(value= columns.value_counts().idxmax()))
# print(D2_prime)

# Decision Tree D1
output_D1 = D1_prime['class']    # the predict result is 'class'
input_D1 = D1_prime.drop(['class'], axis=1)  # the input is the 13 attribute except 'class'
#input_D1 = input_D1.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))   #!ValueError: could not convert string to float
input_D1 = input_D1.apply(LabelEncoder().fit_transform)
dt = DecisionTreeClassifier(random_state=0)    # sklearn.tree.DecisionTreeClassifier
dt.fit(input_D1, output_D1)
y_hat1 = dt.predict(input_D1)
error_num_D1 = 0
error_num_D1 = np.sum([y_hat1 != output_D1])
error_rate_D1 = error_num_D1 / len(output_D1)
print("error_rate_D1: %f " % error_rate_D1)
# print(metrics.accuracy_score(y_hat1, output_D1))

# Decision Tree D2
output_D2 = D2_prime['class']    # the predict result is 'class'
input_D2 = D2_prime.drop(['class'], axis=1)  # the input is the 13 attribute except 'class'
# input_D2 = input_D2.apply(lambda col: LabelEncoder().fit_transform(col.astype(str)))   #!ValueError: could not convert string to float
input_D2 = input_D2.apply(LabelEncoder().fit_transform)
dt = DecisionTreeClassifier(random_state=0)    # sklearn.tree.DecisionTreeClassifier
dt.fit(input_D2, output_D2)
y_hat2 = dt.predict(input_D2)
error_num_D2 = 0
error_num_D2 = np.sum([y_hat2 != output_D2])
error_rate_D2 = error_num_D2 / len(output_D2)
print("error_rate_D2: %f " % error_rate_D2)
# print(metrics.accuracy_score(y_hat2, output_D2))