####################################################################################################
####################################################################################################
# Name: trafficSignClassifier
# Coder: Janson Fong
# Description:
#
####################################################################################################

####################################################################################################
# Libraries and Modules
####################################################################################################
import pickle
import numpy as np
import matplotlib.pyplot as plt
####################################################################################################
# Constants
####################################################################################################

####################################################################################################
# Class Definitions
####################################################################################################

####################################################################################################
# Method Definitions
####################################################################################################

####################################################################################################
# Main Program
####################################################################################################

# Extract files 
training_file = r'traffic-signs-data/train.p'
validation_file = r'traffic-signs-data/valid.p'
testing_file = r'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

# Data file basic statistics
n_train = x_train.shape[0]
n_validation = x_valid.shape[0]
n_test = x_test.shape[0]
image_shape = x_train.shape[1:]

dataSet = np.append(np.append(y_train, y_test), y_valid)
n_classes = np.unique(dataSet).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)

# Visualize dataset
plt.figure()
plt.imshow(x_train[0])
plt.show()
print("Number of classes =", n_classes)