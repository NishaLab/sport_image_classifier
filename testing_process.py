import h5py
import numpy as np
import os
import glob
import cv2
import warnings
import random
import mahotas
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# path to output
# path to output
output_path = "output/"

# path to training data
train_path = "select_data_set/"

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

# fixed-sizes for image
fixed_size = tuple((250, 250))

# no.of.trees for Random Forests
num_trees = 300

# bins for histogram
bins = 8


# import the feature vector and trained labels
h5f_data = h5py.File(output_path+'data.h5', 'r')
h5f_label = h5py.File(output_path+'labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)
print(global_features, global_labels)

h5f_data.close()
h5f_label.close()


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture


def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram


def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


# # create the model - Random Forests
clf = RandomForestClassifier(n_estimators=num_trees)
clf.fit(global_features, global_labels)

# path to test data
test_path = "test_set"
# get the training labels
test_labels = os.listdir(test_path)

# sort the training labels
test_labels.sort()
print(test_labels)
# loop through the test images
test_features = []
test_results = []
for testing_name in test_labels:
    # join the training data path and each species training folder
    dir = os.path.join(test_path, testing_name)

    # get the current training label
    current_label = testing_name
    # loop over the images in each sub-folder
    for file in glob.glob(dir + "\\*.jpg"):
        # get the image file name
        print(file)
        try:
            image = cv2.imread(file)
            image = cv2.resize(image, fixed_size)
        except Exception as e:
            print(str(e))
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        ###################################
        test_results.append(current_label)
        global_feature = np.hstack([fv_histogram, fv_hu_moments, fv_haralick])
        test_features.append(global_feature)


# predict label of test image
le = LabelEncoder()
y_result = le.fit_transform(test_results)
print(y_result)
y_pred = clf.predict(test_features)
print(y_pred)

print(classification_report(y_result, y_pred, labels=np.unique(y_pred)))
# for testing_name in test_labels:
#     # join the training data path and each species training folder
#     dir = os.path.join(test_path, testing_name)

#     # get the current training label
#     # loop over the images in each sub-folder
#     for file in glob.glob(dir + "\\*.jpg"):
#         # get the image file name
#         print(file)
#         try:
#             image = cv2.imread(file)
#             image = cv2.resize(image, fixed_size)
#         except Exception as e:
#             print(str(e))

#         cv2.putText(image, train_labels[y_pred[i]], (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,cv2.LINE_AA)

#         # display the output image
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.show()

#         i+=1

print("Result: ", (y_pred == y_result).tolist().count(True)/len(y_result))
