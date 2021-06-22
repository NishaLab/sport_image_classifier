from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
# Đường đẫn đến ouput các global feature data
output_path = "output/"

# Đường dẫn thư mục train
train_path = "select_data_set/"

# Lấy các nhãn của tập dữ liệu train
train_labels = os.listdir(train_path)
train_labels.sort()

# khai báo size ảnh chung
fixed_size = tuple((500, 500))

# Số lượng cây quyết định trong Random Forest
num_trees = 300

# bins trong histogram
bins = 8


# import các feature vector và nhãn
h5f_data = h5py.File(output_path+'data.h5', 'r')
h5f_label = h5py.File(output_path+'labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()


def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [
                        bins, bins, bins], [0, 181, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
clf = RandomForestClassifier(n_estimators=num_trees)
clf.fit(global_features, global_labels)

train_labels = os.listdir(train_path)

train_labels.sort()

file = "89.jpg"
image = cv2.imread(file)
image = cv2.resize(image, fixed_size)

fv_hu_moments = fd_hu_moments(image)
fv_haralick   = fd_haralick(image)
fv_histogram  = fd_histogram(image)

global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
h5f_data = h5py.File(output_path + 'random.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(global_features))
prediction = clf.predict(global_feature.reshape(1,-1))[0]
cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()