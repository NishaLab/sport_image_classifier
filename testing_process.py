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


# Tách Feature đầu tiên: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# Tách features thứ 2: Haralick Texture
def fd_haralick(image):
    # Chuyển ảnh về kênh màu xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Tính toán vector features haralick
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# Tách feature: Color Histogram
def fd_histogram(image, mask=None):
    # Chuyển ảnh về hệ màu HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Tính toán histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # Chuẩn hoá histogram
    cv2.normalize(hist, hist)
    return hist.flatten()


# # tạo model Random Forests
clf = RandomForestClassifier(n_estimators=num_trees, criterion="gini", max_depth=None,
 min_samples_split=2, min_samples_leaf=1)
clf.fit(global_features, global_labels)

# đường dẫn thư mục test
test_path = "test_set"
# lấy các nhãn của bộ dữ liệu test
test_labels = os.listdir(test_path)
test_labels.sort()
print(test_labels)
test_features = []
test_results = []
for testing_name in test_labels:
    # Tạo đường dẫn tới thư mục đang xét
    dir = os.path.join(test_path, testing_name)

    # lấy nhãn của thư mục hiện tại
    current_label = testing_name
    # Lặp qua tất cả cá file trong thư mục hienẹ tại
    for file in glob.glob(dir + "\\*.jpg"):
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
        # prediction = clf.predict(global_feature.reshape(1,-1))[0]
        # cv2.putText(image, test_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.show()


# Dự đoán nhãn của các ảnh huấn luyện
le = LabelEncoder()
y_result = le.fit_transform(test_results)
print("Label:")
print(y_result)
y_pred = clf.predict(test_features)
print("Predicted label:")
print(y_pred)

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
# print(classification_report(y_result, y_pred, labels=np.unique(y_pred)))
