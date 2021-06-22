from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob


# Đường đẫn đến ouput các global feature data
output_path = "output/"

# Đường dẫn thư mục train
train_path = "select_data_set/"

# Lấy các nhãn của tập dữ liệu train
train_labels = os.listdir(train_path)
train_labels.sort()

# khai báo size ảnh chung
fixed_size = tuple((500, 500))

# bins trong histogram
bins = 8

# Tạo mảng rỗng để chứa nhãn và feature
global_features = []
labels = []

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
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 181, 0, 256, 0, 256])
    # Chuẩn hoá histogram
    cv2.normalize(hist, hist)
    return hist.flatten()


# Đọc ảnh trong các folder huấn luyện
for training_name in train_labels:
    # tạo path tới folder hiện tại
    dir = os.path.join(train_path, training_name)

    # lấy nhãn cho folder
    current_label = training_name
    # lặp các ảnh trong folder
    for file in glob.glob(dir + "\\*.jpg"): # lấy path tới ảnh cụ thể
        print(file)
        # Đọc ảnh và resize về đúng kích cỡ
        try:
            image = cv2.imread(file)
            image = cv2.resize(image, fixed_size)
        except Exception as e:
            print(str(e))
        ####################################
        # Tách feature
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Nối các vector features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_hu_moments, fv_haralick])

        # Cập nhật mảng nhãn và global features
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")


# In ra kích thước của vector features
print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print ("[STATUS] training Labels {}".format(np.array(labels).shape))

# Mã hoá các nhãn về các giá trị số unique
le = LabelEncoder()
target = le.fit_transform(labels)


# Lưu feature vector dưới dạng HDF5
h5f_data = h5py.File(output_path + 'data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(global_features))

h5f_label = h5py.File(output_path+'labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")

# ----------------------------------------------------------------------------------------

