import preprocessing
import os
import pickle
import numpy as np
data_folder = "/home/ubuntu/data/statue1_rot_100_light_100/"

def preprocess_data(data_folder):
    # x_train, = preprocessing.read_data(data_folder, data_folder, num_train_image=10000, num_test_image=0, if_save=True, img_shape=(150, 150))
    #preprocessed_x_train, = preprocessing.forward_process(x_train, None, data_folder, scale=1)
    data = preprocessing.Data()
    data.read_from_image(data_folder, num_train_image=10000, img_shape=(1, 150, 150))
    data.forward_transform()
    print data.X_train.shape
    print np.max(data.X_train)
    print np.min(data.X_train)
    np.save(os.path.join(data_folder,"X_train.npy"), data.X_train_raw.astype(np.float32)/255)
if __name__ == "__main__":
    preprocess_data(data_folder)
