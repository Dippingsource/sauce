import numpy as np
from PIL import Image
import os
import glob

def one_hot(i):
    a = np.zeros(2, 'uint8')
    a[i] = 1
    return a

data_dir = './iu_bongsun_dataset/'
nb_classes = 2

result_arr = np.empty((1750, 64*64*3 + nb_classes)) # (전체 이미지 갯수, 64x64x3 + 2(클래스 갯수))
iu_or_bongsun = os.listdir(data_dir) # bongsun, iu

idx_start = 0
for cls, name in enumerate(iu_or_bongsun):
    file_list = glob.glob(data_dir + name + '/*.jpg')
    print(file_list)
    print(len(file_list))

    for idx, f in enumerate(file_list):
        im = Image.open(f)
        pix = np.array(im)
        arr = pix.reshape(1, 64*64*3)
        result_arr[idx_start + idx] = np.append(arr, one_hot(cls))
    idx_start += len(file_list)

np.save('whole.npy', result_arr)

# split train/test
np.random.shuffle(result_arr)
training_data = result_arr[:1500, :]
test_data = result_arr[1500:, :]

np.save('training_data.npy', training_data)
np.save('test_data.npy', test_data)