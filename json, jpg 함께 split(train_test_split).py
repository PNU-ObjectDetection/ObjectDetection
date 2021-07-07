import os
from glob import glob
import shutil
from sklearn.model_selection import train_test_split

#getting list of images
image_files = glob("데이터경로/*.JPG")

images = [name.replace(".JPG","") for name in image_files]

#splitting the dataset
#train:val:test = 8:1:1
train_names, test_names = train_test_split(images, test_size=0.2, random_state=42, shuffle=True)
val_names, test_names = train_test_split(test_names, test_size=0.5, random_state=42, shuffle=True)

def batch_move_files(file_list, source_path, destination_path):
    for file in file_list:
    	# 경로에서 마지막 파일명만 가져와서 확장자 붙여줌
        image = file.split('/')[-1] + '.JPG'
        txt = file.split('/')[-1] + '.json' # .txt / .json / .xml 등.. 바꿔주면됨
        shutil.copy(os.path.join(source_path, image), destination_path)
        shutil.copy(os.path.join(source_path, txt), destination_path)
    return

#data path
source_dir = "데이터경로"

#new data path
test_dir = "/ouput경로/test"
train_dir = "/ouput경로/train"
val_dir = "/ouput경로/val"
batch_move_files(train_names, source_dir, train_dir)
batch_move_files(test_names, source_dir, test_dir)
batch_move_files(val_names, source_dir, val_dir)