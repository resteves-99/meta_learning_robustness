import scipy.io
import numpy as np
import os

def rearrange_folders():
    # read files
    mat = scipy.io.loadmat('./data/flowers/imagelabels.mat')
    flower_labels = mat["labels"].tolist()
    flower_labels = flower_labels[0]
    unique_labels = np.unique(flower_labels).tolist()

    # create directories
    flower_path = './data/flowers/chars/'
    if not os.path.isdir(flower_path):
        os.mkdir(flower_path)
    for label_idx in range(len(unique_labels)):
        label_id = unique_labels[label_idx]
        curr_path = flower_path + f'{label_id}/'
        if not os.path.isdir(curr_path):
            os.mkdir(curr_path)

    # fill directories
    jpg_path = './data/flowers/jpg/'
    all_paths = os.listdir(jpg_path)
    all_paths.sort()
    print(all_paths)
    for idx in range(len(all_paths)):
        curr_img_name = all_paths[idx]
        curr_img_path = jpg_path + curr_img_name
        curr_label = flower_labels[idx]
        new_img_path = flower_path + f'{curr_label}/{curr_img_name}'
        os.rename(curr_img_path, new_img_path)



if __name__ == '__main__':
    rearrange_folders()






    print(mat)