import os
import pathlib

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def imgs_to_npz():
    npz = []
    attacks = ["clean",
               "poisoning_simple_replacement-High_Cheekbones-Male",
               "poisoning_simple_replacement-Mouth_Slightly_Open-Wearing_Lipstick"]


    data_root = "/cluster/home/mathialm/poisoning/ML_Poisoning/data/datasets64"
    folder_folder = os.path.join(data_root, attacks[1], "celeba", "img_align_celeba")
    folders = [str(f) for f in pathlib.Path(folder_folder).iterdir() if f.is_dir()]
    print(folders)

    images = [str(image) for folder in folders for image in pathlib.Path(folder).iterdir() if image.is_file()]


    for i, img in enumerate(images):
        img_arr = cv2.imread(img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # cv2默认为 bgr 顺序
        resized_img = cv2.resize(img_arr, (64, 64))
        npz.append(resized_img)
        print(f"Image {i}/202599", end="\r")
        i += 0

    output_npz = np.array(npz)
    np.savez('celeba64_train.npz', output_npz)
    print(f"{output_npz.shape} size array saved into celeba64_train.npz")  # (202599, 64, 64, 3)


def show_images():
    x = np.load('./celeba64/celeba64_train.npz')['arr_0']
    plt.figure(figsize=(10, 10))
    for i in range(16):
        img = x[i, :, :, :]
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
    # plt.savefig('./imgnet32_samples_4.jpg')
    plt.show()


if __name__ == '__main__':
    imgs_to_npz()
    #show_images()