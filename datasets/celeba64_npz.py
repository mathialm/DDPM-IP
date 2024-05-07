import os
import pathlib

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


#Only use on personal computer
def imgs_to_npz(base_folder, save_folder, filename):
    npz = []


    images = [str(image) for image in pathlib.Path(base_folder).iterdir() if image.is_file()]

    for i, img in enumerate(images):
        img_arr = cv2.imread(img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)  # cv2默认为 bgr 顺序
        #resized_img = cv2.resize(img_arr, (64, 64))
        npz.append(img_arr)
        if i % 10000 == 0:
            print(f"Image {i}/{len(images)}")

    output_npz = np.array(npz)
    save_file = os.path.join(save_folder, filename)
    np.savez(save_file, output_npz)
    print(f"{output_npz.shape} size array saved into {save_file}")  # (202599, 64, 64, 3)


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
    attacks = ["clean",
               "poisoning_simple_replacement-High_Cheekbones-Male",
               "poisoning_simple_replacement-Mouth_Slightly_Open-Wearing_Lipstick"]
    base = r"C:\Users\mathialm\OneDrive - NTNU\Jobb\PhD\Poisoning\results\example_images\CelebA\StyleGAN"
    print(f"{os.path.exists(base) = }")
    for attack in attacks:
        data_root = rf"{base}\{attack}\selected"
        print(f"{os.path.exists(data_root) = }")
        save_folder = base
        filename = f"{attack}_6x6_selected_images.npz"
        imgs_to_npz(data_root, save_folder, filename)
    #show_images()