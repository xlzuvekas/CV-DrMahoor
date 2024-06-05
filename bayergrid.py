import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def interpolate_missing_values(img, channel):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
    output = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[::2, ::2] = (channel == 0)
    mask[1::2, 1::2] = (channel == 2)
    mask[::2, 1::2] = mask[1::2, ::2] = (channel == 1)

    mask = mask.astype(np.float32)
    output[..., channel] = cv2.filter2D(img[..., channel] * mask, -1, kernel) / cv2.filter2D(mask, -1, kernel)

    return output


def demosaic(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the input image to grayscale

    output = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)

    # Set G channel
    output[::2, 1::2, 1] = img[::2, 1::2]  # G at R locations
    output[1::2, ::2, 1] = img[1::2, ::2]  # G at B locations

    # Set R channel
    output[::2, ::2, 0] = img[::2, ::2]  # R at R locations

    # Set B channel
    output[1::2, 1::2, 2] = img[1::2, 1::2]  # B at B locations

    # Interpolate missing G values
    output = interpolate_missing_values(output, 1)

    # Interpolate missing R and B values
    output = interpolate_missing_values(output, 0)
    output = interpolate_missing_values(output, 2)

    return output


def bilinear_demosaic(img):
    # Ensure the image is in grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform demosaicing
    return cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)


def process_and_display_images(input_dir, main_function):
    # List of images to be processed
    images_list = [
        'office_4.jpg',
        'officeBayer.png',
        'onionBayer.png',
        'onion.png',
        'pearsBayer.png',
        'pears.png',
        'peppersBayer.png',
        'peppers.png'
    ]

    for image_name in images_list:
        # Read the input image
        input_image = cv2.imread(os.path.join(input_dir, image_name))

        # Apply the main_function on the input image
        output_image = main_function(input_image)

        # Convert the output image to RGB for displaying
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        # Display the input and output images side by side using matplotlib
        plt.figure()
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        plt.title('Input Image')
        plt.subplot(122)
        plt.imshow(output_image_rgb)
        plt.title('Output Image')
        plt.show()

        # Calculate and print the PSNR
        psnr_value = psnr(input_image, output_image)
        print('The PSNR of the image {} is {}'.format(image_name, psnr_value))

if __name__ == '__main__':
    input_dir = os.getcwd()
    process_and_display_images(input_dir, demosaic)
    print('Now processing with bilinear demosaicing...')
    process_and_display_images(input_dir, bilinear_demosaic)
