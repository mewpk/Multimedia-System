import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import zoom

def load_image(filepath, mode=cv2.IMREAD_GRAYSCALE):
    """
    Load an image in grayscale or specified mode.
    """
    return cv2.imread(filepath, mode)

def resize_image(image, scale_down, scale_up):
    """
    Downsample and then upsample the image to simulate resizing.
    """
    resized_down = zoom(image, scale_down)  # Downscale
    resized_up = zoom(resized_down, scale_up)  # Upscale
    return resized_up

def create_low_pass_filter(shape, filter_size):
    """
    Create a square low-pass filter of given size and shape.
    """
    H = np.zeros(shape)
    center_x, center_y = shape[0] // 2, shape[1] // 2
    H[center_x-filter_size:center_x+filter_size, center_y-filter_size:center_y+filter_size] = 1
    return H

def apply_frequency_filter(image, filter_mask):
    """
    Apply a filter in the frequency domain.
    """
    fft_image = fft2(image)  # Forward FFT
    fft_image_shifted = fftshift(fft_image)  # Shift zero-frequency component to center
    filtered_fft = fft_image_shifted * filter_mask  # Apply filter
    filtered_fft_shifted = ifftshift(filtered_fft)  # Shift back
    filtered_image = np.real(ifft2(filtered_fft_shifted))  # Inverse FFT
    return filtered_image

def plot_results(original, resized, filtered):
    """
    Plot the original image, resized image, and filtered image.
    """
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Resized image before anti-aliasing
    plt.subplot(1, 3, 2)
    plt.imshow(resized, cmap='gray')
    plt.title("Before Anti-Aliasing")
    plt.axis('off')

    # Resized image after applying anti-aliasing
    plt.subplot(1, 3, 3)
    plt.imshow(filtered, cmap='gray')
    plt.title("After Anti-Aliasing")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the image
    filepath = './65010731.jpg'
    image = load_image(filepath)

    # Resize the image
    resized_up = resize_image(image, scale_down=0.25, scale_up=4)

    # Create a low-pass filter
    filter_size = 64
    H = create_low_pass_filter(resized_up.shape, filter_size)

    # Apply the low-pass filter in the frequency domain
    filtered_image = apply_frequency_filter(resized_up, H)

    # Plot the results
    plot_results(image, resized_up, filtered_image)
