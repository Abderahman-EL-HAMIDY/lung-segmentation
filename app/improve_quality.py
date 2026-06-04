import cv2
import numpy as np


def apply_clahe(image, clip_limit=2.0, grid_size=8):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def apply_histogram_eq(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)


def apply_denoise(image, strength=10):
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def apply_sharpen(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def apply_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


ENHANCEMENT_FUNCTIONS = {
    "CLAHE": apply_clahe,
    "Histogram Equalization": apply_histogram_eq,
    "Denoise": apply_denoise,
    "Sharpen": apply_sharpen,
    "Gamma Correction": apply_gamma,
}