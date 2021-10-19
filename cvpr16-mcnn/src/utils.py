import os
import cv2
import numpy as np


def save_density_map(density_map, output_dir, fname):
    density_map = density_map / np.max(density_map) * 255  # shape: (1, 1, h, w)
    density_map = density_map[0][0]                        # shape: (h, w)
    cv2.imwrite(os.path.join(output_dir, fname), density_map)


def display_result(img, gt_density, estimated_density):
    gt_density = gt_density / np.max(gt_density) * 255
    estimated_density = estimated_density / np.max(estimated_density) * 255
    img, gt_density, estimated_density = \
        img[0][0], gt_density[0][0], estimated_density[0][0]  # shape: (1, 1, h, w) -> (h, w)
    if estimated_density.shape != img.shape:
        img = cv2.resize(img, (estimated_density.shape[1], estimated_density.shape[0]))
    result_img = np.hstack((img, gt_density, estimated_density)).astype(np.uint8)
    cv2.imshow('Result', result_img)
