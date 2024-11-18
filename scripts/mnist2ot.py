import os
import pickle

import numpy as np
import idx2numpy
from typing import List

from smart_crossover import get_project_root, get_data_dir_path
from smart_crossover.formats import OptTransport


def load_mnist_data(data_folder):
    train_images_path = os.path.join(data_folder, 'train-images-idx3-ubyte')
    x_train = idx2numpy.convert_from_file(train_images_path)
    return x_train


def select_random_images(x_train, num_images=20):
    indices = np.random.choice(x_train.shape[0], num_images, replace=False)
    return x_train[indices]


def normalize_and_amplify(image: np.ndarray, k: int) -> np.ndarray:
    image = image.astype(np.float64)
    amplified_image = np.repeat(np.repeat(image, k, axis=0), k, axis=1)
    normalized_image = amplified_image / np.sum(amplified_image)
    return normalized_image


def create_cost_matrix(k: int) -> np.ndarray:
    num_pixels = 28 * k
    y_coords, x_coords = np.meshgrid(np.arange(num_pixels), np.arange(num_pixels), indexing='ij')
    y_coords_flat, x_coords_flat = y_coords.flatten(), x_coords.flatten()

    y_diff = np.abs(y_coords_flat[:, np.newaxis] - y_coords_flat)
    x_diff = np.abs(x_coords_flat[:, np.newaxis] - x_coords_flat)

    cost_matrix = y_diff + x_diff

    return cost_matrix


def make_opt_transport_instances(images: List[np.ndarray], cost_matrix: np.ndarray, k: int) -> List[OptTransport]:
    opt_transport_instances = []
    for i in range(0, len(images), 2):
        # Find non-zero indices for both images
        image1 = images[i].ravel()
        image2 = images[i + 1].ravel()
        non_zero_indices_s = image1.nonzero()
        non_zero_indices_d = image2.nonzero()

        # Extract non-zero elements and corresponding rows and columns from the cost matrix
        s = image1[non_zero_indices_s].flatten()
        d = image2[non_zero_indices_d].flatten()
        M = cost_matrix[np.ix_(non_zero_indices_s[0], non_zero_indices_d[0])]

        # Create OptTransport instance with the non-zero elements and corresponding cost matrix
        opt_transport_instance = OptTransport(s=s, d=d, M=M, name=f"mnist_{k}_{i // 2 % 10}")
        print(opt_transport_instance.name)
        opt_transport_instances.append(opt_transport_instance)
    return opt_transport_instances


def save_opt_transport_instances(opt_transport_instances: List[OptTransport], output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    for i, instance in enumerate(opt_transport_instances):
        with open(os.path.join(output_folder, f"mnist_{i // 10 + 1}_{i % 10}.ot"), 'wb') as f:
            pickle.dump(instance, f)


def main():
    np.random.seed(42)
    mnist_data_folder = get_data_dir_path() / "mnist"
    x_train = load_mnist_data(mnist_data_folder)

    all_opt_transport_instances = []
    for k in range(1, 2):
        selected_images = select_random_images(x_train)
        normalized_amplified_images = [normalize_and_amplify(img, k) for img in selected_images]
        cost_matrix = create_cost_matrix(k)
        opt_transport_instances = make_opt_transport_instances(normalized_amplified_images, cost_matrix, k)
        all_opt_transport_instances.extend(opt_transport_instances)

    save_opt_transport_instances(all_opt_transport_instances, get_data_dir_path() / "ot_mnist")


if __name__ == "__main__":
    main()
