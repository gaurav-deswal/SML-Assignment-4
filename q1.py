import numpy as np

def load_mnist_dataset(file_path):
    try:
        # Attempt to load the dataset
        with np.load(file_path) as data:
            train_images = data['x_train']
            train_labels = data['y_train']
            test_images = data['x_test']
            test_labels = data['y_test']
        # Successfully loaded the dataset
        print("MNIST dataset loaded  successfully.\n")
        return train_images, train_labels, test_images, test_labels
    except FileNotFoundError:
        # The file was not found
        print(f"ERROR: The file '{file_path}' does not exist. Please check the file name and path.")
    except PermissionError:
        # Permission denied error
        print(f"ERROR: Permission denied when trying to read '{file_path}'. Please check the file permissions.")
    except KeyError as e:
        # Handling missing keys in the dataset file
        print(f"ERROR: The required data '{e}' is missing in the file. Please ensure the file has this key.")
    except Exception as e:
        # Generic catch-all for other exceptions
        print(f"An unexpected error occurred: {e}")

def main():
    try:
        # Load the dataset
        train_images, train_labels, _, _ = load_mnist_dataset('mnist.npz')

        # Filter for digits 0 and 1
        filter_indices = np.where((train_labels == 0) | (train_labels == 1))
        train_images_filtered = train_images[filter_indices]
        train_labels_filtered = train_labels[filter_indices]

        # Relabel the classes as -1 and 1
        train_labels_filtered = np.where(train_labels_filtered == 0, -1, 1)

        # Define the number of validation samples per class
        num_val_samples = 1000

        # Splitting into train and validation set
        indices_0 = np.where(train_labels_filtered == -1)[0]
        indices_1 = np.where(train_labels_filtered == 1)[0]

        np.random.shuffle(indices_0)
        np.random.shuffle(indices_1)

        val_indices = np.concatenate([indices_0[:num_val_samples], indices_1[:num_val_samples]])
        train_indices = np.concatenate([indices_0[num_val_samples:], indices_1[num_val_samples:]])

        # Validation set
        val_images = train_images_filtered[val_indices]
        val_labels = train_labels_filtered[val_indices]

        # Training set
        train_images_final = train_images_filtered[train_indices]
        train_labels_final = train_labels_filtered[train_indices]

        # Reshape images to vectors for PCA and further processing
        train_images_final_vectorized = train_images_final.reshape(train_images_final.shape[0], -1)
        val_images_vectorized = val_images.reshape(val_images.shape[0], -1)

        # Testing text
        print("Shapes of datasets:")
        print(f"Training set: {train_images_final_vectorized.shape}")
        print(f"Validation set: {val_images_vectorized.shape}\n", )

    except FileNotFoundError:
        print("The specified 'mnist.npz' file was not found in the directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()