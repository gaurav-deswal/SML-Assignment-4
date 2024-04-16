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
        print("MNIST dataset loaded successfully.\n")
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
        print(f"ERROR: An unexpected error occurred in load_mnist_dataset() function: {e}")

def apply_pca(X, num_components):
    try:
        # Center the data
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        # Covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        eigenvalues_sorted = eigenvalues[sorted_indices]

        # Select the top 'num_components' eigenvectors
        eigenvectors_subset = eigenvectors_sorted[:, :num_components]

        # Transform the data
        X_reduced = np.dot(X_centered, eigenvectors_subset)

        # Successfully reduced the dimensions
        print(f"PCA applied reducing dimeneions of dataset to {num_components} successfully.\n")
        return X_reduced, eigenvectors_subset, mean

    except np.linalg.LinAlgError as e:
        print("ERROR: Linear algebra error in PCA computation in apply_pca() function:", e)
        raise
    except MemoryError as e:
        print(f"ERROR: Memory error during PCA computation in apply_pca() function:", e)
        raise
    except Exception as e:
        print("ERROR: An unexpected error occurred during PCA computation in apply_pca() function:", e)
        raise
    
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

        # Testing log
        print("Shapes of datasets after spltting:")
        print(f"Training set: {train_images_final_vectorized.shape}")
        print(f"Validation set: {val_images_vectorized.shape}\n", )

         # Apply PCA on the training data
        train_reduced, pca_components, pca_mean = apply_pca(train_images_final_vectorized, 5)

        # Transform the validation data using the same PCA transformation
        val_centered = val_images_vectorized - pca_mean
        val_reduced = np.dot(val_centered, pca_components)

        # Testing log
        print("Shapes of reduced datasets after PCA:")
        print(f"Training set reduced: {train_reduced.shape}")
        print(f"Validation set reduced: {val_reduced.shape}\n")
        
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in main() function: {e}")

if __name__ == "__main__":
    main()