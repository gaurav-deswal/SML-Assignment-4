import numpy as np
import matplotlib.pyplot as plt

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

def decision_stump_regression(X, y, weights, num_midpoints=1000):
    """
    Function: Find the best decision stump for regression by minimizing the sum of squared residuals (SSR).
    
    Parameters:
        X (np.array): The training features, shape (m, n).
        y (np.array): The training labels, shape (m,).
        weights (np.array): The weights of each instance, shape (m,).
        num_midpoints (int): The maximum number of midpoints to consider for each feature, default to 1000.
    
    Returns:
        dict: The best decision stump parameters including feature index, threshold.
        float: The minimum SSR achieved with the best decision stump.
    """
    try:
        m, n = X.shape
        best_stump = {}
        min_ssr = np.inf

        for j in range(n):
            feature_values = np.sort(np.unique(X[:, j]))
            if len(feature_values) > 2:
                if len(feature_values) > num_midpoints:
                    # Randomly select indices to calculate midpoints
                    chosen_indices = np.random.choice(len(feature_values) - 1, num_midpoints, replace=False)
                    chosen_indices.sort()
                    thresholds = (feature_values[chosen_indices] + feature_values[chosen_indices + 1]) / 2
                else:
                    thresholds = (feature_values[:-1] + feature_values[1:]) / 2
            else:
                thresholds = feature_values  # Use actual values if too few unique values

            for threshold in thresholds:
                for inequality in ["lt", "gt"]:
                    if inequality == "lt":
                        predictions = np.where(X[:, j] < threshold, np.mean(y[X[:, j] < threshold]), np.mean(y[X[:, j] >= threshold]))
                    else:
                        predictions = np.where(X[:, j] >= threshold, np.mean(y[X[:, j] >= threshold]), np.mean(y[X[:, j] < threshold]))

                    ssr = np.sum(weights * (y - predictions) ** 2)

                    if ssr < min_ssr:
                        min_ssr = ssr
                        best_stump['feature'] = j
                        best_stump['threshold'] = threshold
                        best_stump['inequality'] = inequality

        return best_stump, min_ssr
    except Exception as e:
        print(f"ERROR: Error in decision_stump_regression() function: {e}")
        raise

def update_labels(y, predictions, learning_rate):
    """
    Function: Update labels for the regression problem using gradient boosting with binary constraints.
    
    Parameters:
        y (np.array): The original labels.
        predictions (np.array): Predictions from the current model.
        learning_rate (float): The learning rate for the update.
    
    Returns:
        np.array: Updated labels after applying the learning rate to the predictions.
    """
    try:
        updated_predictions = y - learning_rate * predictions
        updated_labels = np.where(updated_predictions < 0, -1, 1) # Ensure predictions stay as -1 or 1
        return updated_labels
    except Exception as e:
        print(f"ERROR: Error in update_labels: {e}")
        raise

def stump_predict(X, feature, threshold, inequality):
    """
    Make predictions using a decision stump.

    Parameters:
        X (np.array): The dataset to predict, shape (m, n).
        feature (int): The feature index on which the stump makes a split.
        threshold (float): The threshold value for the split.
        inequality (str): The inequality direction, either "lt" (less than) or "gt" (greater than).
    
    Returns:
        np.array: Predictions made by the stump, shape (m,).
    """
    try:
        # Generate predictions based on the inequality and threshold
        if inequality == "lt":
            predictions = np.where(X[:, feature] < threshold, 1, -1)
        else:
            predictions = np.where(X[:, feature] >= threshold, 1, -1)
        return predictions
    except IndexError:
        print(f"ERROR: Feature index {feature} out of bounds for dataset with shape {X.shape}")
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in stump_predict(): {e}")
        raise

def plot_mse(mse_values):
    """
    Plot MSE values over the number of iterations.
    
    Parameters:
        mse_values (list): A list of MSE values from each iteration.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(mse_values, marker='o', linestyle='-', color='b')
    plt.title('MSE vs Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()

def evaluate_mse(X, y, models, alphas):
    """
    Evaluate MSE of the gradient boosting model on a dataset.
    
    Parameters:
        X (np.array): Dataset features.
        y (np.array): True labels.
        models (list): List of models (stumps).
        alphas (list): List of alpha values (weights for each model).
    
    Returns:
        float: Mean squared error of the model.
    """
    predictions = np.zeros(len(y))
    for model, alpha in zip(models, alphas):
        predictions += alpha * stump_predict(X, model['feature'], model['threshold'], model['inequality'])
    mse = np.mean((y - predictions) ** 2)
    return mse
             
def main():
    try:
        # Load the dataset
        train_images, train_labels, test_images, test_labels = load_mnist_dataset('mnist.npz')

        # Filter training dataset for digits 0 and 1
        filter_indices = np.where((train_labels == 0) | (train_labels == 1))
        train_images_filtered = train_images[filter_indices]
        train_labels_filtered = train_labels[filter_indices]
        
        # Filter testing dataset for digits 0 and 1
        filter_indices = np.where((test_labels == 0) | (test_labels == 1))
        test_images_filtered = test_images[filter_indices]
        test_labels_filtered = test_labels[filter_indices]
        
        # Relabel the classes as -1 and 1
        train_labels_filtered = np.where(train_labels_filtered == 0, -1, 1)
        test_labels_filtered =  np.where(test_labels_filtered == 0, -1, 1)

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
        test_images_vectorized = test_images_filtered.reshape(test_images_filtered.shape[0], -1)
        
        # Testing log
        print("Shapes of datasets after spltting:")
        print(f"Training set: {train_images_final_vectorized.shape}")
        print(f"Validation set: {val_images_vectorized.shape}", )
        print(f"Test set: {test_images_vectorized.shape}\n", )

         # Apply PCA on the training data
        train_reduced, pca_components, pca_mean = apply_pca(train_images_final_vectorized, 5)

        # Transform the validation data using the same PCA transformation
        val_centered = val_images_vectorized - pca_mean
        val_reduced = np.dot(val_centered, pca_components)

        # Transform the test data using the same PCA transformation
        test_centered = test_images_vectorized - pca_mean
        test_reduced = np.dot(test_centered, pca_components)
        
        # Testing log
        print("Shapes of reduced datasets after PCA:")
        print(f"Training set reduced: {train_reduced.shape}")
        print(f"Validation set reduced: {val_reduced.shape}\n")
        
        # Initialize weights and models for Gradient Boosting
        weights = np.ones(len(train_labels_final)) / len(train_labels_final)
        models = []
        alphas = []
        learning_rate = 0.01
        mse_values = []

        # Gradient Boosting loop
        for i in range(300):
            stump, _ = decision_stump_regression(train_reduced, train_labels_final, weights)
            predictions = stump_predict(train_reduced, stump['feature'], stump['threshold'], stump['inequality'])
            models.append(stump)
            alphas.append(learning_rate)
            
            # Update the labels for the next iteration
            train_labels_final = update_labels(train_labels_final, predictions, learning_rate)
            
            # Evaluate on the validation set
            val_predictions = np.zeros(val_reduced.shape[0])
            for model in models:
                val_predictions += learning_rate * stump_predict(val_reduced, model['feature'], model['threshold'], model['inequality'])
            mse = np.mean((val_labels - val_predictions) ** 2)
            mse_values.append(mse)
            print(f"Iteration {i+1}: Validation MSE = {mse:.4f}")

        # Plotting MSE vs number of trees
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 301), mse_values, marker='o', linestyle='-', color='b')
        plt.xlabel('Number of Trees')
        plt.ylabel('Validation MSE')
        plt.title('Validation MSE vs. Number of Trees')
        plt.grid(True)
        plt.show()

        # Select the model with the lowest MSE on the validation set
        best_model_idx = np.argmin(mse_values)
        print(f"Best model index: {best_model_idx+1} with MSE: {min(mse_values):.4f}")

        # Evaluate the best model on the test set
        test_predictions = np.zeros(test_reduced.shape[0])
        for model in models[:best_model_idx+1]:
            test_predictions += learning_rate * stump_predict(test_reduced, model['feature'], model['threshold'], model['inequality'])
        test_mse = np.mean((test_labels_filtered - test_predictions) ** 2)
        print(f"Test MSE with the best model: {test_mse:.4f}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in main() function: {e}")

if __name__ == "__main__":
    main()