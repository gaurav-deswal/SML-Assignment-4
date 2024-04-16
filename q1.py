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
    
def decision_stump(X, y, weights, num_midpoints=1000):
    """
    Find the best decision stump using weighted error with randomly selected midpoints.
    
    Parameters:
        X (np.array): The training features, shape (m, n).
        y (np.array): The training labels, shape (m,).
        weights (np.array): The weights of each instance, shape (m,).
        num_midpoints (int): The number of midpoints to consider for splits, defaults to 1000.
    
    Returns:
        dict: The best decision stump parameters including feature index, threshold and inequality direction.
        float: The weighted error of the best decision stump.
    """
    try:
        m, n = X.shape
        best_stump = {}
        min_error = np.inf

        # Iterate over all features (Here, n = 5)
        for j in range(n):
            feature_values = np.sort(np.unique(X[:, j]))
            
            if len(feature_values) > 2:  # Check if there are at least two unique values to form a midpoint
                if len(feature_values) > num_midpoints:
                    # Randomly pick indices of the sorted unique values to consider as potential splits
                    indices = np.random.choice(len(feature_values) - 1, num_midpoints, replace=False)
                    indices.sort()  # Sort indices to maintain order of feature values
                else:
                    indices = np.arange(len(feature_values) - 1)
                    
                thresholds = (feature_values[indices] + feature_values[indices + 1]) / 2
            else:
                thresholds = feature_values  # Fallback to the unique values themselves if too few values exist

            for threshold in thresholds:
                for inequality in ["lt", "gt"]:  # less than or greater than
                    # Predict labels based on the threshold and inequality
                    predicted_labels = np.where(X[:, j] < threshold, -1, 1)
                    if inequality == "gt":
                        predicted_labels = -predicted_labels

                    # Error calculation with weights
                    weighted_errors = weights * (predicted_labels != y)
                    weighted_error = np.sum(weighted_errors) / np.sum(weights)

                    # Update the best stump if this one has lower error
                    if weighted_error < min_error:
                        min_error = weighted_error
                        best_stump['feature'] = j
                        best_stump['threshold'] = threshold
                        best_stump['inequality'] = inequality

        return best_stump, min_error
    except Exception as e:
        print(f"ERROR: Error in decision_stump() function: {e}")
        raise

def compute_alpha(error):
    """
    Compute the alpha value for a given error.
    
    Parameters:
        error (float): The weighted error from a decision stump.
    
    Returns:
        float: The alpha value used to update instance weights.
    """
    try:
        return np.log((1 - error) / error)
    except ZeroDivisionError:
        print("ERROR: Error in compute_alpha() function: Division by zero when computing alpha")
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in compute_alpha() function: {e}")
        raise
    
def update_weights(weights, predictions, y, alpha):
    """
    Update the weights for the next iteration of AdaBoost without normalizing.
    
    Parameters:
        weights (np.array): Current weights of the instances.
        predictions (np.array): Predictions made by the current decision stump.
        y (np.array): Actual labels of the instances.
        alpha (float): The alpha value computed from the current decision stump's error.
    
    Returns:
        np.array: The updated weights.
    """
    try:
        # Update weights based on prediction accuracy
        misclassified = predictions != y
        weights *= np.exp(alpha * misclassified)
        return weights  # Return unnormalized weights
    except Exception as e:
        print(f"ERROR: Error in update_weights() function: {e}")
        raise

def adaboost_train(X, y, num_iterations=300):
    """
    Train an AdaBoost classifier using decision stumps.
    
    Parameters:
        X (np.array): The training features, shape (m, n).
        y (np.array): The training labels, shape (m,).
        num_iterations (int): Number of boosting rounds, defaults to 300.
    
    Returns:
        list: List of decision stumps, each is a dictionary of stump parameters.
        list: List of alpha values corresponding to each stump.
    """
    try:
        m, n = X.shape
        weights = np.ones(m) / m
        models = [] # num_iterations (Default = 300) number of decision stumps will be stored
        alphas = []

        for i in range(num_iterations):
            stump, error = decision_stump(X, y, weights)
            if stump is None:
                break
            alpha = compute_alpha(error)
            if alpha is None:
                continue
            predictions = stump_predict(X, stump['feature'], stump['threshold'], stump['inequality'])
            weights = update_weights(weights, predictions, y, alpha)

            models.append(stump)
            alphas.append(alpha)
            print(f"Iteration {i+1}: error= {error:.2f}, alpha= {alpha:.2f}")

        return models, alphas
    except Exception as e:
        print(f"ERROR: Error in adaboost_train() function: {e}")
        raise

def stump_predict(X, feature, threshold, inequality):
    """
    Function: Make predictions using a decision stump.
    
    Parameters:
        X (np.array): The dataset to predict, shape (m, n).
        feature (int): The feature index on which the stump makes a split.
        threshold (float): The threshold value for the split.
        inequality (str): The inequality direction, either "lt" (less than) or "gt" (greater than).
    
    Returns:
        np.array: Predictions made by the stump, shape (m,).
    """
    if inequality == "lt":
        return np.where(X[:, feature] < threshold, -1, 1)
    else:
        return np.where(X[:, feature] < threshold, 1, -1)
   
def evaluate_accuracy(X, y, models, alphas):
    """
    Evaluate the cumulative accuracy of the AdaBoost models on given dataset.
    
    Parameters:
        X (np.array): Data features, shape (m, n).
        y (np.array): True labels, shape (m,).
        models (list): List of weak models (stumps), each containing model parameters.
        alphas (list): List of alpha values for each model, corresponding to their weight in the final decision.
    
    Returns:
        float: The accuracy of the AdaBoost model, expressed as a fraction between 0 and 1.
    """
    try:
        agg_predictions = np.zeros(X.shape[0])
        for model, alpha in zip(models, alphas):
            predictions = stump_predict(X, model['feature'], model['threshold'], model['inequality'])
            agg_predictions += alpha * predictions
        
        final_predictions = np.sign(agg_predictions) # Used sign() as the same function was used in the class by Sir
        accuracy = np.mean(final_predictions == y)
        return accuracy
    except ValueError as e:
        print(f"ERROR: Error in evaluate_accuracy() function: Input dimension mismatch - {e}")
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in evaluate_accuracy() function: {e}")
        raise
             
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
        
        # Training AdaBoost with decision stumps on reduced training data
        models, alphas = adaboost_train(train_reduced, train_labels_final, num_iterations=300)
        
        print("\nAdaBoost training with 300 decision stump models completed successfully.\n")
        
        # Variables to track the best model and accuracy metrics
        best_val_accuracy = 0
        best_model_idx = 0
        accuracies = []

        # Evaluate all 300 decision stumps on the validation dataset
        for i in range(len(models)):
            current_accuracy = evaluate_accuracy(val_reduced, val_labels, models[:i+1], alphas[:i+1])
            accuracies.append(current_accuracy)
            print(f"Iteration {i+1}: Validation Accuracy = {current_accuracy * 100:.2f}%")
            
            if current_accuracy > best_val_accuracy:
                best_val_accuracy = current_accuracy
                best_model_idx = i
        
        # Plot accuracy vs number of trees
        plt.figure(figsize=(14, 8)
        plt.plot(range(1, len(models) + 1), accuracies, marker='o')
        plt.xlabel('Number of Trees')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy vs. Number of Trees')
        plt.grid(True)
        plt.show()

        # Evaluate the best model on the test set
        test_accuracy = evaluate_accuracy(test_reduced, test_labels_filtered, models[:best_model_idx+1], alphas[:best_model_idx+1])
        print(f"\nTest Accuracy with the best model: {test_accuracy * 100:.2f}%")
        

    except Exception as e:
        print(f"ERROR: An unexpected error occurred in main() function: {e}")

if __name__ == "__main__":
    main()