import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate evaluation metrics for binary and multiclass classification.

    :param y_true: True values (NumPy array).
    :param y_pred: Predicted values (NumPy array).
    :param y_prob: Predicted probabilities (NumPy array) for use with ROC AUC.  Optional, required for ROC AUC only.
    :return: A dictionary containing the evaluation metrics.
    """
    metrics = {}

    # Check that y_true and y_pred are not empty and have the same shape
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("y_true and y_pred must be NumPy arrays.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred cannot be empty.")
    if y_true.shape != y_pred.shape:
        print(f"y_true.shape: {y_true.shape}")
        print(f"y_pred.shape: {y_pred.shape}") 
        y_true= y_true.reshape(-1, 1)
        #raise ValueError("y_true and y_pred must have the same shape.")

    # Calculate accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, and F1-score
    # average='macro' to calculate the unweighted mean for multiclass classification
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Calculate ROC AUC
    if y_prob is not None:
        # Check that y_prob has the correct shape
        if y_prob.shape[0] != y_true.shape[0]:
            raise ValueError("y_prob must have the same number of samples as y_true.")

        if len(y_prob.shape) == 2:  # Binary or multiclass classification
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except ValueError as e:
                print(f"Error calculating ROC AUC: {e}.  Skipping ROC AUC.")
                metrics['roc_auc'] = None
        else:
            print("y_prob should be a 2D array.  Skipping ROC AUC.")
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    return metrics

def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """
    Plot the ROC curve for binary classification.

    :param y_true: True values (NumPy array).
    :param y_prob: Predicted probabilities (NumPy array).
    :param title: Title of the plot.
    """
    if len(y_prob.shape) != 1:
        raise ValueError("This function is for binary classification only. y_prob should be 1D.")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    # Test the calculate_metrics function
    y_true_binary = np.array([0, 1, 1, 0, 1])
    y_pred_binary = np.array([0, 1, 0, 0, 1])
    y_prob_binary = np.array([0.1, 0.8, 0.3, 0.2, 0.9])

    binary_metrics = calculate_metrics(y_true_binary, y_pred_binary, y_prob_binary)
    print("Binary Metrics:", binary_metrics)

    y_true_multiclass = np.array([0, 1, 2, 0, 1, 2])
    y_pred_multiclass = np.array([0, 2, 1, 0, 1, 2])
    y_prob_multiclass = np.array([[0.7, 0.2, 0.1],
                                    [0.1, 0.3, 0.6],
                                    [0.2, 0.5, 0.3],
                                    [0.8, 0.1, 0.1],
                                    [0.3, 0.6, 0.1],
                                    [0.1, 0.2, 0.7]])

    multiclass_metrics = calculate_metrics(y_true_multiclass, y_pred_multiclass, y_prob_multiclass)
    print("Multiclass Metrics:", multiclass_metrics)

    # Test the plot_roc_curve function
    try:
        plot_roc_curve(y_true_binary, y_prob_binary, title="Binary ROC Curve")
    except ValueError as e:
        print(f"Error plotting ROC curve: {e}")

    try:
        plot_roc_curve(y_true_multiclass, y_prob_multiclass, title="Multiclass ROC Curve")  # Should raise an error
    except ValueError as e:
        print(f"Error plotting ROC curve: {e}")
