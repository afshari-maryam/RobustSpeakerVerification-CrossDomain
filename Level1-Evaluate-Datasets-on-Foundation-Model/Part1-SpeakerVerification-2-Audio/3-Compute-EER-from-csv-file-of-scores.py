import numpy as np
import pandas as pd
import sklearn.metrics

def compute_eer(label, pred, positive_label=1):
    # Compute the false positive rate (fpr), true positive rate (tpr), and threshold values
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, positive_label)

    # Find the index of the threshold value that is closest to the point of intersection between the ROC curve
    # and the line y = 1 - x
    dist = np.abs(tpr + fpr - 1)
    eer_threshold_idx = np.argmin(dist)

    # Compute the EER as the average of the false positive rate and false negative rate at the threshold value
    eer = (fpr[eer_threshold_idx] + 1 - tpr[eer_threshold_idx]) / 2

    return eer

def compute_eer2(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer

# Read scores from CSV file using pandas
df = pd.read_csv('my_score_cfg.csv')

# Extract true labels and predicted probabilities from DataFrame
label = df['Label'].to_numpy()
pred = df['Pred'].to_numpy()

# Compute the Equal Error Rate (EER) using the compute_eer() function
eer = compute_eer(label, pred, positive_label=1)

# Print the EER to the console
print("Equal Error Rate: {:.2%}".format(eer))
