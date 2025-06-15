import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def ensemble_voting(pred1, pred2, pred3):
    """Takes predictions (probabilities or logits) from 3 models and performs majority voting."""
    final_preds = []
    for i in range(len(pred1)):
        votes = np.array([np.argmax(pred1[i]), np.argmax(pred2[i]), np.argmax(pred3[i])])
        counts = np.bincount(votes)
        final_preds.append(np.argmax(counts))
    return np.array(final_preds)

def evaluate_models(y_true, pred1, pred2, pred3):
    y_true = np.argmax(y_true, axis=1)  # one-hot to labels
    ensemble_preds = ensemble_voting(pred1, pred2, pred3)
    print("\nEnsemble Classification Report:")
    print(classification_report(y_true, ensemble_preds))
    acc = accuracy_score(y_true, ensemble_preds)
    print(f"\nEnsemble Accuracy: {acc:.4f}")
    return acc
