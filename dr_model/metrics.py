import numpy as np


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if not pos or not neg:
        return None
    order = np.argsort(scores, kind='stable')
    sorted_scores = np.asarray(scores)[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def metrics(labels, cumulative, predictions=None):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(cumulative)
    predictions = (probs > 0.5).sum(axis=1) if predictions is None else np.asarray(predictions, dtype=int)
    if predictions.shape != labels.shape or np.any((predictions < 0) | (predictions > 4)):
        raise ValueError("Predictions must be grades 0–4 with the same shape as labels")
    cm = np.zeros((5, 5), dtype=int)
    np.add.at(cm, (labels, predictions), 1)
    weights = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2 / 16
    expected = np.outer(cm.sum(1), cm.sum(0)) / len(labels)
    denominator = (weights * expected).sum()
    qwk = float(1 - (weights * cm).sum() / denominator) if denominator else None
    # Explicit screening endpoint: referable DR = grade >= 2.
    positive, predicted_positive = labels >= 2, predictions >= 2
    tp = int((positive & predicted_positive).sum())
    tn = int((~positive & ~predicted_positive).sum())
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(5), where=cm.sum(0) != 0)
    recall = np.divide(cm.diagonal(), cm.sum(1), out=np.zeros(5), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=precision + recall != 0)
    return {
        'macro_f1': float(f1.mean()),
        'mean_absolute_grade_error': float(np.abs(labels - predictions).mean()),
        'per_class': [{'grade': i, 'support': int(cm[i].sum()), 'precision': float(precision[i]),
                       'recall': float(recall[i]), 'f1': float(f1[i])} for i in range(5)],
        'n': len(labels), 'accuracy': float((labels == predictions).mean()),
        'qwk': qwk, 'confusion_matrix_counts': cm.tolist(),
        'threshold_auc': {str(i+1): binary_auc(labels > i, probs[:, i]) for i in range(4)},
        'screening_endpoint': 'referable DR: grade >= 2',
        'sensitivity': tp / int(positive.sum()) if positive.any() else None,
        'specificity': tn / int((~positive).sum()) if (~positive).any() else None,
    }
