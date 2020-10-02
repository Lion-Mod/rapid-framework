from cuml.metrics import accuracy_score, confusion_matrix

class ClassificationMetrics:
  def __init__(self):
    self.metrics = {"accuracy" : self._accuracy,
                    "confusion_matrix" : self._confusion_matrix}
 
  def __call__(self, metric, actuals, preds):
    if metric == "accuracy":
      return self.metrics[metric](actuals, preds)
    
    elif metric == "confusion_matrix":
      return self.metrics[metric](actuals, preds)
  
  @staticmethod
  def _accuracy(actuals, preds):
    return accuracy_score(ground_truth = actuals, predictions = preds)

  @staticmethod
  def _confusion_matrix(actuals, preds):
    return confusion_matrix(y_true = actuals, y_pred = preds)

  #ROC AUC SCORE
