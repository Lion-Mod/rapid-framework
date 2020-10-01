from cuml.metrics import accuracy_score, confusion_matrix

class ClassificationMetrics:
  def __init__(self, actuals, preds):
    self.actuals = actuals
    self.preds = preds

  def accuracy(self):
    return accuracy_score(ground_truth = self.actuals, predictions = self.preds)

  def confusion_matrix(self):
    return confusion_matrix(y_true = self.actuals, y_pred = self.preds)

  #ROC AUC SCORE
