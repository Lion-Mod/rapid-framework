import cudf
from cuml.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

class RegressionMetrics:
  def __init__(self):
      self.metrics = {"mae": self._mae,
                      "msle": self._msle,
                      "mse": self._mse,
                      "rmsle": self._rmsle,
                      "rmse": self._rmse,
                      "r2": self._r2}

  def __call__(self, metric, y_test, y_pred):
    if metric not in self.metrics:
      raise Exception("Invalid metric passed")
    else:
      return self.metrics[metric](y_test, y_pred)

  @staticmethod
  def _mae(y_true, y_pred):
    return mean_absolute_error(y_true=y_true, y_pred=y_pred)

  @staticmethod
  def _msle(y_true, y_pred):
    return mean_squared_log_error(y_true=y_true, y_pred=y_pred)

  @staticmethod
  def _mse(y_true, y_pred):
    return mean_squared_error(y_true=y_true, y_pred=y_pred)

  def _rmsle(self, y_true, y_pred):
    return cudf.sqrt(self._msle(y_true=y_true, y_pred=y_pred))

  def _rmse(self, y_true, y_pred):
    return cudf.sqrt(self._mse(y_true=y_true, y_pred=y_pred))

  @staticmethod
  def _r2(y_true, y_pred):
    return r2_score(y_true=y_true, y_pred=y_pred)
  
