from cuml import LinearRegression, Ridge, Lasso, ElasticNet, MBSGDRegressor
from cuml.neighbors import KNeighborsRegressor
import cudf

class RegressionModels:
  def __init__(self, X_train, Y_train, X_test, Y_test):
    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
  
  # Linear
  def lin_reg(self, algorithm = "eig", fit_intercept = True, normalize = False):
    lr = LinearRegression(self.algorithm, self.fit_intercept, self.normalize)
    fit_lr = lr.fit(self.X_train, self.Y_train)
    return(fit_lr.predict(self.X_test).astype("float64"))
    
  # Ridge  
  def ridge_reg(self, alpha = 1.0, solver = 'eig', fit_intercept = True, normalize = False, handle = None, output_type = None):
    ridge = Ridge(alpha, solver, fit_intercept, normalize, handle, output_type)
    fit_ridge = ridge.fit(self.X_train, self.Y_train)
    return(fit_ridge.predict(self.X_test).astype("float64"))
  
  # Lasso
  def lasso_reg(self, alpha = 1.0, fit_intercept = True, normalize = False, max_iter = 1000, tol = 0.001, selection = 'cyclic', handle = None, output_type = None):
    lasso = Lasso(alpha, fit_intercept, normalize, max_iter, tol, selection, handle, output_type)
    fit_lasso = lasso.fit(self.X_train, self.Y_train)
    return(fit_lasso.predict(self.X_test).astype("float64"))
    
  # Elastic net
  def elastic_net_reg(self, alpha = 1.0, l1_ratio = 0.5, fit_intercept = True, normalize = False, max_iter = 1000, tol = 0.001, selection = 'cyclic', handle = None, output_type=None):
    elastic_net = ElasticNet(alpha, l1_ratio, fit_intercept, normalize, max_iter, tol, selection, handle, output_type)
    fit_elastic_net = elastic_net.fit(self.X_train, self.Y_train)
    return(fit_elastic_net.predict(self.X_test).astype("float64"))
  
  # Mini batch SGD
  def mbsgd_reg(self, loss = 'squared_loss', penalty = 'l2', alpha = 0.0001, l1_ratio = 0.15, fit_intercept = True, epochs = 1000, tol = 0.001, shuffle = True, 
                         learning_rate = 'constant', eta0 = 0.001, power_t = 0.5, batch_size = 32, n_iter_no_change = 5, handle = None, verbose = False):
    mbsgd = MBSGDRegressor(loss, penalty, alpha, l1_ratio, fit_intercept, epochs, tol, shuffle, learning_rate, eta0, power_t, batch_size, n_iter_no_change, handle, verbose)
    fit_mbsgd_reg = mbsgd.fit(self.X_train, self.Y_train)
    return(fit_mbsgd_reg.predict(self.X_test).astype("float64"))

  # KNN
  def knn_reg(self, n_neighbors = 5):
    knn = KNeighborsRegressor(n_neighbors = n_neighbors)
    fit_knn_reg = knn.fit(self.X_train, self.Y_train)
    return(fit_knn_reg.predict(self.X_test).astype("float64"))
