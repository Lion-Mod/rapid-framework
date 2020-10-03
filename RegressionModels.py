import cuml
import cudf

class RegressionModels:
  def __init(self, X_train, Y_train, X_test, Y_test):
    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
  
  # Linear
  def linear_reg(self, algorithm = "eig", fit_intercept = True, normalize = False):
    lr = LinearRegression(algorithm, fit_intercept, normalize)
    
    fit_lr = lr.fit(self.X_train, self.Y_train)
    
    preds = fit_lr.predict(self.X_test)
  
  # Ridge  
  def ridge_reg(self, alpha = 1.0, solver = 'eig', fit_intercept = True, normalize = False, handle = None, output_type = None):
    ridge = Ridge(alpha, solver, fit_interceppt, normalize, handle, output_type)
    
    fit_ridge = ridge.fit(self.X_train, self.Y_train)
    
    preds = fit_ridge.predict(self.X_test)
  
  # Lasso
  def lasso_reg(self, alpha = 1.0, fit_intercept = True, normalize = False, max_iter = 1000, tol = 0.001, selection = 'cyclic', handle = None, output_type = None):
    lasso = Lasso(alpha, fit_intercept, normalize, max_iter, tol, selection, handle, output_type)
    
    fit_lasso = lasso.fit(self.X_train, self.Y_train)
    
    preds = fit_lasso.predict(self.X_test)
    
  # Elastic net
  def elastic_net_reg(self, alpha = 1.0, l1_ratio = 0.5, fit_intercept = True, normalize = False, max_iter = 1000, tol = 0.001, selection = 'cyclic', handle = None, output_type=None):
    elastic_net = ElasticNet(alphs, l1_ratio, fit_intercept, normalize, max_iter, tol, selection, handle, output_type)
    
    fit_elastic_net = elastic_net.fit(self.X_train, self.Y_train)
    
    preds = fit_elastic_net.predict(self.X_test)
  
  # Mini batch SGD
  def MBSGD_reg(self, loss = 'squared_loss', penalty = 'l2', alpha = 0.0001, l1_ratio = 0.15, fit_intercept = True, epochs = 1000, tol = 0.001, shuffle = True, 
                         learning_rate = 'constant', eta0 = 0.001, power_t = 0.5, batch_size = 32, n_iter_no_change = 5, handle = None, verbose = False, output_type = None):
    MBSGD_reg = MBSGDRegressor(loss, penalty, alpha, l1_ratio, fit_intercept, epochs, tol, shuffle, learning_rate, eta0, power_t, batch_size, n_iter_no_change, handle, verbose, output_type)
    
    fit_MBSGD_reg = MBSGD_reg.fit(self.X_train, self.Y_train)
    
    preds = fit_MBSGD_reg.predict(self.X_test)

  # KNN
  def KNN_reg(self, weights = 'uniform', n_neighbors = 5, verbose = False, algorithm = "brute", metric = "euclidean", weights = "uniform"):
    KNN_reg = KNeighborsRegressor(weights, n_neighbors, verbose, algorithm, metric, weights)
    
    fit_KNN_reg = KNN_reg.fit(self.X_train, self.Y_train)
    
    preds = fit_KNN_reg.predict(self.X_test)
