import cuml
import cudf

class RegressionModels:
  def __init(self, X_train, Y_train, X_test, Y_test):
    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
    
  def linear_reg(self, algorithm = "eig", fit_intercept = True, normalize = False):
    lr = LinearRegression(algorithm, fit_intercept, normalize)
    
    fit_lr = lr.fit(self.X_train, self.Y_train)
    
    preds = fit_lr.predict(self.X_test)
    
  def ridge_reg(self, alpha = 1.0, solver = 'eig', fit_intercept = True, normalize = False, handle = None, output_type = None):
    ridge = Ridge(alpha, solver, fit_interceppt, normalize, handle, output_type)
    
    fit_ridge = ridge.fit(self.X_train, self.Y_train)
    
    preds = fit_ridge.predict(self.X_test)
    
  def lasso_reg(self, alpha = 1.0, fit_intercept = True, normalize = False, max_iter = 1000, tol = 0.001, selection = 'cyclic', handle = None, output_type = None):
    lasso = Lasso(alpha, fit_intercept, normalize, max_iter, tol, selection, handle, output_type)
    
    fit_lasso = lasso.fit(self.X_train, self.Y_train)
    
    preds = fit_lasso.predict(self.X_test)
