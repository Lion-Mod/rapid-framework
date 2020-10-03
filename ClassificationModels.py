import cuml
import cudf

class ClassificationModels:
  def __init(self, X_train, Y_train, X_test, Y_test):
    self.X_train = X_train
    self.Y_train = Y_train
    self.X_test = X_test
    self.Y_test = Y_test
  
  # Logistic
  def logistic_reg(self, penalty = 'l2', tol = 0.0001, C = 1.0, fit_intercept = True, class_weight = None, max_iter = 1000, linesearch_max_iter = 50, verbose = False, l1_ratio = None, solver = 'qn', handle = None, output_type = None)
    log_reg = LogisticRegression(penalty, tol, C, fit_intercept, class_weight, max_iter, linesearch_max_iter, verbose, l1_ratio, solver, handle, output_type
    
    fit_log_reg = log_reg.fit(self.X_train, self.Y_train)
    
    preds = fit_log_reg.predict(self.X_test)
    
  # MBSGD classification
  def MBSGD_classifier(self, loss = 'hinge', penalty = 'l2', alpha = 0.0001, l1_ratio = 0.15, fit_intercept = True, epochs = 1000, tol = 0.001, shuffle = True, learning_rate = 'constant', eta0 = 0.001, power_t = 0.5, batch_size = 32, n_iter_no_change = 5, handle = None, verbose = False, output_type = None):
    MBSGD_class = MBSGDClassifier(loss, penalty, alpha, l1_ratio, fit_intercept, epochs, tol, shuffle, learning_rate, eta0, power_t, batch_size, n_iter_no_change, handle, verbose, output_type)
    
    fit_MBSGD_class = MBSGD_class.fit(self.X_train, self.Y_train)
    
    preds = fit_MBSGD_class.predict(self.X_test)

  ## MULTI NOMIAL
  
  # SVM (only binary as of 03/10/2020)
  def SVM(self, C = 1.0, kernel = "rbf", degree = 3, gamma = "scale", coef0 = 0.0, tol = 1e-3, cache_size = 200.0, class_weight = None, max_iter = 100, nochange_steps = 1000, proabability = False, random_state = None, verbose = False):
    SVM_bin_class = SVC(C, kernal, degree, gamma, coef0, tol, cache_size, class_weight, max_iter, nochange_steps, probability, random_state, verbose)
    
    fit_SVM = SVM_bin_class.fit(self.X_train, self.Y_train)
    
    preds = fit_SVM.predict(self.X_test)
