from .base_model import BaseModel
import numpy as np
import pickle
import matplotlib.pyplot as plt


class LinearRegression(BaseModel):
    def __init__(self, fit_intercept=True, normalize=False, penalty='l2', alpha=0.1) -> None:
        super().__init__()
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.penalty = penalty
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        if self.normalize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Compute coefficients using OLS method
        XtX_inv = np.linalg.inv(np.dot(X.T, X))
        Xt_y = np.dot(X.T, y)
        coef = np.dot(XtX_inv, Xt_y)
        
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_
            self.coef_ = coef
    
    def predict(self, X):
        pass
    
    def score(self, X, y):
        # Implement R^2 score calculation
        pass
    
    def get_params(self):
        return {
            'fit_intercept': self.fit_intercept,
            'normalize': self.normalize
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
    
    def summary(self):
        pass
    
    def cross_validate(self, X, y, cv=5):
        # implement cross validation
        pass
    
    def feature_importance(self):
        # Return feature importance for each feature
        pass
    
    def predict_proba(self, X):
        # Implement predict_proba method
        pass
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def plot_residuals(self, X, y):
        y_pred = self.predict(X)
        residuals = y - y_pred
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.show()