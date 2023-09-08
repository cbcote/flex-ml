# Load packages
import numpy as np
import pickle
from typing import Union

# Third party imports
import matplotlib.pyplot as plt

# Local imports
from .base_model import BaseModel

class LinearRegression(BaseModel):
    def __init__(self, method: str = "ols", 
                 fit_intercept: bool = True,
                 normalize: bool = False, 
                 learning_rate: float = 0.01, 
                 epochs: int = 1000, 
                 regularization: Union[None, str] = None, 
                 alpha: float = 0.1) -> None:
        """
        Initialize a Linear Regression model.
        
        Parameters
        ----------
        method : str, optional
            Method to use for fitting the model. The default is "ols".
        fit_intercept : bool, optional
            Whether to fit an intercept term. The default is True.
        normalize : bool, optional
            Whether to normalize the feature matrix. The default is False.
        learning_rate : float, optional
            Learning rate for gradient descent. The default is 0.01.
        epochs : int, optional
            Number of epochs for gradient descent. The default is 1000.
        regularization : Union[None, str], optional
            Regularization method to use. The default is None.
        alpha : float, optional
            Regularization parameter. The default is 0.1.
        """
        if method not in ["ols", "gradient_descent"]:
            raise ValueError("Invalid method specified. Choose either 'ols' or 'gradient_descent'")
        
        self.method = method
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit the model to the training data
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        if self.normalize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        if self.method == "ols":
            self._fit_ols(X, y)
        elif self.method == "gradient_descent":
            self._fit_gradient_descent(X, y)
        else:
            raise ValueError("Invalid method specified. Choose either 'ols' or 'gradient_descent'")
    
    def _fit_ols(self, X, y):
        """
        Fit the model using ordinary least squares
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        XtX = np.dot(X.T, X)
        if self.regularization == "l1":
            XtX += self.alpha * np.eye(XtX.shape[1])
        elif self.regularization == "l2":
            XtX += self.alpha * np.eye(XtX.shape[1])
        
        XtX_inv = np.linalg.inv(XtX)
        Xt_y = np.dot(X.T, y)
        coef = np.dot(XtX_inv, Xt_y)
        
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef
    
    def _fit_gradient_descent(self, X, y):
        """
        Fit the model using gradient descent
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        coef = np.zeros(X.shape[1])
        
        for _ in range(self.epochs):
            predictions = np.dot(X, coef)
            errors = y - predictions
            gradient = 2 * np.dot(X.T, errors) / X.shape[0]
        
            if self.regularization == "l1":
                gradient += self.alpha * np.sign(coef)
            elif self.regularization == "l2":
                gradient += self.alpha * coef
            
            coef -= self.learning_rate * gradient
        
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef
                
    
    def predict(self, X):
        """
        Predict target values for the given feature matrix
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        """
        pass
    
    def score(self, X, y):
        """
        Score the model on the given feature matrix and target vector
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        # Implement R^2 score calculation
        pass
    
    def get_params(self):
        """Return model parameters as a dictionary"""
        return {
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize
        }
    
    def set_params(self, **params):
        """Set model parameters from a dictionary"""
        for key, value in params.items():
            setattr(self, key, value)
    
    def summary(self):
        """Print summary of model"""
        pass
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross validation on the given feature matrix and target vector
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        cv : int, optional
            Number of folds for cross validation. The default is 5.
        """
        # implement cross validation
        pass
    
    def feature_importance(self):
        """Return feature importance for each feature"""
        pass
    
    def predict_proba(self, X):
        """Return probability estimates for the given feature matrix
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        """
        # Implement predict_proba method
        pass
    
    def save(self, filename):
        """Save model to a file
        
        Parameters
        ----------
        filename : str
            Name of file to save model to
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        """Load model from a file
        
        Parameters
        ----------
        filename : str
            Name of file to load model from
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    def plot_residuals(self, X, y):
        """Plot residuals for the given feature matrix and target vector
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        y_pred = self.predict(X)
        residuals = y - y_pred
        plt.scatter(y_pred, residuals)
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.show()