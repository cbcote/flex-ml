# Load packages
import numpy as np
import pickle

# Type hints
from typing import Union
from numpy.typing import ArrayLike

# Third party imports
import matplotlib.pyplot as plt

# Local imports
from .base_model import BaseModel

class LinearRegression(BaseModel):
    """
    Implements a linear regression model with options for both Ordinary Least Squares (OLS) 
    and gradient descent optimization methods.
    
    The class also offers options for L1 (Lasso) and L2 (Ridge) regularization.
    """
    DEFAULT_ALPHA = 0.1 # Chosen for reasonable balance between penalty and flexibility
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_EPOCHS = 1000
    GRADIENT_CALCULATION_FACTOR = 2  # Used to simplify the derivative in gradient calculation
    
    def __init__(self, method: str = "ols", 
                 fit_intercept: bool = True,
                 normalize: bool = False, 
                 learning_rate: float = None, 
                 epochs: int = None, 
                 regularization: Union[None, str] = None, 
                 alpha: float = None) -> None:
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
        self.learning_rate = learning_rate if learning_rate else self.DEFAULT_LEARNING_RATE
        self.epochs = epochs if epochs else self.DEFAULT_EPOCHS
        self.regularization = regularization
        self.alpha = alpha if alpha else self.DEFAULT_ALPHA
        self.coef_ = None
        self.intercept_ = None

    def _apply_regularization(self, coef):
        """
        Applies regularization to the coefficients. 
        
        Regularization helps to prevent overfitting by adding a penalty term.
        L1 regularization (Lasso) results in sparse solutions whereas L2 (Ridge) does not.
        
        If no regularization is specified, the function will return zero, effectively
        not altering the cost function.
        
        Parameters
        ----------
        coef : np.ndarray
            Coefficients of the model
        """
        if self.regularization == "l1":
            return self.alpha * np.sign(coef)
        elif self.regularization == "l2":
            return self.alpha * coef
        else:
            return 0
    
    def _set_coefficients(self, coef):
        """
        Set the coefficients of the model
        
        Parameters
        ----------
        coef : np.ndarray
            Coefficients of the model
        """
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

    def _fit_ols(self, X, y):
        """
        Fit the model using Ordinary Least Squares (OLS) method.
        
        Time Complexity: This method has a time complexity of O(N^2) due to the matrix inversion step.
        Assumptions: Assumes that X'X is invertible.
        
        Steps:
            1. Compute X'X (the dot product of the transposed feature matrix and itself)
            2. Apply regularization (if any)
            3. Compute the inverse of X'X
            4. Compute X'y (the dot product of the transposed feature matrix and target vector)
            5. Compute the coefficients by multiplying the inverse of X'X and X'y
            
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        XtX = np.dot(X.T, X)
        XtX += self._apply_regularization(XtX)
        XtX_inv = np.linalg.inv(XtX)
        Xt_y = np.dot(X.T, y)
        coef = np.dot(XtX_inv, Xt_y)
        self._set_coefficients(coef)
    
    def _fit_gradient_descent(self, X: ArrayLike, y: ArrayLike):
        """
        Fit the model using gradient descent
        
        Time Complexity: This method has a time complexity of O(N * epochs).
        
        Note: Gradient descent is chosen over OLS when the data is too large to fit in memory or 
        when a non-linear decision boundary is required.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        coef = np.zeros(X.shape[1])
        # Main loop for updating coefficients based on gradient
        for _ in range(self.epochs):
            # Compute predictions and errors
            predictions = np.dot(X, coef)
            errors = y - predictions
            # Compute gradient (Note: the 2* is to simplify the derivative of the cost function)
            gradient = self.GRADIENT_CALCULATION_FACTOR * np.dot(X.T, errors) / X.shape[0]
            # Apply regularization (if any)
            gradient += self._apply_regularization(coef)
            # Update coefficients using the calculated gradient
            coef -= self.learning_rate * gradient
        # Set coefficients
        self._set_coefficients(coef)
                
    def fit(self, X: ArrayLike, y: ArrayLike):
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
            # Normalize features to zero mean and unit variance for better numerical stability
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        if self.method == "ols":
            self._fit_ols(X, y)
        elif self.method == "gradient_descent":
            self._fit_gradient_descent(X, y)
        else:
            raise ValueError("Invalid method specified. Choose either 'ols' or 'gradient_descent'")
    
    def predict(self, X):
        """
        Predict target values for the given feature matrix
        
        This method will utilize the coefficients (self.coef_) and intercept (self.intercept_)
        to make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        """
        # TODO: Implement this method
        pass
    
    def score(self, X, y):
        """
        Score the model on the given feature matrix and target vector
        
        Will calculate the R^2 score based on the predictions and actual values.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        """
        # TODO: Implement R^2 score calculation
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
        # TODO: Implement this method
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
        # TODO: implement cross validation
        pass
    
    def feature_importance(self):
        """Return feature importance for each feature"""
        # TODO: Implement this method
        pass
    
    def predict_proba(self, X):
        """Return probability estimates for the given feature matrix
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        """
        # TODO: Implement predict_proba method
        pass
    
    def save(self, filename):
        """
        Saves the model as a Pickle file. Ensure that the destination directory exists.
    
        Note: Be cautious while loading pickled files from an untrusted source as it might be insecure.
        
        Parameters
        ----------
        filename : str
            Name of file to save model to
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        """
        Loads the model from a Pickle file.
    
        Warning: Only load Pickle files from trusted sources to avoid security risks.
        
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

    def __str__(self):
            return f"Linear Regression Model (method: {self.method}, fit_intercept: {self.fit_intercept}, learning_rate: {self.learning_rate}, epochs: {self.epochs}, regularization: {self.regularization}, alpha: {self.alpha})"

    def __repr__(self):
            return f"LinearRegression(method='{self.method}', fit_intercept={self.fit_intercept}, learning_rate={self.learning_rate}, epochs={self.epochs}, regularization={self.regularization}, alpha={self.alpha})"