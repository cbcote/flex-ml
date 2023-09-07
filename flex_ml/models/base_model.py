class BaseModel:
    def fit(X, y):
        raise NotImplementedError("fit method should be overridden by a subclass")
    
    def predict(self, X):
        raise NotImplementedError("predict method should be overridden by a subclass")