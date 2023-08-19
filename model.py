from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

class Modelling():
    def __init__(self, X_train, y_train) -> None:
        # Define parameter grid to search over
        self.param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt']
        }
        self.X_train = X_train
        self.y_train = y_train
    
    def perform_hyperparameter_tuning(self):
        """
        performs hyperparameter tuning on the dataset
        """

        # Initialize a RandomForestClassifier
        model = RandomForestClassifier()

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=self.param_grid, cv=5, scoring='accuracy')

        # Fit the GridSearchCV object to your data
        grid_search.fit(self.X_train, self.y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        return best_params
    
    def model_data_with_best_param(self, best_params:dict):
        """
        model data with best parameters
        """
        # Initialize a RandomForestClassifier
        model = RandomForestClassifier(**best_params)

        # Fit the model to your data
        model.fit(self.X_train, self.y_train)

        return model
    
    def make_predictions(self, model, X_test):
        """
        make predictions on the test data
        """
        # Make predictions on the test data
        predictions = model.predict(X_test)

        return predictions
