from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from src.data_preprocessing import load_data

X_train, X_test, y_train, y_test = load_data()

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring="neg_mean_squared_error")
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)