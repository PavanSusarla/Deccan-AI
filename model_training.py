from sklearn.ensemble import RandomForestRegressor
import joblib
from src.data_preprocessing import load_data

X_train, X_test, y_train, y_test = load_data()

# Train Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "models/house_price_model.pkl")
print("Model saved successfully!")