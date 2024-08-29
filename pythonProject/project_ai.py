import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Create a directory to save plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Custom Linear Regression Implementation
class CustomLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

# Custom Neural Network Implementation
class CustomNeuralNetwork:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias

            # Calculate the gradients
            dw = (1 / n_samples) * np.dot(X.T, (linear_model - y))
            db = (1 / n_samples) * np.sum(linear_model - y)

            # Update the weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Load dataset
df = pd.read_csv('housing_price_dataset.csv')

# Feature Engineering
df['TotalRooms'] = df['Bedrooms'] + df['Bathrooms']
df['Age'] = 2024 - df['YearBuilt']

# Define features and target variable
X = df.drop('Price', axis=1)
y = df['Price']

# Preprocess categorical and numerical features
numeric_features = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'TotalRooms', 'Age']
categorical_features = ['Neighborhood']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Models
dummy_mean = DummyRegressor(strategy='mean')
dummy_median = DummyRegressor(strategy='median')

# Train and Evaluate Baseline Models
dummy_mean.fit(X_train, y_train)
dummy_median.fit(X_train, y_train)

y_pred_mean = dummy_mean.predict(X_test)
y_pred_median = dummy_median.predict(X_test)

print("Mean Predictor MAE:", mean_absolute_error(y_test, y_pred_mean))
print("Median Predictor MAE:", mean_absolute_error(y_test, y_pred_median))

# Define models
models = {
    'Custom Linear Regression': CustomLinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Custom Neural Network': CustomNeuralNetwork(n_iterations=1000, learning_rate=0.01),
}

# Train models and get predictions
results = {}
predictions = pd.DataFrame({'Actual Price': y_test})

for name, model in models.items():
    # Preprocess data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    print(f'Training {name}...')
    model.fit(X_train_preprocessed, y_train)

    # Predictions
    y_pred = model.predict(X_test_preprocessed)

    if name in ['Custom Linear Regression', 'Custom Neural Network']:
        predictions[name] = y_pred

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
    print(f'{name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}')

# Save the best model based on RMSE
best_model_name = min(results, key=lambda k: results[k]['RMSE'])
best_model = models[best_model_name]

# Save the best model
joblib.dump(best_model, 'best_model_custom.pkl')

# Load the model
loaded_model = joblib.load('best_model_custom.pkl')

# Function to predict new data
def predict_new_data(model, new_data):
    new_data_preprocessed = preprocessor.transform(new_data)
    return model.predict(new_data_preprocessed)

# Example of new data prediction
new_data = pd.DataFrame({
    'SquareFeet': [2500],
    'Bedrooms': [3],
    'Bathrooms': [2],
    'Neighborhood': ['Urban'],
    'YearBuilt': [2005],
    'TotalRooms': [5],  # Adding this for completeness
    'Age': [2024 - 2005]  # Adding this for completeness
})

predicted_price = predict_new_data(loaded_model, new_data)
print(f'Predicted price: {predicted_price}')

# Plotting model performance
model_names = list(results.keys())
mae_scores = [results[name]['MAE'] for name in model_names]
mse_scores = [results[name]['MSE'] for name in model_names]
rmse_scores = [results[name]['RMSE'] for name in model_names]

plt.figure(figsize=(14, 7))
plt.subplot(1, 3, 1)
sns.barplot(x=model_names, y=mae_scores)
plt.title('Mean Absolute Error')
plt.xticks(rotation=45)
plt.savefig('plots/mean_absolute_error.png')

plt.subplot(1, 3, 2)
sns.barplot(x=model_names, y=mse_scores)
plt.title('Mean Squared Error')
plt.xticks(rotation=45)
plt.savefig('plots/mean_squared_error.png')

plt.subplot(1, 3, 3)
sns.barplot(x=model_names, y=rmse_scores)
plt.title('Root Mean Squared Error')
plt.xticks(rotation=45)
plt.savefig('plots/root_mean_squared_error.png')

plt.tight_layout()
plt.savefig('plots/model_performance.png')
plt.show()

# Validate the model with cross-validation
cv_scores = cross_val_score(Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)]), X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
print(f'Cross-validated RMSE: {-np.mean(cv_scores)}')

# Save predictions to CSV
predictions.to_csv('model_predictions.csv', index=False)
print(predictions.head())
