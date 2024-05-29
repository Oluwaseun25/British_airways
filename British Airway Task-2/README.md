# British Airways Task: Predictive Model to Understand Factors that Influence Buying Behaviour

## Project Overview

This project focuses on building a predictive model to understand the factors that influence customer buying behavior for British Airways. The goal is to predict customer bookings using various features such as the number of passengers, sales channel, trip type, and other relevant variables.

## Tools and Libraries Used

- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical computations
- **Scikit-learn**: For machine learning tasks, including feature selection, scaling, and modeling
- **SMOTE**: For oversampling to balance the dataset
- **Matplotlib** and **Seaborn**: For data visualization
- **Jupyter Notebook**: For interactive coding and analysis

## Dataset Description

The dataset contains the following columns:
- `num_passengers`: Number of passengers
- `sales_channel`: Sales channel used for booking
- `trip_type`: Type of trip (e.g., one-way, round trip)
- `purchase_lead`: Days between purchase and flight
- `length_of_stay`: Duration of stay at the destination
- `flight_hour`: Hour of the flight
- `flight_day`: Day of the flight
- `route`: Flight route
- `booking_origin`: Origin of the booking
- `wants_extra_baggage`: Indicator if extra baggage is wanted
- `wants_preferred_seat`: Indicator if a preferred seat is wanted
- `wants_in_flight_meals`: Indicator if in-flight meals are wanted
- `flight_duration`: Duration of the flight
- `booking_complete`: Target variable indicating if the booking was completed

## Project Steps

### 1. Exploratory Data Analysis (EDA)
First, we explore the data to understand its structure and statistical properties.

#### Example Code Snippet:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('path_to_dataset.csv')

# Display basic statistics
print(df.describe())

# Visualize distribution of target variable
sns.countplot(x='booking_complete', data=df)
plt.title('Distribution of Booking Completion')
plt.show()
```

### 2. Encode Categorical Variables
Categorical variables are encoded to numeric values for machine learning models.

#### Example Code Snippet:
```python
from sklearn.preprocessing import LabelEncoder

categorical_features = ['sales_channel', 'trip_type', 'route', 'booking_origin']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le
```

### 3. Feature Selection and Mutual Information
We use mutual information to select important features.

#### Example Code Snippet:
```python
from sklearn.feature_selection import mutual_info_classif

X = df.drop('booking_complete', axis=1)
y = df['booking_complete']

# Calculate mutual information
mi = mutual_info_classif(X, y)
mi_series = pd.Series(mi, index=X.columns)
mi_series.sort_values(ascending=False)
```


### 4. Data Scaling
We scale the data to handle outliers and ensure all features contribute equally to the model.

#### Example Code Snippet:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 5. Balancing Dataset
We use SMOTE to oversample the minority class and balance the dataset.

#### Example Code Snippet:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_res, y_res = smote.fit_resample(X_scaled, y)
```

### 6. Model Training and Evaluation
We train several models and evaluate their performance.

#### Example Code Snippet:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Logistic Regression:', accuracy_score(y_test, y_pred_lr))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print('Decision Tree:', accuracy_score(y_test, y_pred_dt))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Random Forest:', accuracy_score(y_test, y_pred_rf))

# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print('XGBoost:', accuracy_score(y_test, y_pred_xgb))

# Evaluate the best model (XGBoost) using precision, recall, f1 score, and AUC
print('XGBoost Performance:')
print('Accuracy:', accuracy_score(y_test, y_pred_xgb))
print('Precision:', precision_score(y_test, y_pred_xgb))
print('Recall:', recall_score(y_test, y_pred_xgb))
print('F1 Score:', f1_score(y_test, y_pred_xgb))
print('AUC Score:', roc_auc_score(y_test, y_pred_xgb))
```

<!-- ![Mutal Score and best performance metrics](C:\Users\USER\Documents\Work\BA\data\British Airway Task-2\Mutal Score and best performance metrics.png) -->

### 7. Feature Importance
We analyze the feature importance to understand which features contribute most to the prediction.

#### Example Code Snippet:
```python
importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.show()
```

<!-- ![Important features and best performance metrics](C:\Users\USER\Documents\Work\BA\data\British Airway Task-2/Important features and best performance metrics.png) -->

## Conclusion
This project successfully builds a predictive model to understand factors influencing British Airways customer bookings. The XGBoost model performed best in terms of accuracy, precision, f1 score, and AUC score. The most important features influencing the booking completion are `purchase_lead`, `length_of_stay`, `flight_hour`, `flight_day`, `route`, and `booking_origin`.

## How to Run the Project
1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
   ```
2. **Run the Jupyter Notebook**: Open and run the notebook `british_airways_task.ipynb` in Jupyter.

## License
This project is licensed under the MIT License.