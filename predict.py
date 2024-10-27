import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

model = joblib.load('best_model.pkl')
test_data = pd.read_csv('test_data.csv')
numeric_features = ['Inches', 'Ram', 'Weight', 'ScreenW', 'ScreenH', 'CPU_freq']
categorical_features = ['GPU_model']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])
pipeline.fit(train_data[numeric_features + categorical_features], train_data['Price_euros'])
joblib.dump(pipeline, 'best_model.pkl')
predictions = pipeline.predict(test_data[numeric_features + categorical_features])
submission = pd.DataFrame({'Laptop_id': test_data['Laptop_id'], 'PredictedPrice': predictions})
submission.to_csv('submission.csv', index=False)
