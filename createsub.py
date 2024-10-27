import pandas as pd

submission_df = pd.read_csv('submission.csv') 
submission_format = submission_df[['Laptop_id', 'Price_euros']].copy()
submission_format.rename(columns={'Price_euros': 'Prices_euros'}, inplace=True)
submission_format['Prices_euros'] = submission_format['Prices_euros'].round(6)
submission_format.to_csv('submission.csv', index=False)
print("Final submission file created: submission.csv")
