
import pandas as pd
try:
    df = pd.read_csv('../tugas-1/gofood_dataset.csv')
    print("Columns:", df.columns.tolist())
    print("First row:", df.iloc[0].to_dict())
except Exception as e:
    print(e)
