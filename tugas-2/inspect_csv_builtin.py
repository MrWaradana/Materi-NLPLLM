
import csv
try:
    with open('../tugas-1/gofood_dataset.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print("Headers:", headers)
        print("First row:", next(reader))
except Exception as e:
    print(e)
