import pandas as pd

b = pd.read_csv('data/processed/business_filtered.csv', nrows=5)
print('Sample categories:')
for idx, cat in enumerate(b['categories'].head()):
    print(f'{idx+1}. {cat}')







