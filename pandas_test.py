# Import pandas package
import pandas as pd

# Define a dictionary containing employee data
data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj', 'Geeku'],
        'Age': [27, 24, 22, 32, 15],
        'Address': ['Delhi', 'Kanpur', 'Allahabad', 'Kannauj', 'Noida'],
        'Qualification': ['Msc', 'MA', 'MCA', 'Phd', '10th']}

# Convert the dictionary into DataFrame
df = pd.DataFrame(data)

# select all columns
print(df)

df = df.sample(frac=1)
print(df)