import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Importing and data Preprocessing or data cleaning
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(df.head())
    df['Description'] = df['Description'].str.strip()
    df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
    df['InvoiceNo'] = df['InvoiceNo'].astype('str')
    df = df[~df['InvoiceNo'].str.contains('C')]
    return df


# Load and preprocess data
data_file = '../PythonRepo2/OnlineRetail.csv'
data = load_and_preprocess_data(data_file)

# Generate basket for France
basket = (data[data['Country'] == "France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))

# Function to encode quantity
def encode_quantity(x):
    return 1 if x > 0 else 0

# Convert basket to binary matrix to avoid warning in output
basket_sets = basket.applymap(encode_quantity)
basket_sets.drop('POSTAGE', inplace=True, axis=1)

# Generate frequent itemsets
min_support = 0.07
frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

# Generate association rules
min_threshold = 1
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)

# Filter rules based on lift and confidence
filtered_rules = rules[(rules['lift'] >= 6) & (rules['confidence'] >= 0.8)]

print(filtered_rules.head())

# The final output includes the filtered association rules that satisfy the specified conditions.
