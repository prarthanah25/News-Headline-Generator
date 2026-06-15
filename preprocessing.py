import pandas as pd

# Load dataset
df = pd.read_csv("news_summary.csv", encoding="ISO-8859-1")

# Keep useful columns
df = df[['ctext', 'headlines']]

# Remove null values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert to lowercase
df['ctext'] = df['ctext'].str.lower()
df['headlines'] = df['headlines'].str.lower()

# Remove spaces
df['ctext'] = df['ctext'].str.strip()
df['headlines'] = df['headlines'].str.strip()

# Save cleaned dataset
df.to_csv("cleaned_news_dataset.csv", index=False)

print("Dataset Preprocessed Successfully")