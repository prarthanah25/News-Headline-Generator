import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("cleaned_news_dataset.csv")

# Article length
df['article_length'] = df['ctext'].astype(str).apply(lambda x: len(x.split()))

# Headline length
df['headline_length'] = df['headlines'].astype(str).apply(lambda x: len(x.split()))

# =====================================
# HISTOGRAM
# =====================================

plt.figure(figsize=(10,6))
plt.hist(df['article_length'], bins=30)

plt.title("Distribution of Article Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.show()

# =====================================
# BOXPLOT
# =====================================

plt.figure(figsize=(8,5))
plt.boxplot(df['article_length'])

plt.title("Box Plot of Article Length")
plt.ylabel("Word Count")

plt.show()

# =====================================
# HEATMAP
# =====================================

numeric_data = df[['headline_length', 'article_length']]

corr = numeric_data.corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()