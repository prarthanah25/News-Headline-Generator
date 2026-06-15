import pandas as pd
from transformers import pipeline
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# Load dataset
df = pd.read_csv("cleaned_news_dataset.csv")

# Load model
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

# Sample article
sample_article = df['ctext'][0]

# Generate headline
result = summarizer(
    sample_article,
    max_length=20,
    min_length=5,
    do_sample=False
)

generated_headline = result[0]['summary_text']

# Actual headline
actual_headline = df['headlines'][0]

print("Generated Headline:")
print(generated_headline)

print("\nActual Headline:")
print(actual_headline)

# ROUGE Scores
scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

scores = scorer.score(
    actual_headline,
    generated_headline
)

print("\nROUGE-1:", scores['rouge1'])
print("ROUGE-2:", scores['rouge2'])
print("ROUGE-L:", scores['rougeL'])

# BLEU Score
reference = [actual_headline.split()]
candidate = generated_headline.split()

bleu = sentence_bleu(reference, candidate)

print("\nBLEU Score:", bleu)