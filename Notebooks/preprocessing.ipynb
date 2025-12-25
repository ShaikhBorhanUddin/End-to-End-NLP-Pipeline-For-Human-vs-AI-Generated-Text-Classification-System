from google.colab import drive
drive.mount('../content/drive')

import pandas as pd
df_1 = pd.read_csv('../content/drive/My Drive/Human vs AI Generated Text Classification/1765533433232_96d2a190cc.csv')

df_1

df_1.dtypes

df_1.duplicated().sum()

There are no duplicate values in the first dataset.

df_1.isnull().sum()

343 null valus in 'notes' column.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

custom_bins = [0, 50, 100, 150, 200, 250, 300]
df_1['length_chars_binned'] = pd.cut(df_1['length_chars'], bins=custom_bins, right=False)
binned_counts = df_1['length_chars_binned'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=binned_counts.index.astype(str), y=binned_counts.values, palette='viridis', hue=binned_counts.index.astype(str), legend=False)
plt.title('Character length count of df_1')
plt.xlabel('Character Length bins')
plt.ylabel('Count')
plt.xticks(rotation=0, ha='center')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

custom_bins_words = [0, 10, 15, 20, 25, 30, 35, 40, 45]
df_1['length_words_binned'] = pd.cut(df_1['length_words'], bins=custom_bins_words, right=False)
binned_words_counts = df_1['length_words_binned'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=binned_words_counts.index.astype(str), y=binned_words_counts.values, palette='viridis', hue=binned_words_counts.index.astype(str), legend=False)
plt.title('Word count of df_1')
plt.xlabel('Word Length Bins')
plt.ylabel('Count')
plt.xticks(rotation=0, ha='center')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

label_counts = df_1['label'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
plt.title('Distribution of Labels')
plt.axis('equal')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

correlation_columns = ['length_chars', 'length_words', 'quality_score', 'sentiment', 'plagiarism_score']
df_corr = df_1[correlation_columns]
correlation_matrix = df_corr.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

The following columns were excluded from model training to prevent data leakage, reduce bias, and ensure that the classifier learned intrinsic linguistic patterns rather than relying on metadata or weak auxiliary signals. The **quality_score** column was dropped because readability-based metrics capture stylistic uniformity but show high overlap between human and AI-generated text, making them weak and non-decisive predictors. Similarly, the **sentiment** column was excluded as emotional tone exhibits very weak correlation with AI authorship and does not reliably distinguish between human and AI writing. Metadata attributes such as **topic** were removed due to the risk of shortcut learning, as topic information can indirectly reveal labels without reflecting authorship style. The **source_detail** column was excluded because it explicitly identifies content origin (e.g., human author IDs or AI model names), which would cause severe data leakage and artificially inflate model performance. The **timestamp** column was dropped as temporal information is irrelevant to linguistic structure and may introduce chronological bias. The **plagiarism** column was excluded because plagiarism indicators are not causally related to AI text generation and are often noisy or inconsistently defined. Finally, the **notes** column was removed due to its subjective and human-annotated nature, which lacks consistency and does not represent intrinsic text characteristics. These columns were retained only for exploratory data analysis, stratified sampling, and bias analysis, while the final model was trained exclusively on text-based features.

relevant_columns = ['text', 'label', 'length_chars', 'length_words']
df_1_modified = df_1[relevant_columns].copy()
df_1_modified

**punctuation_ratio** measures how much punctuation a text uses relative to its length.

Formula: punctuation_ratio = number_of_punctuation_characters / number_of_characters

Why this matters:

AI-generated text tends to use safe, grammatically “correct” punctuation, avoid expressive or erratic punctuation (!!!, ?!, —, etc.).

On the other hand humans are inconsistent, overuse or underuse punctuation, use stylistic punctuation.

This makes punctuation usage a useful stylometric signal.

**repetition_score** measures how repetitive a text is, i.e., how often words or phrases are reused. AI text repeats patterns more consistently than human text. Human writing may contain similar expressions, phrases and words over and over whereas AI generated texts use more moderated, neural texts and synonyms.

Formula: repetition_score = 1 - (unique_words / total_words)

High repetition score → fewer unique words → more repetition

Low repetition score → more lexical diversity

import string

def punctuation_ratio(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    punct = sum(1 for c in text if c in string.punctuation)
    return punct / len(text)

def repetition_score(text):
    if not isinstance(text, str):
        return 0.0
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    return 1 - len(set(words)) / len(words)

df_1_modified["punctuation_ratio"] = df_1_modified["text"].apply(punctuation_ratio)
df_1_modified["repetition_score"] = df_1_modified["text"].apply(repetition_score)

df_1_modified

df_1_modified['label'] = df_1_modified['label'].map({'human': 0, 'ai': 1}).astype(int)
df_1_modified.head()

df_1_modified.to_csv('/content/drive/My Drive/Human vs AI Generated Text Classification/df_1_modified.csv', index=False)

df_2 = pd.read_csv('/content/drive/My Drive/Human vs AI Generated Text Classification/your_dataset_5000.csv')

df_2

df_2.info()

df_2.isnull().sum()

df_2.duplicated().sum()

A very large portion of the dataset consists of repeated rows. Total duplicate rows 4540. This means only 460 rows are unique.


df_2.drop_duplicates(inplace=True)
print("Shape of df_2 after removing duplicates:", df_2.shape)
display(df_2.head())

df_2["text"] = df_2["text"].astype(str)
df_2["length_chars"] = df_2["text"].str.len()
df_2["length_words"] = df_2["text"].str.split().str.len()
df_2["punctuation_ratio"] = df_2["text"].apply(punctuation_ratio)
df_2["repetition_score"] = df_2["text"].apply(repetition_score)
df_2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

min_len = df_2['length_chars'].min()
max_len = df_2['length_chars'].max()

# Create custom bins that cover the actual range of data, with an interval of 50
custom_bins = np.arange(0, max_len + 50, 50).tolist()

df_2['length_chars_binned'] = pd.cut(df_2['length_chars'], bins=custom_bins, right=False)
binned_counts = df_2['length_chars_binned'].value_counts().sort_index()

# Filter out bins with zero counts
binned_counts = binned_counts[binned_counts > 0]

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=binned_counts.index.astype(str), y=binned_counts.values, palette='viridis', hue=binned_counts.index.astype(str), legend=False)
plt.title('Character length count of df_2')
plt.xlabel('Character Length Bins')
plt.ylabel('Count')
plt.xticks(rotation=0, ha='center')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')
plt.tight_layout()
plt.show()

df_2_modified = df_2.copy()
df_2_modified.head()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

min_len_words = df_2['length_words'].min()
max_len_words = df_2['length_words'].max()

# Create custom bins that cover the actual range of data, with an interval of 10 for words
custom_bins_words = np.arange(0, max_len_words + 10, 10).tolist()

df_2['length_words_binned'] = pd.cut(df_2['length_words'], bins=custom_bins_words, right=False)
binned_words_counts = df_2['length_words_binned'].value_counts().sort_index()

# Filter out bins with zero counts
binned_words_counts = binned_words_counts[binned_words_counts > 0]

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=binned_words_counts.index.astype(str), y=binned_words_counts.values, palette='viridis', hue=binned_words_counts.index.astype(str), legend=False)
plt.title('Word count of df_2')
plt.xlabel('Word Length Bins')
plt.ylabel('Count')
plt.xticks(rotation=0, ha='center')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

label_counts_df2 = df_2['label'].value_counts()
label_names = {0: 'human', 1: 'ai'}
plt.figure(figsize=(8, 8))
plt.pie(label_counts_df2, labels=label_counts_df2.index.map(label_names), autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
plt.title('Distribution of Labels in df_2')
plt.axis('equal')
plt.show()

df_2_modified = df_2_modified.drop(columns=['length_chars_binned', 'length_words_binned'], errors='ignore')
df_2_modified.head()

df_2_modified.to_csv('/content/drive/My Drive/Human vs AI Generated Text Classification/df_2_modified.csv', index=False)

df_3 = pd.read_csv('/content/drive/My Drive/Human vs AI Generated Text Classification/AI_Human_balanced_dataset.csv')

df_3

df_3.info()

df_3 = df_3.rename(columns={'generated': 'label'})

df_3

df_3['label'] = df_3['label'].astype(int)
print("DataFrame df_3 info after converting 'label' to int:")
df_3.info()

df_3

df_3.isnull().sum()

df_3.duplicated().sum()

df_3 is too large to handle and time consuming. So, 20k entries from df_3 (10k human and 10k AI generated) is taken before proceeding. It is named as df_3_truncated.

# Separate df_3 into two dataframes based on label
df_3_label_0 = df_3[df_3['label'] == 0]
df_3_label_1 = df_3[df_3['label'] == 1]

# Sample 10,000 entries from each label
sample_size = 10000
df_3_sampled_0 = df_3_label_0.sample(n=min(len(df_3_label_0), sample_size), random_state=42)
df_3_sampled_1 = df_3_label_1.sample(n=min(len(df_3_label_1), sample_size), random_state=42)

# Concatenate the sampled dataframes to create df_3_truncated
df_3_truncated = pd.concat([df_3_sampled_0, df_3_sampled_1])

# Shuffle the new dataframe to mix the labels
df_3_truncated = df_3_truncated.sample(frac=1, random_state=42).reset_index(drop=True)

print("Shape of df_3_truncated:", df_3_truncated.shape)
display(df_3_truncated.head())

Sentence aware truncation:

This is to ensure maximum text length is 512 tokens and there are no broken sententence in text column.

import nltk
nltk.download('punkt_tab')

import nltk
nltk.download('punkt')


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


AI vs hman text classification relies mostly on writing style, repetation, sentence structure, lexical and error patterns (humans make incosistent expressions). In practice the first 200 - 400 tokens already contain enough signal. The rest is often redundant stylistically. So, 512 tokens are often enough for text classification and hence sentence aware text truncation does not destroy the work.

import string
import nltk

def truncate_text(text, max_tokens=512):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    sentences = nltk.sent_tokenize(text)
    truncated = []
    total_tokens = 0

    for sent in sentences:
        # Encode sentence WITHOUT special tokens
        sent_tokens = len(
            tokenizer.encode(sent, add_special_tokens=False)
        )

        if total_tokens + sent_tokens > max_tokens:
            break

        truncated.append(sent)
        total_tokens += sent_tokens

    return " ".join(truncated).strip()

df_3_truncated['text'] = df_3_truncated['text'].apply(truncate_text)
display(df_3_truncated.head())

df_3_truncated["text"] = df_3_truncated["text"].astype(str)
df_3_truncated["length_chars"] = df_3_truncated["text"].str.len()
df_3_truncated["length_words"] = df_3_truncated["text"].str.split().str.len()
df_3_truncated["punctuation_ratio"] = df_3_truncated["text"].apply(punctuation_ratio)
df_3_truncated["repetition_score"] = df_3_truncated["text"].apply(repetition_score)
df_3_truncated

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

min_len = df_3_truncated['length_chars'].min()
max_len = df_3_truncated['length_chars'].max()

# Create custom bins that cover the actual range of data, with an interval of 50
custom_bins = np.arange(0, max_len + 50, 500).tolist()

df_3_truncated['length_chars_binned'] = pd.cut(df_3_truncated['length_chars'], bins=custom_bins, right=False)
binned_counts = df_3_truncated['length_chars_binned'].value_counts().sort_index()

# Filter out bins with zero counts
binned_counts = binned_counts[binned_counts > 0]

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=binned_counts.index.astype(str), y=binned_counts.values, palette='viridis', hue=binned_counts.index.astype(str), legend=False)
plt.title('Character length count of df_3_truncated')
plt.xlabel('Character Length Bins')
plt.ylabel('Count')
plt.xticks(rotation=0, ha='center')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

min_len_words = df_3_truncated['length_words'].min()
max_len_words = df_3_truncated['length_words'].max()

# Create custom bins that cover the actual range of data, with an interval of 10 for words
custom_bins_words = np.arange(0, max_len_words + 10, 50).tolist()

df_3_truncated['length_words_binned'] = pd.cut(df_3_truncated['length_words'], bins=custom_bins_words, right=False)
binned_words_counts = df_3_truncated['length_words_binned'].value_counts().sort_index()

# Filter out bins with zero counts
binned_words_counts = binned_words_counts[binned_words_counts > 0]

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=binned_words_counts.index.astype(str), y=binned_words_counts.values, palette='viridis', hue=binned_words_counts.index.astype(str), legend=False)
plt.title('Word count of df_3_truncated')
plt.xlabel('Word Length Bins')
plt.ylabel('Count')
plt.xticks(rotation=0, ha='center')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')
plt.tight_layout()
plt.show()

df_3_truncated.to_csv('/content/drive/My Drive/Human vs AI Generated Text Classification/df_3_truncated.csv', index=False)

df_combined = pd.concat([df_1_modified, df_2_modified, df_3_truncated], ignore_index=True)
df_combined

df_combined = df_combined.drop(columns=['length_chars_binned', 'length_words_binned'])
df_combined

df_combined.duplicated().sum()

df_combined.drop_duplicates(inplace=True)

print("Shape of df_combined after removing duplicates:", df_combined.shape)

df_combined.to_csv('/content/drive/My Drive/Human vs AI Generated Text Classification/df_combined.csv', index=False)

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_combined["text"] = df_combined["text"].apply(clean_text)

This step normalizes the raw text to make it suitable for TF-IDF feature extraction. The **clean_text()** function converts all text to lowercase, removes punctuation and other non-alphanumeric characters using regular expressions, and standardizes spacing by collapsing multiple spaces into one and trimming leading or trailing whitespace.

df_combined

Lemmatization using spaCy is not strictly necessary in all text classification tasks, but it can be very beneficial, especially for distinguishing between human and AI-generated text.

Here's why it's generally a good idea for this type of task:

Reduces Dimensionality:

It reduces words to their base or dictionary form (lemma). For example, 'running', 'runs', and 'ran' all become 'run'. This reduces the total number of unique tokens in vocabulary, which can simplify model and prevent overfitting.

Improves Feature Representation:

By grouping different inflections of a word, lemmatization helps model treat them as the same concept. This can improve the quality of features extracted (e.g., for TF-IDF or word embeddings), as 'good' and 'better' are recognized as related to 'well'.

Focuses on Semantic Meaning:

It allows the model to focus more on the core meaning of words rather than their grammatical variations. This can be crucial for style analysis, where the semantic content might be similar but the stylistic choices differ.

Potential for Better Accuracy:

By normalizing word forms, lemmatization can lead to better generalization and potentially higher accuracy for classification model, especially if the differences between human and AI text are subtle and relate to core vocabulary usage.

!pip install spacy

!pip install -U spacy
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

import spacy

nlp = spacy.load('en_core_web_sm')

def lemmatize_texts(texts, batch_size=1000):
    lemmatized = []
    for doc in nlp.pipe(
        texts,
        batch_size=batch_size,
        disable=["ner", "parser"]
    ):
        lemmatized.append(" ".join(token.lemma_ for token in doc))
    return lemmatized

df_combined['text_lemmatized'] = lemmatize_texts(df_combined['text'].tolist())

df_combined

df_combined_lemmatized = df_combined.copy(deep=True)
df_combined_lemmatized.head()

df_combined_lemmatized = df_combined_lemmatized.drop(columns=['text'])
df_combined_lemmatized = df_combined_lemmatized.rename(columns={'text_lemmatized': 'text'})
df_combined_lemmatized.head()

df_combined_lemmatized.info()

df_combined_lemmatized.head()

df_combined_lemmatized.to_csv('/content/drive/My Drive/Human vs AI Generated Text Classification/df_combined_lemmatized.csv', index=False)
