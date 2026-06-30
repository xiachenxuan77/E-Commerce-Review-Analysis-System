# =========================
# Data Processing
# =========================

import pandas as pd

# NLP
import nltk
import re
import string

from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLP resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def sentiment_label(rating):

    if rating >= 4:
        return "Positive"

    elif rating == 3:
        return "Neutral"

    else:
        return "Negative"


stop_words = set(stopwords.words('english'))
negation_words = {
    "not",
    "no",
    "never",
    "dont",
    "didnt",
    "isnt",
    "wasnt",
    "cant",
    "but",
    "however",
    "although",
    "yet"
}

stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    if pd.isna(text):
        return ""

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r'http\S+|www\S+', '', text)

    # remove html
    text = re.sub(r'<.*?>', '', text)

    # remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # tokenization
    tokens = text.split()

    # stopword removal + lemmatization
    cleaned_tokens = []

    for word in tokens:

        if word not in stop_words:

            lemma = lemmatizer.lemmatize(word)

            if len(lemma) > 1:
                cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)



def main():

    # =========================
    # Load Dataset
    # =========================

    df = pd.read_csv(
        "amazon_reviews.csv",
        engine="python"
    )

    print("Dataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # =========================
    # Missing Values
    # =========================

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Values': missing.values,
        'Missing Percentage': (
            missing.values / len(df) * 100
        ).round(2)
    })

    missing_df = missing_df.sort_values(
        by='Missing Values',
        ascending=False
    )

    print("\nMissing Values:")
    print(missing_df.head(20))

    # =========================
    # Category Statistics
    # =========================

    print("\nNumber of Categories:")
    print(df['category'].nunique())

    print("\nTop 20 Categories:")
    print(df['category'].value_counts().head(20))

    print("\nRating Distribution:")
    print(df['rating'].value_counts().sort_index())

    # =========================
    # Select Required Columns
    # =========================

    df = df[
        [
            'reviewText',
            'rating',
            'category',
            'brand',
            'reviewTime'
        ]
    ]

    print("\nAfter Column Selection:")
    print(df.shape)

    # =========================
    # Handle Missing Values
    # =========================

    df = df.dropna(
        subset=['reviewText']
    )

    print("\nAfter Removing Missing Reviews:")
    print(df.shape)

    df['brand'] = df['brand'].fillna('Unknown')

    # =========================
    # Remove Duplicate Reviews
    # =========================

    before = len(df)

    df = df.drop_duplicates(
        subset=['reviewText']
    )

    after = len(df)

    print("\nDuplicates Removed:", before - after)
    print("Remaining Reviews:", after)



    # =========================
    # Sample Cleaning Demo
    # =========================

    sample = df['reviewText'].iloc[0]

    print("\nORIGINAL REVIEW:")
    print(sample)

    print("\nCLEANED REVIEW:")
    print(clean_text(sample))

    # =========================
    # Clean All Reviews
    # =========================

    tqdm.pandas()

    df['cleaned_review'] = (
        df['reviewText']
        .progress_apply(clean_text)
    )
    # =========================
    # Remove Empty Cleaned Reviews
    # =========================

    before = len(df)

    df = df[df["cleaned_review"].str.strip() != ""].copy()

    after = len(df)

    print(f"\nRemoved {before - after} empty cleaned reviews.")
    print(f"Remaining Reviews: {after}")

    
    before = len(df)

    df = df[df["cleaned_review"].str.strip() != ""].copy()

    after = len(df)

    print(f"\nRemoved {before-after} empty cleaned reviews.")
    
    # =========================
    # Review Length
    # =========================

    df['review_length'] = (
        df['cleaned_review']
        .str.split()
        .str.len()
    )

    # =========================
    # Generate Initial Sentiment Labels
    # =========================

    df["sentiment_label"] = df["rating"].apply(sentiment_label)

    print("\nInitial Sentiment Distribution:")
    print(df["sentiment_label"].value_counts())

    # =========================
    # Stratified Sampling (80k)
    # =========================

    TARGET_SIZE = 80000

    if len(df) > TARGET_SIZE:

        sampled_df = []

        for label, group in df.groupby("sentiment_label"):

            n_samples = round(len(group) / len(df) * TARGET_SIZE)
 
            sampled_group = group.sample(
                n=n_samples,
                random_state=42)

            sampled_df.append(sampled_group)

        df = (
            pd.concat(sampled_df).sample(frac=1, random_state=42).reset_index(drop=True))


        if len(df) > TARGET_SIZE:
            df = df.sample(
            n=TARGET_SIZE,
            random_state=42
        ).reset_index(drop=True)

        print("\nAfter Stratified Sampling:")
        print(df.shape)

        print("\nSentiment Distribution:")
        print(df["sentiment_label"].value_counts(normalize=True))

    else:

        print("\nDataset contains fewer than 80,000 reviews.")

    # =========================
    # Rename Columns
    # =========================

    df = df.rename(
        columns={
            'reviewText': 'original_review',
            'category': 'product_category',
            'reviewTime': 'review_time'
        }
    )

    # =========================
    # Add Review ID
    # =========================

    df = df.reset_index(drop=True)

    df.insert(
        0,
        'review_id',
        range(1, len(df) + 1)
    )

    # =========================
    # Final Dataset
    # =========================

    final_df = df[
    [
        'review_id',
        'original_review',
        'cleaned_review',
        'rating',
        'sentiment_label',
        'product_category',
        'brand',
        'review_length',
        'review_time'
    ]
]

    print("\nFinal Columns:")
    print(final_df.columns.tolist())

    print("\nFinal Dataset Shape:")
    print(final_df.shape)

    # =========================
    # Dataset Statistics
    # =========================

    total_reviews = len(final_df)

    positive_reviews = (
        final_df['sentiment_label']
        .eq('Positive')
        .sum()
    )

    neutral_reviews = (
        final_df['sentiment_label']
        .eq('Neutral')
        .sum()
    )

    negative_reviews = (
        final_df['sentiment_label']
        .eq('Negative')
        .sum()
    )

    avg_review_length = (
        final_df['review_length']
        .mean()
    )

    num_categories = (
        final_df['product_category']
        .nunique()
    )

    num_brands = (
        final_df['brand']
        .nunique()
    )

    print("\n========== DATASET SUMMARY ==========")
    print(f"Total Reviews: {total_reviews}")
    print(f"Positive Reviews: {positive_reviews}")
    print(f"Neutral Reviews: {neutral_reviews}")
    print(f"Negative Reviews: {negative_reviews}")
    print(f"Average Review Length: {avg_review_length:.2f}")
    print(f"Number of Categories: {num_categories}")
    print(f"Number of Brands: {num_brands}")
    print("Train/Test Split Ratio: 80:20")

    # Remove Reviews with Empty Text
    # Original review is used first (RoBERTa),
    # cleaned review is used as a fallback.

    def normalize_text(text):

        if pd.isna(text):
            return ""

        text = str(text)

        # remove html
        text = re.sub(r"<.*?>", " ", text)

        # remove url
        text = re.sub(r"http\S+|www\S+", " ", text)

        # normalize whitespace
        return " ".join(text.split())


    # Use original review first
    final_df["model_input"] = (
        final_df["original_review"]
        .fillna("")
        .apply(normalize_text)
)

    # If original review is empty, use cleaned_review
    final_df["model_input"] = final_df["model_input"].where(
    final_df["model_input"].str.strip() != "",
    final_df["cleaned_review"].fillna("")
)

    # Remove reviews with no remaining text
    final_df = final_df[
    final_df["model_input"].str.strip() != ""
].copy()

    # Remove temporary column
    final_df.drop(columns=["model_input"], inplace=True)

    print("\nAfter Empty Review Filtering:")
    print(final_df.shape)

    # =========================
    # Save Final Corpus
    # =========================

    final_df.to_csv(
        "final_corpus.csv",
        index=False
    )

    print("\nSaved Successfully:")
    print("final_corpus.csv")


if __name__ == "__main__":
    main()



