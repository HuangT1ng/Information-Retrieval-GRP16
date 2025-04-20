import pandas as pd
import json
import spacy
import nltk
import re
import fasttext
import string
import contractions
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)


counter = 0

def increment_counter():
    global counter 
    counter += 1 


class DataProcessorNew:
    def __init__(self,
                 acronym_file='settings/acronyms.json',
                 negators_file='settings/negators.txt',
                 duplicate_threshold=0.9
                 ):
        self.lemmatizer = WordNetLemmatizer()
        self.translator = GoogleTranslator(source='auto', target='en')
        with open(negators_file, 'r') as file:
            self.negators = set(line.strip() for line in file)
        self.stop_words = set(stopwords.words('english')) - self.negators
        self.language_detector = fasttext.load_model('models/lid.176.bin')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.duplicate_threshold = duplicate_threshold
        with open(acronym_file, 'r') as f:
            self.acronyms = json.load(f)


    def load_dataframe(self, path):
        data = []
        with open(path, 'r') as file:
            for i, line in enumerate(file): 
                data.append(json.loads(line.strip()))

        data_df = pd.DataFrame(data)

        return data_df
    

    def contains_words(self, review):
        # Check if the review contains any words
        return bool(re.search(r'[a-zA-Z]+', review))
 

    def is_valid_review(self, review):
        # Ignore reviews that are empty or not a valid string
        if not isinstance(review, str) or review.strip() == "" or not self.contains_words(review) or review == None:
            print("Not a valid review!")
            return False
        return True
    

    def expand_acronyms(self, text):
        # Expand common acronyms
        words = text.split()
        expanded_words = [self.acronyms.get(word, word) for word in words]

        return " ".join(expanded_words)
    

    def clean_text(self, review):
        # Convert to lowercase
        review = review.lower()

        # Remove html tags
        html_pattern = r'<.*?>' 
        review = re.sub(html_pattern, '', review)

        # Remove numbers
        number_pattern = r'\d+'
        review = re.sub(number_pattern, '', review)

        # Expand acryonyms
        review = self.expand_acronyms(review)

        # Add space after full stop
        review = re.sub(r'(\w)\.(\w)', r'\1. \2', review)
        review = re.sub(r'([a-zA-Z])\\/([a-zA-Z])', r'\1\/ \2', review)

        # Standardise apostrophe
        review = review.replace("\u2019", "'")

        # Replace backslashed / with /
        review = review.replace("\\/", "/")

        return review
    

    def detect_language(self, review):
        try:
            label, _ = self.language_detector.predict(review)
            return label[0].replace('__label__', '') if label else 'unknown'

        except Exception as e:
            print(f"Error: {e}")
            return 'unknown'
    

    def split_text(self, text, max_chars=5000):
        sentences = re.split(r'(?<=\.)\s+', text)  # Split after full stops followed by spaces
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:  # +1 for space
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


    def translate_text(self, review):
        if len(review) > 5000:
            chunks = self.split_text(review)
            translated_chunks = [self.translator.translate(chunk) for chunk in chunks]
            return " ".join(translated_chunks)
        else:
            translation = self.translator.translate(review)
            return translation
    

    def remove_punctuation(self, review):
        words = review.split()
        cleaned_words = [word.strip(string.punctuation) for word in words]
        return " ".join(cleaned_words)
    

    def remove_single_letter(self, review):
        words = review.split()
        cleaned_words = [word for word in words if len(word) != 1]
        return " ".join(cleaned_words)
    

    def clean_after_translation(self, review):
        print(review)
        # Remove punctuation
        review = self.remove_punctuation(review)
        print("Removed punctuation.")

        # Split contractions
        review = contractions.fix(review)

        # Perform tokenization
        tokens = word_tokenize(review)
        print("Tokenized words.")

        # Remove stop words which are unimportant (e.g. "the", "is", "and")
        tokens = [word for word in tokens if word not in self.stop_words]
        print("Removed stop words.")

        if not self.is_valid_review(" ".join(tokens)):
            return ""

        # Lemmatize words to convert words to their root form to reduce the vocabulary dimensionality
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        print("Lemmatized words")
        
        # Remove extra punctuation after lemmatizing, if any
        review = " ".join(tokens)
        review = self.remove_punctuation(review)
        print("Removed punctuation.")

        # Remove single letters, if any appears after tokenization
        review = self.remove_single_letter(review)
        print("Removed single letters.")

        print(counter)
        increment_counter()

        return review
    

    def process_text(self, reviews_df):
        # Remove invalid reviews (null, non-strings, no words)
        reviews_df = reviews_df.loc[reviews_df['text'].apply(self.is_valid_review)]
        reviews_df = reviews_df.loc[reviews_df['text'].notna()].reset_index(drop=True)
        print("Removed invalid reviews.")

        # Clean the text (remove html, numbers, convert to lowercase, add spacers)
        reviews_df['text'] = reviews_df['text'].apply(self.clean_text)
        reviews_df = reviews_df.loc[reviews_df['text'].apply(self.is_valid_review)]
        reviews_df = reviews_df.loc[reviews_df['text'].notna()].reset_index(drop=True)

        # Perform language detection. This prevents from incorrectly processing non-english text and english text.
        reviews_df.loc[:, 'detected_language'] = reviews_df['text'].apply(self.detect_language)
        print("Successfully detected languages.")

        # Translate non-english texts to english for tokenization.
        non_english_count = (reviews_df['detected_language'] != 'en').sum()
        print(f"Number of non-English reviews: {non_english_count}")
        tqdm.pandas()
        reviews_df['translated_text'] = reviews_df.progress_apply(
            lambda row: self.translate_text(row['text']) if row['detected_language'] != 'en' else row['text'],
            axis=1
        )
        print("Successfully translated all reviews.")

        # Clean the text again
        reviews_df['cleaned_text'] = reviews_df['translated_text'].apply(self.clean_after_translation)
        reviews_df = reviews_df.loc[reviews_df['cleaned_text'].apply(self.is_valid_review)]
        reviews_df = reviews_df.loc[reviews_df['cleaned_text'].notna()].reset_index(drop=True)
        print("Removed invalid reviews.")

        return reviews_df
    

    def analyze_sentiment(self, text):
        if not text:
            return 'neutral', 0.0

        scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = scores['compound']

        if compound_score >= 0.05:
            sentiment = 'positive'
        elif compound_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return sentiment, compound_score
    

    def balance_sentiment(self, reviews_df, target_ratio=0.5):
        # Count the occurrences of each sentiment label
        sentiment_distribution = reviews_df['sentiment'].value_counts()
        print(f"Original sentiment distribution: {sentiment_distribution}")

        # Exclude neutral sentiments if present
        if 'neutral' in sentiment_distribution:
            print("Excluding neutral sentiment for balancing...")
            df_filtered = reviews_df[reviews_df['sentiment'] != 'neutral']
            sentiment_distribution = df_filtered['sentiment'].value_counts()
        else:
            df_filtered = reviews_df

        # If the data is already balanced (within 5%) or too small to adjust, return the original DataFrame
        if len(sentiment_distribution) < 2 or abs(sentiment_distribution.get('positive', 0) / len(df_filtered) - target_ratio) < 0.05:
            print("The dataset is already balanced or too small to balance.")
            return reviews_df

        # Identify the majority and minority sentiment classes
        majority_class = sentiment_distribution.idxmax()
        minority_class = sentiment_distribution.idxmin()

        # Calculate the desired number of samples for the majority class
        target_majority_size = int(sentiment_distribution[minority_class] / target_ratio * (1 - target_ratio))

        # Sample from the majority class to match the desired size
        majority_samples = df_filtered[df_filtered['sentiment'] == majority_class].sample(
            n=min(target_majority_size, sentiment_distribution[majority_class]),
            random_state=42
        )

        # Extract the minority class
        minority_samples = df_filtered[df_filtered['sentiment'] == minority_class]

        # Include neutral sentiments again if they were excluded
        if 'neutral' in reviews_df['sentiment'].unique():
            neutral_samples = reviews_df[reviews_df['sentiment'] == 'neutral']
            balanced_df = pd.concat([majority_samples, minority_samples, neutral_samples])
        else:
            balanced_df = pd.concat([majority_samples, minority_samples])

        print(f"New sentiment distribution: {balanced_df['sentiment'].value_counts()}")
        return balanced_df


    def detect_duplicates(self, reviews_df):
        print(f"Detecting near-duplicates...")

        # Setting the threshold
        threshold = self.duplicate_threshold

        # Filter out any rows with missing or empty text
        texts = reviews_df['cleaned_text'].dropna().tolist()

        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
        tfidf_matrix = vectorizer.fit_transform(texts)

        duplicate_indices = set()

        batch_size = 1000
        for start_idx in tqdm(range(0, len(texts), batch_size), desc="Detecting duplicates"):
            end_idx = min(start_idx + batch_size, len(texts))
            batch_matrix = tfidf_matrix[start_idx:end_idx]

            # Compute pairwise cosine similarity for the current batch
            similarities = cosine_similarity(batch_matrix, tfidf_matrix)

            # Identify duplicates within the batch
            for batch_idx, similarity_scores in enumerate(similarities):
                doc_idx = start_idx + batch_idx

                # Find documents with similarity greater than the threshold
                duplicate_indices.update(
                    doc_idx for similar_idx in np.where(similarity_scores > threshold)[0]
                    if similar_idx != doc_idx and similar_idx > doc_idx
                )

        # Flag duplicates and filter them out
        reviews_df['is_duplicate'] = reviews_df.index.isin(duplicate_indices)
        df_without_duplicates = reviews_df[~reviews_df['is_duplicate']]

        print(f"Initial number of records: {len(reviews_df)}")
        print(f"Removed {len(duplicate_indices)} duplicate entries. {len(df_without_duplicates)} entries remain.")
        return df_without_duplicates


    def extract_image_urls(self, image_data, size_key):
        if isinstance(image_data, list):
            return [img.get(size_key, 'NA') for img in image_data if isinstance(img, dict)]
        elif isinstance(image_data, dict):
            return image_data.get(size_key, 'NA')
        return 'NA'

    
    def run_pipeline(self, reviews_path, items_path):
        
        print("Running the pipeline...")


         # First step of the pipeline is to load the electronics reviews and metadata and store as a dataframe.
        reviews_df = self.load_dataframe(reviews_path)
        print(f"Reviews dataframe succesfully loaded with {len(reviews_df)} records.")


        # Second step of the pipeline is to process the review texts.
        reviews_df = self.process_text(reviews_df)


        # Third step of the pipeline is to remove duplicate review texts.
        reviews_df = self.detect_duplicates(reviews_df)


        # Fourth step of the pipeline is to balance the dataset to ensure similar numbers of positive and negative reviews.
        print("Analyzing the sentiment of reviews...")
        sentiment_results = reviews_df['cleaned_text'].apply(self.analyze_sentiment)
        reviews_df['sentiment'] = sentiment_results.apply(lambda x: x[0])
        reviews_df['sentiment_score'] = sentiment_results.apply(lambda x: x[1])
        reviews_df = self.balance_sentiment(reviews_df)


        # Fifth step is to convert the date into DD-MM-YYYY HH:MM:SS
        reviews_df['timestamp'] = pd.to_datetime(reviews_df['timestamp'], unit='ms').dt.strftime('%d-%m-%Y %H:%M:%S')


        # Sixth step is to select relevant columns from the electronics review dataset
        print("Columns in reviews dataset:", reviews_df.columns)
        reviews_df_cleaned = reviews_df[['rating', 'images', 'parent_asin', 'timestamp', 'helpful_vote', 'verified_purchase', 'detected_language', 'cleaned_text', 'text']].copy()
        # extract image urls
        reviews_df_cleaned['small_image_url'] = reviews_df_cleaned['images'].apply(lambda x: self.extract_image_urls(x, 'small_image_url'))
        reviews_df_cleaned['medium_image_url'] = reviews_df_cleaned['images'].apply(lambda x: self.extract_image_urls(x, 'medium_image_url'))
        reviews_df_cleaned['large_image_url'] = reviews_df_cleaned['images'].apply(lambda x: self.extract_image_urls(x, 'large_image_url'))
        reviews_df_cleaned = reviews_df_cleaned.drop(columns=['images'])
        reviews_df_cleaned.insert(0, 'review_id', range(1, len(reviews_df_cleaned) + 1)) # Create a unique index


        # Seventh step is to load the item metadata and store as a dataframe.
        items_df = self.load_dataframe(items_path)
        print(f"Items dataframe succesfully loaded with {len(items_df)} items.")


        # Eighth step is to select relevant columns from the electronics item dataset.
        items_df_cleaned = items_df[['main_category', 'title', 'price', 'store', 'details', 'parent_asin']]
        # extract brand
        items_df_cleaned['brand'] = items_df_cleaned.get('Brand Name', 'NA')
        # extract country of origin
        items_df_cleaned['country_of_origin'] = items_df_cleaned.get('Country of Origin', 'NA')
        # delete details column
        items_df_cleaned = items_df_cleaned.drop(columns=['details'])


        # Nineth step is to merge both reviews, along with the reviewed item's metadata, into one table for indexing
        combined_df = pd.merge(reviews_df_cleaned, items_df_cleaned, on='parent_asin')


        # Tenth step is to rename the columns
        combined_df = combined_df.drop(columns=['parent_asin'])
        combined_df_renamed = combined_df.rename(columns={
            'review_id': 'review_id',
            'rating': 'user_rating',
            'small_image_url': 'review_image_url_small',
            'medium_image_url': 'review_image_url_medium',
            'large_image_url': 'review_image_url_large',
            'timestamp': 'review_timestamp',
            'helpful_vote': 'number_of_helpful_votes',
            'verified_purchase': 'verified_purchase',
            'detected_language': 'review_language',
            'cleaned_text': 'review_text_cleaned',
            'text': 'review_text_original',
            'main_category': 'product_category',
            'title': 'product_name',
            'price': 'product_price_USD',
            'store': 'product_store',
            'brand': 'product_brand',
            'country_of_origin': 'product_country_of_origin'
        })

        combined_df_renamed.to_json("full_table_clean_new.json", orient='records', lines=True)
        print(f"Combined dataset successfully created with {len(combined_df_renamed)} records and {len(combined_df_renamed.columns)} columns: {combined_df_renamed.columns}")

        return reviews_df

if __name__ == "__main__":
    processor = DataProcessorNew()
    reviews_path = "dataset/electronics_reviews.json"
    items_path = "dataset/electronics_items.json"
    data = processor.run_pipeline(reviews_path, items_path)