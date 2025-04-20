import pandas as pd
import ast
import pysolr
import json
from datetime import datetime, timezone
from tqdm import tqdm


def parse_date(date_str):
    """Convert DD-MM-YYYY HH:MM:SS to ISO 8601 UTC format for Solr."""
    try:
        dt = datetime.strptime(date_str, '%d-%m-%Y %H:%M:%S') 
        dt = dt.replace(tzinfo=timezone.utc)  
        return dt.isoformat().replace("+00:00", "Z")  
    except ValueError:
        return None


def parse_json_field(field_value):
    """Parse fields that are expected to be JSON or list-like strings."""
    try:
        if isinstance(field_value, str):
            if field_value.startswith('[') and field_value.endswith(']'):
                return json.loads(field_value)
            else:
                return ast.literal_eval(field_value)
        return field_value
    except Exception as e:
        print(f"Error parsing JSON field: {e}")
        return None


# Load the cleaned dataset
def load_dataframe(path):
    data = []
    with open(path, 'r') as file:
        for i, line in enumerate(file): 
            data.append(json.loads(line.strip()))

    data_df = pd.DataFrame(data)

    return data_df


# Connect to Solr on the server 8983 with core name opinion_search
solr = pysolr.Solr('http://localhost:8983/solr/opinion_search/', always_commit=True)

# Load the cleaned dataset
data_path = "dataset/full_table_clean_final.json"
df = load_dataframe(data_path)
print(f"Loaded {len(df)} records")

# Prepare documents for Solr indexing
solr_documents = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Preparing Solr documents"):
    solr_doc = {}

    # Unique Identifier: review_id
    solr_doc['review_id'] = str(row.get('review_id', ''))

    # User Rating: user_rating (Float 1.0 - 5.0)
    solr_doc['user_rating'] = float(row.get('user_rating', 0.0))

    # Review Images (multivalued URLs)
    solr_doc['small_image_url'] = parse_json_field(row.get('review_image_url_small', '[]'))
    solr_doc['medium_image_url'] = parse_json_field(row.get('review_image_url_medium', '[]'))
    solr_doc['large_image_url'] = parse_json_field(row.get('review_image_url_large', '[]'))

    # Review Timestamp: review_timestamp
    review_timestamp = row.get('review_timestamp', '')
    if review_timestamp:
        solr_doc['review_timestamp'] = parse_date(review_timestamp)

    # Helpful Votes: number_of_helpful_votes
    solr_doc['number_of_helpful_votes'] = int(row.get('number_of_helpful_votes', 0))

    # Verified Purchase: verified_purchase
    solr_doc['verified_purchase'] = bool(row.get('verified_purchase', False))

    # Detected Language: review_language
    solr_doc['review_language'] = row.get('review_language', '')

    # Cleaned Review Text: review_text_cleaned
    solr_doc['review_text_cleaned'] = str(row.get('review_text_cleaned', ''))

    # Original Review Text: review_text_original
    solr_doc['review_text_original'] = row['review_text_original']

    # Product Metadata
    solr_doc['product_category'] = row.get('product_category', '')
    solr_doc['product_name'] = row.get('product_name', '')
    solr_doc['product_price_USD'] = float(row.get('product_price_USD', 0.0))
    solr_doc['product_store'] = row.get('product_store', '')
    solr_doc['product_brand'] = row.get('product_brand', '')
    solr_doc['product_country_of_origin'] = row.get('product_country_of_origin', '')

    # Sentiment Label
    solr_doc['review_subjectivity'] = row.get('subjectivity', '')
    solr_doc['review_polarity'] = row.get('polarity', '')

    solr_documents.append(solr_doc)
    if i % 1000 == 0:
        print(f"Processed {i} documents:", solr_doc)


# Batch indexing of documents
batch_size = 500
for start in tqdm(range(0, len(solr_documents), batch_size), desc="Indexing batches"):
    batch = solr_documents[start:start + batch_size]
    try:
        solr.add(batch)
    except Exception as e:
        print(f"Error during batch {start}-{start + batch_size}: {e}")

print("Indexing complete!")

'''
#to test out the index:
query = '*:*'  
params = {
    'q': query,  
    'rows': 10,  
    'fl': '*'
}

results = solr.search(**params)

print(f"Found {len(results)} results")
for doc in results:
    print(doc)  
'''