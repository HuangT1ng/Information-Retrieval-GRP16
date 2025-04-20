import streamlit as st
import pysolr
import pandas as pd
import altair as alt
from datetime import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

st.set_page_config(
    page_title="Mini Search Engine",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stMarkdown h1 {
        color: #4A90E2;
    }
    .stMarkdown p {
        font-size: 16px;
    }
    .result-card {
        background: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-card h3 {
        margin-bottom: 0;
    }
    .result-card p {
        margin-top: 5px;
        line-height: 1.5;
    }
    .facet-badge {
        background: #e0e0e0;
        border-radius: 15px;
        padding: 2px 8px;
        margin-right: 5px;
        cursor: pointer;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔍 Mini Search Engine")
st.write(
    "Welcome to the Mini Search Engine. Use the form below to search and filter results. "
    "You can also see a word cloud generated from the review texts."
)

st.sidebar.header("Additional Filters")
rating_min, rating_max = st.sidebar.slider("User Rating Range", 1.0, 5.0, (1.0, 5.0), 0.5)
subjectivity_options = {
    "All": None,
    "Opinionated": 1,
    "Neutral": 0
}
selected_subjectivity = st.sidebar.selectbox("Review Subjectivity", list(subjectivity_options.keys()))
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2025, 1, 1))

sort_options = {
    "Relevance (Default)": None,
    "Newest First": "review_timestamp desc",
    "Oldest First": "review_timestamp asc",
    "Highest Rating": "user_rating desc",
    "Lowest Rating": "user_rating asc"
}
sort_choice = st.sidebar.selectbox("Sort By", list(sort_options.keys()))


with st.form("search_form"):
    st.subheader("Search")
    col1, col2 = st.columns(2)
    with col1:
        product_name_query = st.text_input("Search by Product Name", value="", help="Enter product name for fuzzy search")
    with col2:
        query = st.text_input("Advanced Query", value="*:*", help="You can use field-specific queries, boolean operators, or wildcards.")
    rows = st.number_input("Number of results", min_value=1, max_value=200, value=10, step=1)
    submitted = st.form_submit_button("Search")

solr = pysolr.Solr('http://localhost:8983/solr/opinion_search/', always_commit=True)

if submitted:
    main_query = query
    if product_name_query.strip():
        main_query = f"product_name:*{product_name_query.strip()}*"
        if query.strip() and query != "*:*":
            main_query = f"({main_query}) AND ({query})"

    filter_queries = []
    filter_queries.append(f"user_rating:[{rating_min} TO {rating_max}]")
    start_str = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
    end_str = datetime.combine(end_date, datetime.max.time()).isoformat() + "Z"
    filter_queries.append(f"review_timestamp:[{start_str} TO {end_str}]")

    sort_param = sort_options[sort_choice]
    
    params = {
        "q": main_query,
        "rows": rows,
        "fl": "review_id, user_rating, review_timestamp, review_text_original, product_name, small_image_url, review_polarity, review_subjectivity",
        "fq": filter_queries,
        "facet": "on",
        "facet.field": "product_category",
        "facet.limit": 10
    }
    if sort_param:
        params["sort"] = sort_param

    if selected_subjectivity != "All":
        filter_queries.append(f"review_subjectivity:{subjectivity_options[selected_subjectivity]}")

    try:
        search_start_time = datetime.now()
        results = solr.search(**params)
        search_end_time = datetime.now()
        search_time = (search_end_time - search_start_time).total_seconds()
        
        st.write(f"Found {len(results)} results in {search_time * 1000:.2f} ms")
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        all_text = []  
        timestamps = []  
        for doc in results:
            review_polarity = doc.get('review_polarity', [None])[0] 
            if review_polarity == 1:
                sentiment_counts["Positive"] += 1
            elif review_polarity == 0:
                sentiment_counts["Negative"] += 1
            else:
                sentiment_counts["Neutral"] += 1

            review_text = doc.get('review_text_original', 'N/A')[0]
            if review_text and review_text != 'N/A':
                all_text.append(review_text)
            if doc.get('review_timestamp'):
                timestamps.append(doc['review_timestamp'])

        if all_text:
            text_combined = " ".join(all_text)
            words = word_tokenize(text_combined.lower()) 
            stop_words = set(stopwords.words('english'))
            word_freq = {}
            for word in words:
                if len(word) > 3 and word.isalpha() and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])
            
            st.subheader("Distribution")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                wordcloud = WordCloud(
                    width=400,
                    height=200,
                    background_color=None,
                    mode="RGBA",
                    colormap='Pastel1'
                ).generate(text_combined)
                
                fig, ax = plt.subplots(figsize=(5, 2.5))
                fig.patch.set_alpha(0)
                ax.set_facecolor('none')
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(5, 2.5))
                bars = ax.bar(range(len(top_words)), list(top_words.values()), color='skyblue')
                ax.set_xticks(range(len(top_words)))
                ax.set_xticklabels(list(top_words.keys()), rotation=45, ha='right', color='white')
                ax.tick_params(axis='both', colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.set_facecolor('none')
                fig.patch.set_alpha(0)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col3:
                if sum(sentiment_counts.values()) > 0:
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
                    wedges = ax.pie(
                        sentiment_counts.values(),
                        colors=colors,
                        startangle=90,
                        labels=[''] * 3,  
                        autopct=''  
                    )
                    ax.legend(
                        wedges[0],  
                        sentiment_counts.keys(),
                        title="Sentiment",
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                        labelcolor='white'
                    )
                    ax.set_facecolor('none')
                    fig.patch.set_alpha(0)
                    plt.tight_layout()  
                    st.pyplot(fig)
                    plt.close(fig)

        st.subheader("Search Results")
        for doc in results:
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"### Review ID: {doc.get('review_id', 'N/A')}", unsafe_allow_html=True)
            images = doc.get('small_image_url', [])
            if images:
                for img_url in images[:2]: 
                    st.image(img_url, width=100)

            rating = doc.get('user_rating', [0])[0]
            stars_html = ""
            if isinstance(rating, (int, float)):
                full_stars = int(rating)
                half_star = 1 if rating - full_stars >= 0.5 else 0
                empty_stars = 5 - full_stars - half_star

                stars_html = (
                    "★" * full_stars + 
                    "☆" * empty_stars
                )
                star_color = "gold"
                st.markdown(f"**User Rating:** {rating:.1f} <span style='color:{star_color}; font-size:20px;'>{stars_html}</span>", unsafe_allow_html=True)
            else:
                st.markdown("**User Rating:** N/A")

            ts_list = doc.get('review_timestamp', [])
            raw_timestamp = ts_list[0] if isinstance(ts_list, list) and ts_list else 'N/A'
            if raw_timestamp != 'N/A':
                try:
                    dt = datetime.fromisoformat(raw_timestamp.replace("Z", ""))
                    formatted_ts = dt.strftime("%b %d, %Y at %I:%M %p")
                except:
                    formatted_ts = raw_timestamp
            else:
                formatted_ts = 'N/A'
            st.markdown(f"**Review Timestamp:** {formatted_ts}", unsafe_allow_html=True)

            st.markdown(f"**Product Name:** {doc.get('product_name', 'N/A')[0]}", unsafe_allow_html=True)

            review_text = doc.get('review_text_original', 'N/A')[0]
            st.markdown(f"**Review Text:** {review_text}", unsafe_allow_html=True)

            review_polarity = doc.get('review_polarity', [None])[0] 
            if review_polarity == 1:
                sentiment = "positive"
                polarity_color = "green"
            elif review_polarity == 0:
                sentiment = "negative"
                polarity_color = "red"
            else:
                sentiment = "neutral (because subjectivity is neutral)"
                polarity_color = "grey"

            st.markdown(f"**Review Polarity:** <span style='color:{polarity_color}; font-weight:bold;'>{sentiment}</span>", unsafe_allow_html=True)
            review_subjectivity = doc.get('review_subjectivity', [None])[0] 
            subjectivity = "opinionated" if review_subjectivity == 1 else "neutral" if review_subjectivity == 0 else "not applicable"
            st.markdown(f"**Review Subjectivity:** {subjectivity}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            if review_text and review_text != 'N/A':
                all_text.append(review_text)
            if doc.get('review_timestamp'):
                timestamps.append(doc['review_timestamp'])


        if timestamps:
            date_data = []
            for t in timestamps:
                try:
                    dt = datetime.fromisoformat(t.replace("Z", ""))
                    date_data.append(dt)
                except Exception as e:
                    pass
            if date_data:
                df = pd.DataFrame({"review_date": date_data})
                df["month"] = df["review_date"].dt.to_period("M")
                chart_data = df.groupby("month").size().reset_index(name="count")
                chart_data["month"] = chart_data["month"].dt.to_timestamp()
                st.subheader("Review Timeline")
                c = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        x="month:T",
                        y="count:Q",
                        tooltip=["month:T", "count:Q"]
                    )
                    .properties(width=700, height=300)
                )
                st.altair_chart(c, use_container_width=True)

        if timestamps:
            date_data = []
            for t in timestamps:
                try:
                    dt = datetime.fromisoformat(t.replace("Z", ""))
                    date_data.append(dt)
                except Exception as e:
                    pass
            if date_data:
                df = pd.DataFrame({"review_date": date_data})
                df["month"] = df["review_date"].dt.to_period("M")
                chart_data = df.groupby("month").size().reset_index(name="count")
                chart_data["month"] = chart_data["month"].dt.to_timestamp()
                st.subheader("Review Timeline")
                c = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        x="month:T",
                        y="count:Q",
                        tooltip=["month:T", "count:Q"]
                    )
                    .properties(width=700, height=300)
                )
                st.altair_chart(c, use_container_width=True)
    except Exception as e:
        st.error(f"Error during search: {e}")
else:
    st.info("Please enter your query and click 'Search' above.")
