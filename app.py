import streamlit as st
from transformers import pipeline

# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="News Headline Generator",
    page_icon="📰",
    layout="centered"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    textarea {
        font-size: 18px !important;
    }

    .stButton button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 50px;
    }

    </style>
""", unsafe_allow_html=True)

# =====================================
# TITLE
# =====================================

st.title("📰 News Headline Generator")

st.write(
    "Generate short headlines from long news articles using Transformer-based NLP"
)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    return summarizer

summarizer = load_model()

# =====================================
# USER INPUT
# =====================================

news_text = st.text_area(
    "Enter Full News Article",
    height=300,
    placeholder="Paste long news article here..."
)

# =====================================
# BUTTON
# =====================================

if st.button("Generate Headline"):

    if news_text.strip() == "":

        st.warning("Please enter news article")

    else:

        with st.spinner("Generating headline..."):

            result = summarizer(
                news_text,
                max_length=20,
                min_length=5,
                do_sample=False
            )

            generated_headline = result[0]['summary_text']

        st.subheader("Generated Headline")

        st.success(generated_headline)