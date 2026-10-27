import streamlit as st
import pickle
import re
import html
from pathlib import Path

MODEL_PATH = Path("movie_sentiment_model.pkl")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


st.set_page_config(
    page_title="Movie Sentiment Classifier",
    page_icon="🎬",
)

st.title("🎬 Movie Review Sentiment Classifier")
st.write(
    "Enter a movie review below and the model will predict whether "
    "the sentiment is positive or negative."
)

review = st.text_area(
    "Movie Review", height=180, placeholder="Type or paste a movie review here..."
)

if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        model = load_model()

        cleaned = clean_text(review)
        prediction = int(model.predict([cleaned])[0])
        probabilities = model.predict_proba([cleaned])[0]

        if prediction == 1:
            label = "Positive 😊"
            confidence = probabilities[1]
        else:
            label = "Negative 😠"
            confidence = probabilities[0]

        st.subheader("Result")
        st.metric("Predicted Sentiment", label)
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.write("Probability breakdown:")
        st.write(f"Negative: {probabilities[0]:.4f}")
        st.write(f"Positive: {probabilities[1]:.4f}")

        st.progress(float(probabilities[1]))

        with st.expander("View cleaned text"):
            st.text(cleaned)
