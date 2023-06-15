import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import joblib

# Load the saved model
loaded_model = joblib.load('best_model.pkl')

# Define the function to extract and preprocess the text
def extract_and_preprocess_text(url):
    # Send a GET request to the URL and retrieve the HTML content
    response = requests.get(url)
    html_content = response.content

    # Create a Beautiful Soup object to parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the text from the HTML
    text = soup.get_text()

    # Preprocess the text
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.strip()  # Remove leading/trailing spaces

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join the filtered tokens back into a string
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text

# Create the Streamlit web app
def main():
    #st.title("URL Text Classification")
    st.markdown("<h1 style='text-align: center;'>URL Text Classification</h1>", unsafe_allow_html=True)

    # Input URL
    url = st.text_input("Enter the URL:")
    if url:
        # Extract and preprocess the text
        preprocessed_text = extract_and_preprocess_text(url)

        # Display the preprocessed text
        st.subheader("Preprocessed Text:")
        selected_text = st.selectbox("", [preprocessed_text], key='preprocessed_text')

        # Predict the category
        category = loaded_model.predict([selected_text])[0]

        # Display the predicted category
        st.subheader("Predicted Category:")
        st.markdown(f"<h3 style='text-align: center; color: blue;'>{category}</h3>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
