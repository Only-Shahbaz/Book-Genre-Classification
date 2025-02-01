import streamlit as st
import pickle
import re
import nltk
import nltk.corpus

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('stopwords')


model = pickle.load(open('Model/model.pkl', 'rb'))
vectorizer = pickle.load(open('Model/vectorizer.pkl', 'rb'))
le = pickle.load(open('Model/le.pkl', 'rb'))
df_resampled = pickle.load(open('Model/df_resampled.pkl', 'rb'))

def clean_description(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join words back into a single string
    return ' '.join(words)

def recommend_books(predicted_genre, df_resampled, top_n=5):
    recommended_book = df_resampled[df_resampled['Genre_2'] == predicted_genre].sample(top_n)
    return recommended_book[['Title','Book-Author','Image-URL-L']]


st.title('Book Genre Classification')

description = st.text_area(
        "Book Description",
        height=200,
        placeholder="Enter the book description here..."
    )
if st.button("Predict Genre"):
        if description.strip() == "":
            st.warning("Please enter a book description.")
        else:
            # Show spinner while processing
            with st.spinner("Analyzing..."):
                cleaned_description = clean_description(description)
                X = vectorizer.transform([cleaned_description])
                genre = model.predict(X)
                genre = le.inverse_transform(genre)[0]
                st.success(f"The genre of the book is: {genre}")

                st.subheader("You may also like:")

                recommended_books = recommend_books(genre, df_resampled)

                for index, book in recommended_books.iterrows():
                    st.write(f"Title: {book['Title']}")
                    st.write(f"Author: {book['Book-Author']}")
                    st.image(book['Image-URL-L'], width=200)
