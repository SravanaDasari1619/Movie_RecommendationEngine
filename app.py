import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import re

# Load the movie dataset from Hugging Face
dataset = load_dataset("Pablinho/movies-dataset", split="train")
movies_df = pd.DataFrame(dataset)

# Optionally display available columns for debugging (commented out to hide)
# st.write("Available columns:", movies_df.columns.tolist())

# Function to recommend movies based on watch history
def recommend_movies_by_history(watch_history):
    # Combine relevant features for TF-IDF vectorization
    movies_df['combined_features'] = (
        movies_df['Genre'].fillna('') + ', ' +
        movies_df['Overview'].fillna('') + ', ' +
        movies_df['Popularity'].astype(str) + ', ' +
        movies_df['Vote_Average'].astype(str)
    )

    vectorizer = TfidfVectorizer()
    feature_vector = vectorizer.fit_transform(movies_df['combined_features'])
    similarity = cosine_similarity(feature_vector)

    recommended_movies = []
    reasons = []

    for movie in watch_history:
        movie = movie.strip()

        if not movie:
            st.error("Please enter a valid movie title.")
            return [], []

        try:
            # Get the index of the movie in the DataFrame (case insensitive)
            movie_index = movies_df[movies_df['Title'].str.lower() == movie.lower()].index[0]

            # Get similarity scores for the selected movie
            distances = similarity[movie_index]

            # Get indices of recommended movies based on similarity scores
            recommended_indices = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:4]  # Top 3 recommendations

            for i in recommended_indices:
                recommended_movie_title = movies_df.iloc[i[0]]['Title']
                if recommended_movie_title not in recommended_movies:
                    recommended_movies.append(recommended_movie_title)
                    reasons.append(f"Similar to '{movie}' due to genre and storyline.")

        except IndexError:
            st.error(f"Movie '{movie}' not found in the dataset.")

    return recommended_movies, reasons

# Function to suggest movies based on current mood
def suggest_movies_based_on_mood(user_query):
    mood_keywords = {
        'happy': ['comedy', 'funny', 'light-hearted', 'animation', 'joyful', 'cheerful'],
        'sad': ['drama', 'tear-jerker', 'sad'],
        'adventure': ['adventure', 'action', 'thriller'],
    }

    extracted_mood = None

    # Check for mood keywords in user input
    for mood, keywords in mood_keywords.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', user_query, re.IGNORECASE) for keyword in keywords):
            extracted_mood = mood
            break

    if extracted_mood is None:
        st.error("No valid mood detected from your input.")
        return [], []

    # Filter movies based on extracted mood
    mood_based_recommendations = []
    reasons = []

    # Define genres based on extracted mood
    if extracted_mood == "happy":
        mood_genres = ['Animation', 'Comedy']
    elif extracted_mood == "sad":
        mood_genres = ['Drama']
    elif extracted_mood == "adventure":
        mood_genres = ['Adventure', 'Action']

    # Loop through each genre and filter movies accordingly
    for genre in mood_genres:
        filtered_movies = movies_df[movies_df['Genre'].str.contains(genre, case=False, na=False)]

        for _, row in filtered_movies.iterrows():
            if row['Title'] not in mood_based_recommendations:
                mood_based_recommendations.append(row['Title'])
                reasons.append(f"Recommended because it fits your mood for '{extracted_mood}'.")

    return mood_based_recommendations[:5], reasons  # Limit to top 5 recommendations

# Streamlit UI
st.title('Movie Recommendation App')

option = st.selectbox('Choose an option:', ['Recommend Movies Based on My Watch History',
                                            'Suggest Movies Based on Current Mood'])

if option == 'Recommend Movies Based on My Watch History':
    watch_history_input = st.text_input('Enter your favorite movie:')

    if st.button('Get Recommendations'):
        watch_history = [movie.strip() for movie in watch_history_input.split(',')]

        st.write(f"User Watch History: {watch_history}")

        if watch_history:
            recommended_movies, reasons = recommend_movies_by_history(watch_history)

            if recommended_movies:
                st.write("Recommended Movies:")
                for movie, reason in zip(recommended_movies, reasons):
                    st.write(f"- {movie}: {reason}")
            else:
                st.write("No recommendations found based on your input.")

elif option == 'Suggest Movies Based on Current Mood':
    user_query_input = st.text_input('Describe your mood or preferences (e.g., "I am in the mood for comedy movies"):')

    if st.button('Get Suggestions'):
        user_query = user_query_input.strip()

        if user_query:
            suggested_movies, reasons = suggest_movies_based_on_mood(user_query)

            if suggested_movies:
                st.write("Suggested Movies:")
                for movie, reason in zip(suggested_movies, reasons):
                    st.write(f"- {movie}: {reason}")
            else:
                st.write("No suggestions found based on your input.")
