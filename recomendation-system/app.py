import logging
import os
import pickle
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_tmdb_bearer_token() -> str:
    """Read the optional TMDB token from environment or Streamlit secrets."""
    token = os.environ.get("TMDB_BEARER_TOKEN")
    if token:
        return token
    try:
        return st.secrets["TMDB_BEARER_TOKEN"]
    except (FileNotFoundError, KeyError):
        return ""


TMDB_BEARER_TOKEN = load_tmdb_bearer_token()
MOVIE_DATA_PATH = Path(__file__).with_name("movie_dict.pkl")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_data(show_spinner="Loading the movie collection…")
def load_movies() -> pd.DataFrame:
    """Load the preprocessed movie data exported by the notebook."""
    if not MOVIE_DATA_PATH.is_file():
        raise FileNotFoundError(f"Movie data was not found at {MOVIE_DATA_PATH}.")

    with MOVIE_DATA_PATH.open("rb") as movie_file:
        movie_data = pickle.load(movie_file)

    if not isinstance(movie_data, dict):
        raise ValueError("movie_dict.pkl must contain a dictionary of movie data.")

    movies = pd.DataFrame(movie_data)
    required_columns = {"movie_id", "title", "tags"}
    missing_columns = required_columns.difference(movies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"movie_dict.pkl is missing required columns: {missing}.")

    movies["title"] = movies["title"].fillna("").astype(str)
    movies["tags"] = movies["tags"].fillna("").astype(str)
    movies = movies[movies["title"].str.strip().ne("")].reset_index(drop=True)
    if movies.empty:
        raise ValueError("The saved movie collection contains no movie titles.")
    return movies


@st.cache_resource(show_spinner="Preparing movie recommendations…")
def load_movie_vectors(tags: tuple[str, ...]) -> csr_matrix:
    """Build the same CountVectorizer representation used by the notebook."""
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    try:
        return vectorizer.fit_transform(tags)
    except ValueError as error:
        raise ValueError("The saved movie tags do not contain usable terms.") from error


def recommend_movies(
    movie_index: int,
    movie_vectors: csr_matrix,
    count: int = 8,
) -> list[tuple[int, float]]:
    """Return the most similar movie indexes and their cosine-similarity scores."""
    scores = cosine_similarity(movie_vectors[movie_index], movie_vectors).ravel()
    scores[movie_index] = -1
    ranked_movies = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(index, float(score)) for index, score in ranked_movies[:count] if score > 0]


@st.cache_data(ttl=3600, show_spinner=False)
def movie_poster(movie_id: object, bearer_token: str) -> tuple[str | None, str | None]:
    """Fetch a movie poster URL from TMDB when a bearer token is configured."""
    if not bearer_token or pd.isna(movie_id):
        return None, None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {bearer_token}",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as error:
        logger.warning("TMDB poster lookup failed for movie %s: %s", movie_id, error)
        return None, "TMDB could not load one or more posters. Check your token and network connection."

    if not isinstance(data, dict):
        logger.warning("TMDB returned an unexpected response for movie %s", movie_id)
        return None, "TMDB returned an unexpected response while loading posters."

    poster_path = data.get("poster_path")
    if not poster_path:
        return None, None
    return f"https://image.tmdb.org/t/p/w500{poster_path}", None


st.title("🎬 Movie Recommender")
st.caption("Discover films with similar stories, genres, cast, and keywords.")

try:
    movies = load_movies()
    movie_vectors = load_movie_vectors(tuple(movies["tags"]))
except (FileNotFoundError, ValueError, pickle.UnpicklingError) as error:
    st.error(str(error))
    st.info("Keep the notebook-exported movie_dict.pkl in the same folder as app.py.")
    st.stop()

selected_index = st.selectbox(
    "Choose a movie",
    options=range(len(movies)),
    index=None,
    format_func=lambda index: movies.at[index, "title"],
    placeholder="Search for a movie…",
)

if selected_index is None:
    st.write("Choose a movie above to see your recommendations.")
else:
    selected_movie = movies.iloc[selected_index]
    st.subheader(f"Because you chose {selected_movie['title']}")

    recommendations = recommend_movies(selected_index, movie_vectors)
    if not recommendations:
        st.warning("No similar movies were found for this title.")
    else:
        columns = st.columns(4)
        poster_errors = set()
        for position, (movie_index, score) in enumerate(recommendations):
            movie = movies.iloc[movie_index]
            with columns[position % len(columns)]:
                poster_url, poster_error = movie_poster(
                    movie["movie_id"],
                    TMDB_BEARER_TOKEN,
                )
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown("🎞️")
                st.markdown(f"**{movie['title']}**")
                st.caption(f"Similarity: {score:.0%}")
                if poster_error:
                    poster_errors.add(poster_error)

        for poster_error in poster_errors:
            st.warning(poster_error)

if not TMDB_BEARER_TOKEN:
    st.caption("Posters are optional. Configure TMDB_BEARER_TOKEN in Streamlit secrets or your environment to enable them.")

with st.sidebar.expander("Credits"):
    st.markdown(
        "[![TMDB](https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg)](https://www.themoviedb.org/)"
    )
    st.markdown(
        "This product uses the [TMDB API](https://www.themoviedb.org/) "
        "but is not endorsed or certified by TMDB."
    )
