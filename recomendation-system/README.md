# Movie Recommender

This Streamlit app uses the preprocessed `movie_dict.pkl` exported by
`Movie_recomendation.ipynb`. It recreates the notebook's CountVectorizer and
cosine-similarity recommendation approach when the app starts.

## Run locally

From this directory, install dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The recommendation engine works without an API token. To display posters, set
`TMDB_BEARER_TOKEN` as an environment variable or in Streamlit secrets. Never
put the token in source code or commit your secrets file.

For local development, create `.streamlit/secrets.toml` in this directory:

```toml
TMDB_BEARER_TOKEN = "paste-your-new-token-here"
```

The secrets file is excluded by the repository's `.gitignore`. For Streamlit
Community Cloud, add `TMDB_BEARER_TOKEN` in the app's **Settings → Secrets**.

If a token was committed or pushed previously, deleting it from the latest
version is not enough: revoke it with TMDB, issue a replacement, and remove it
from Git history before making the repository public.

## Data and attribution

The training data is the [TMDB 5000 Movie Metadata dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
Movie posters are loaded from TMDB at runtime. The app's Credits section includes
TMDB's logo and required notice: “This product uses the TMDB API but is not
endorsed or certified by TMDB.”
