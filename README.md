# AI-Powered Movie Recommendation System

A movie recommendation system built on the MovieLens dataset, combining
content-based filtering (genre similarity) and collaborative filtering
(user-rating similarity), enriched with metadata from the TMDb API.

## Problem Statement

Streaming platforms lose engagement and retention when users can't quickly
find something worth watching - decision fatigue drives abandonment. This
project explores two complementary recommendation strategies and compares
their strengths:

- **Content-based filtering** - recommends movies similar in genre to what
  a user has already rated highly. Works even for brand-new movies with no
  ratings yet (the "cold start" case).
- **Collaborative filtering** - recommends movies liked by _other users_
  with similar taste, using rating similarity (`scipy.spatial.distance.cosine`
  / `scipy.stats.pearsonr`). Captures taste signals that genre metadata
  alone can't (e.g. two thrillers can be genre-similar but very different
  in quality/tone).

The trade-off between the two - and why a real system needs both - is the
core insight this project demonstrates.

## Data Source

- **[MovieLens dataset via Kaggle](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)**
  (`movies.csv`, `ratings.csv`)
- **[TMDb API](https://www.themoviedb.org/documentation/api)** - used to
  enrich movies by title/year with poster, overview, popularity, and
  cast/crew data. Requires a free TMDb API key.

> **Note on synthetic data:** a small table of synthetic user profiles
> (`name`, `age`) appears in the notebook purely for display/demo purposes,
> clearly labeled as synthetic. These fields are **never** inputs to the
> recommendation engine - the engine only ever uses real `ratings.csv` data
> (genre preferences and watch/rating history).

## How to Run

1. **Clone the repo**

    ```bash
    git clone https://github.com/idoberk/movie-recommendation-system.git
    cd movie-recommendation-system
    ```

2. **Set up the environment**

    ```bash
    conda env create -f environment.yml
    conda activate movie-recommendation-system
    ```

3. **Get the dataset**
   Download the dataset zip from the
   [Kaggle dataset page](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)
   (Kaggle names it `archive.zip` by default - feel free to rename it,
   the filename itself doesn't matter). Unzip it and make sure
   `movies.csv` and `ratings.csv` end up **directly inside** a local
   `data/` folder at the repo root, with no extra subfolder in between:

    ```
    data/
    ├── movies.csv
    └── ratings.csv
    ```

    `data/` is gitignored, so these files are never committed.

4. **Get a TMDb API key**
   Sign up for a free key at [themoviedb.org](https://www.themoviedb.org/documentation/api),
   then set it as an environment variable rather than hardcoding it:

    ```bash
    export TMDB_API_KEY="your-key-here"
    ```

5. **Run the notebook**
   Open `notebooks/movie_recommendations.ipynb` in VS Code (Colab extension)
   or Jupyter and run top to bottom. TMDb responses are cached locally on
   first run to avoid re-querying the API on subsequent runs.

## Repo Structure

```
movie-recommendation-system/
├── README.md
├── environment.yml
├── .gitignore
├── data/               # gitignored - place downloaded CSVs here
├── notebooks/
│   └── movie_recommendations.ipynb
└── src/                # reusable helper functions (preprocessing, similarity, plotting)
```

## Main Findings

_(To be filled in after analysis - e.g. dataset sparsity percentage,
content-based vs. collaborative filtering comparison, key EDA takeaways.)_

## Limitations

_(To be filled in - e.g. cold-start handling, sparsity impact on
collaborative filtering quality, any biases in the rating data.)_
