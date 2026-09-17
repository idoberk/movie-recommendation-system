# AI-Powered Movie Recommendation System

A movie recommendation system built on the MovieLens dataset, using
**k-means clustering** to group users by genre-preference/rating pattern
and recommend top-rated movies within a user's own cluster, enriched
with metadata from the TMDb API.

## Problem Statement

Streaming platforms lose engagement and retention when users can't
quickly find something worth watching - decision fatigue drives
abandonment. This project addresses that discovery problem with
**personalization through clustering**: rather than requiring dense,
pairwise rating overlap between individual users (what classic
collaborative filtering needs), users are grouped by their own relative
genre-preference pattern, and each user is recommended the top-rated,
sufficiently-reviewed movies within their own cluster.

This is a direct response to the dataset's core challenge - severe
sparsity (see Main Findings below). Clustering only needs a user's own
rating pattern to place them in a group, not specific overlap with other
individual users, which is what makes it workable at this sparsity
level. The trade-off is that cluster-level recommendations are coarser
than an individual-level model would give - quantified below rather
than glossed over.

## Data Source

- **[MovieLens dataset via Kaggle](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system)**
  (`movies.csv`, `ratings.csv`). No demographic data exists in this
  dataset - no real age, name, or stated preferences per user.
- **[TMDb API](https://www.themoviedb.org/documentation/api)** - used to
  enrich recommended movies with poster, overview, and popularity data.
  Requires a free TMDb API key.

> **Note on synthetic data:** a small table of synthetic user profiles
> (`name`, `age`) appears in the notebook purely for display/demo
> purposes, generated with Faker and clearly labeled as synthetic.
> These fields are **never** inputs to the recommendation engine - the
> engine only ever uses real `ratings.csv` data (genre preferences and
> rating history).

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
   Sign up for a free account at [themoviedb.org](https://www.themoviedb.org/),
   then generate a key under Settings > API. Create a `.env` file at the
   repo root (gitignored - never commit this) with:

    ```
    TMDB_API_KEY=your_key_here
    ```

5. **Run the notebook**
   Open `notebooks/movie_recommendations.ipynb` in Jupyter or VS Code
   and run top to bottom. If using VS Code, make sure the notebook's
   kernel is pointed directly at the `movie-recommendation-system` conda
   environment's `python.exe` - `conda env config vars set` doesn't
   reach kernels launched by VS Code, so the TMDb step won't pick up the
   API key otherwise. TMDb responses are cached locally in
   `data/tmdb_cache.json` after the first run, so re-running the
   notebook doesn't re-query movies already fetched.

## Repo Structure

```
movie-recommendation-system/
├── README.md
├── environment.yml
├── .gitignore
├── data/               # gitignored - place downloaded CSVs here
└── notebooks/
    └── movie_recommendations.ipynb
```

## Main Findings

**Data quality:** both CSVs are clean at the row level (0 nulls, 0
duplicates), but have real quirks that shaped several design decisions
downstream. Ratings skew heavily positive - 4.0 is the single most
common rating (~26.6% of all 25M ratings), and ratings of 3.5+ make up
~63% of the total. 8.1% of movies (5,062 of 62,423) carry a literal
"(no genres listed)" placeholder instead of real genre data, and 0.66%
of titles have no parseable release year.

**Sparsity is the headline data-quality finding.** The user-item rating
matrix is **99.74% sparse** - only 25,000,095 of the ~9.6 billion
theoretical (162,541 users × 59,047 movies) cells are actually filled
in. Combined with a long-tail pattern (48.8% of movies have 5 or fewer
ratings, while a handful of blockbusters absorb tens of thousands), most
movies simply don't carry enough signal for reliable movie-by-movie
comparison. This is the practical justification for grouping users by
pattern (k-means) rather than requiring dense pairwise overlap between
individuals.

**Hypothesis test:** comparing mean ratings between Horror and
Film-Noir (Welch's t-test, used because the two genres' variances differ
meaningfully - ratio 1.556) found a statistically significant difference
(p < 0.001): Film-Noir averages 3.93 vs. Horror's 3.29, a 0.63-point gap
on the 0.5-5.0 scale. With this much data (1.89M Horror ratings, 247K
Film-Noir ratings) even a tiny difference would register as
"significant," but a 0.63-point gap is large enough on this scale to
represent a real, noticeable difference in how these genres are
typically rated.

**Modeling:** users are clustered (k-means, k=5) on their average
rating per genre, centered on their own overall average so clustering
captures _relative_ taste rather than just how generously someone rates
everything, and whitened before clustering. The five resulting clusters
are genuinely distinct - e.g. one leans toward Animation/Musical and
away from Horror, another strongly avoids Children's/Animated content in
favor of Crime/War/Drama.

**Evaluation, stated honestly:**

- Silhouette scores are low across k=4/5/6 (0.061 / 0.039 / 0.037) -
  the clusters overlap substantially rather than forming cleanly
  separated groups, meaning genre taste behaves more like a continuum
  than five hard-walled personality types.
- Despite that, a cluster-cohesion check found real signal: on average,
  ~15% of a user's personally-loved movies (rated ≥4.5) appear in their
  own cluster's top-50 list - roughly 170x above the ~0.085% expected by
  pure chance.
- A hand-crafted synthetic preference profile (loves Horror/Thriller,
  dislikes Musical/Children's movies) correctly routed to the cluster
  whose real profile matches that pattern, confirming the pipeline
  behaves sensibly on a known input.

**Bottom line:** cluster-based recommendations clear the bar of "better
than random" by a wide, independently-verified margin, but they're a
coarser signal than an individual-level model would give. For a real
product, this points to a hybrid rollout - cluster-based recommendations
as a reasonable default, with an individual-level layer (e.g. genre
similarity to specific highly-rated movies) added over time to sharpen
results for the ~85% of a user's favorites that cluster-level
recommendations currently miss.

## Limitations

- **Cluster-level, not individual-level:** every user in a cluster gets
  the same recommendation list. The cohesion check (~15% overlap) shows
  this captures real but partial signal - most of an individual user's
  personal favorites aren't in their cluster's list.
- **TMDb enrichment covers a small, curated set of movies** (the
  recommendation demo's output), not the full 62,423-movie catalog -
  enriching everything isn't necessary for this project and isn't
  practical at this scale within TMDb's rate limits.
- **Simulated user profiles are cosmetic only:** names and ages are
  fake. Genre preferences and watch history are real. No age- or
  name-based finding should ever be inferred from this project, since
  none would reflect anything real about the underlying users.
- **The MovieLens 25M dataset only includes users with > 20 ratings:**
  so "cold start" in this project is really about _new movies_ with no
  ratings yet, not brand-new users - every user in this dataset already
  clears that floor.
