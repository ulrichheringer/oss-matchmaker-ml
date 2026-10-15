# Open Source Matchmaker

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)  [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)  [![MIT License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](LICENSE)  ![Synthetic Data](https://img.shields.io/badge/Data-Synthetic-6b7280?style=for-the-badge)

Recommends repositories and issues from a small catalog using text similarity, with a simple collaborative signal when interaction history is available.

## Data

The catalog and user profiles are deterministic synthetic examples generated locally in `pipeline.py`. No private data or external download is required.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Evaluation results

With seed 42, retrieval was evaluated for eight synthetic profiles against a 24-item catalog. Precision@5 was 0.6000 and recall@5 was 1.0000. The recommender uses TF-IDF for content matching and can blend in a simple collaborative signal when there is enough interaction history. These scores apply only to the artificial catalog; they do not predict relevance on GitHub.

## Limitations

The catalog and profiles are small and synthetic. Real recommendations would need current repository and issue metadata, user-selected interests, and evaluation against actual feedback.
