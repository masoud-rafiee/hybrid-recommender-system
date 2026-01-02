# 🚀 Hybrid Movie Recommender System

**MLP + Genetic Algorithm (GA) & Particle Swarm Optimization (PSO) on MovieLens 100K**

This repository demonstrates an end-to-end pipeline for building and optimizing a movie recommender system using Python. It covers data ingestion, exploratory analysis, baseline modeling with a Multilayer Perceptron (MLP) regressor, hyperparameter search with GA and PSO, and generation of personalized recommendations.

---

## 🔍 Table of Contents

1. [Features](#-features)
2. [Demo](#-demo)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [Results](#-results)
7. [Optimization Comparison](#-optimization-comparison)
8. [Top-5 Recommendations](#-top-5-recommendations)
9. [Contributing](#-contributing)
10. [License](#-license)
11. [Contact](#-contact)

---

## ✨ Features

- **Data Preprocessing**: Merge & clean MovieLens 100K ratings and movie metadata.
- **Exploratory Data Analysis**: Genre distribution, rating counts, user interest overlap.
- **Baseline Modeling**: MLP regressor with grid search hyperparameter tuning.
- **GA & PSO Optimization**: Evolve MLP hyperparameters for improved RMSE.
- **Recommendation Engine**: Generate top-N movie recommendations for any user.
- **Visualization**: Convergence plots comparing baseline, GA, and PSO performance.

---

## 📊 Demo



```bash
$ python src/recommend.py
Top 5 recommendations for user 1 using Baseline:
  ...
Top 5 recommendations for user 1 using GA:
  ...
Top 5 recommendations for user 1 using PSO:
  ...
```

---

## 📁 Project Structure

```
├── data/                       # Raw MovieLens CSVs
├── figures/                    # Generated plots (.png)
├── models/                     # Saved pipelines and histories
│   ├── baseline_mlp.pkl
│   ├── ga_mlp.pkl
│   ├── pso_mlp.pkl
│   ├── ga_history.json
│   └── pso_history.json
├── src/                        # Source code
│   ├── compare.py              # Convergence plotting
│   ├── data_loader.py          # CSV loading & merging
│   ├── eda.py                  # EDA functions
│   ├── GA.py                   # Genetic Algorithm tuning
│   ├── model.py                # Baseline MLP pipeline
│   ├── Main.py                 # End-to-end runner
│   ├── PSO.py                  # Particle Swarm Optimization
│   └── recommend.py            # Recommendation script
├── README.md                   # Project overview (this file)
```

---

## ⚙️ Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/masoud-rafiee/hybrid-movie-recommender.git
   cd hybrid-movie-recommender
   ```
2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .\.venv\Scripts\activate  # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

> > **Important:** The full dataset is too large to host here. Download the MovieLens 20M data from [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data) and place the extracted CSV files (`ratings.csv`, `movies.csv`, etc.) into the `data/` folder before running the project.

---

## 🚀 Usage

1. **Run full pipeline** (EDA, baseline, GA, PSO, sample prediction):
   ```bash
   python src/Main.py
   ```
2. **Tune with GA only**:
   ```bash
   python src/GA.py
   ```
3. **Tune with PSO only**:
   ```bash
   python src/PSO.py
   ```
4. **Generate recommendations for user 1**:
   ```bash
   python src/recommend.py
   ```
5. **Compare convergence**:
   ```bash
   python src/compare.py
   ```

---

## 📈 Results

- **Baseline RMSE:** 1.1575
- **GA-optimized RMSE:** 1.1238
- **PSO-optimized RMSE:** 1.1238



---

## ⚖️ Optimization Comparison

| Algorithm | Generations to Converge | Final RMSE |
| --------- | ----------------------- | ---------- |
| Baseline  | N/A                     | 1.1575     |
| GA        | 2                       | 1.1238     |
| PSO       | 1                       | 1.1238     |

---

## 🎥 Top-5 Recommendations (User 1)

| Model    | Movie ID | Title                             | Pred Rating |
| -------- | -------- | --------------------------------- | ----------- |
| Baseline | 2394     | Prince of Egypt, The (1998)       | 6.9046      |
| GA       | 4027     | O Brother, Where Art Thou? (2000) | 4.1293      |
| PSO      | 4027     | O Brother, Where Art Thou? (2000) | 4.1293      |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Expand the search space
- Integrate embedding-based deep models
- Deploy via a web framework (Flask, FastAPI)

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 📬 Contact

**Masoud Rafiee**\
Email: mrafiee22\@ubishops.ca\
GitHub: [@](https://github.com/MASOUD-RAFIEE) masoud-rafiee

