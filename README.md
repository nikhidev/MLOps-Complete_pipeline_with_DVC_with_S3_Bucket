# Complete MLOps Pipeline

An end-to-end **MLOps project** that demonstrates how to build, version, reproduce, and evaluate a machine learning pipeline using **Python, Scikit-learn, DVC, Git, and ML workflow automation**.

The project uses a **Spam/Ham text classification dataset** and implements the complete machine learning lifecycle:

**Data Ingestion → Data Preprocessing → Feature Engineering → Model Building → Model Evaluation**

---

## 🚀 Project Overview

This project demonstrates how a traditional machine learning model can be converted into a reproducible MLOps pipeline.

The pipeline performs the following tasks:

1. Downloads and ingests the dataset.
2. Splits the dataset into training and testing data.
3. Cleans and preprocesses the text.
4. Converts text into numerical features using **TF-IDF**.
5. Trains a **Random Forest Classifier**.
6. Evaluates the trained model.
7. Tracks pipeline stages and parameters using **DVC**.
8. Versions source code and configuration using **Git**.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK**
* **PyYAML**
* **DVC**
* **Git & GitHub**
* **Pickle**
* **TF-IDF**
* **Random Forest Classifier**

---

## 📂 Project Structure

```text
Complete_MLOPS_Pipeline/
│
├── .dvc/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   ├── interim/
│   │   ├── train_processed.csv
│   │   └── test_processed.csv
│   │
│   └── processed/
│       ├── train_tfidf.csv
│       └── test_tfidf.csv
│
├── experiments/
│   └── mynotebook.ipynb
│
├── models/
│   └── model.pkl
│
├── reports/
│   └── metrics.json
│
├── src/
│   ├── data_ingestion.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
│
├── logs/
│
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── README.md
```

---

## 🔄 MLOps Pipeline

```text
                    ┌─────────────────────┐
                    │   Dataset / CSV     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Ingestion     │
                    │ data_ingestion.py    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing   │
                    │ data_processing.py   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │   TF-IDF Vectorizer │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Building    │
                    │ Random Forest       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ Metrics / Reports   │
                    └─────────────────────┘
```

---

## 📌 Pipeline Stages

### 1. Data Ingestion

The data ingestion stage downloads the spam dataset and performs the initial preparation.

It:

* Loads the CSV dataset.
* Removes unnecessary columns.
* Renames columns to `target` and `text`.
* Splits the dataset into training and testing datasets.
* Stores the resulting datasets in `data/raw/`.

Output:

```text
data/raw/train.csv
data/raw/test.csv
```

---

### 2. Data Preprocessing

The preprocessing stage cleans the raw text data.

Typical preprocessing operations include:

* Converting text to lowercase.
* Tokenization.
* Removing stopwords.
* Removing punctuation.
* Stemming/normalizing text.
* Removing duplicate records.
* Encoding the target variable.

Output:

```text
data/interim/train_processed.csv
data/interim/test_processed.csv
```

---

### 3. Feature Engineering

Machine learning algorithms cannot directly work with raw text.

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical feature vectors.

The vectorizer is fitted only on the training dataset:

```python
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

This prevents information from the test dataset from leaking into the training process.

Output:

```text
data/processed/train_tfidf.csv
data/processed/test_tfidf.csv
```

---

### 4. Model Building

A **Random Forest Classifier** is trained using the generated TF-IDF features.

The model parameters are maintained in `params.yaml`:

```yaml
model_building:
  n_estimators: 25
  random_state: 2
```

The trained model is saved using Pickle:

```text
models/model.pkl
```

---

### 5. Model Evaluation

The final stage evaluates the trained model against the test dataset.

The evaluation stage can generate metrics such as:

* Accuracy
* Precision
* Recall
* F1 Score

The metrics are stored in:

```text
reports/metrics.json
```

---

# ⚙️ DVC Pipeline

This project uses **DVC (Data Version Control)** to manage data, parameters, pipeline stages, and reproducibility.

The pipeline is defined in:

```text
dvc.yaml
```

The pipeline contains:

```text
data_ingestion
        ↓
data_preprocessing
        ↓
feature_engineering
        ↓
model_building
        ↓
model_evaluation
```

---

## 📊 Parameters

The main machine learning parameters are maintained in `params.yaml`.

```yaml
data_ingestion:
  test_size: 0.2

feature_engineering:
  max_features: 50

model_building:
  n_estimators: 25
  random_state: 2
```

Keeping parameters outside the Python source code makes it easier to experiment with different configurations.

---

# 💻 Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd Complete_MLOPS_Pipeline
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

## 3. Activate the virtual environment

### Windows

```powershell
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Pipeline

Instead of running every stage manually, DVC can reproduce the complete pipeline.

Run:

```bash
dvc repro
```

DVC checks the dependencies, parameters, outputs, and previous pipeline state before executing the required stages.

---

# 🔍 Run Individual Stages

You can also execute individual stages manually.

### Data Ingestion

```bash
python src/data_ingestion.py
```

### Data Preprocessing

```bash
python src/data_processing.py
```

### Feature Engineering

```bash
python src/feature_engineering.py
```

### Model Building

```bash
python src/model_building.py
```

### Model Evaluation

```bash
python src/model_evaluation.py
```

---

# 📈 DVC Commands

### Reproduce the pipeline

```bash
dvc repro
```

### View the pipeline DAG

```bash
dvc dag
```

### Check pipeline status

```bash
dvc status
```

### Check parameter changes

```bash
dvc params diff
```

### Check metrics

```bash
dvc metrics show
```

### Track changes with Git

```bash
git status
```

---

# 🔁 Reproducibility

One of the main goals of this project is **reproducibility**.

If a parameter is changed in `params.yaml`, DVC can identify which stages are affected and rerun only the necessary parts of the pipeline.

For example:

```yaml
model_building:
  n_estimators: 50
  random_state: 2
```

After changing the parameter:

```bash
dvc repro
```

DVC can detect that the model-building stage needs to be executed again.

---

# 🧠 Key MLOps Concepts Demonstrated

This project demonstrates several important MLOps concepts:

* Machine Learning Pipeline
* Data Version Control
* Pipeline Reproducibility
* Parameter Management
* Data Processing
* Feature Engineering
* Model Training
* Model Evaluation
* Git Version Control
* Experiment Configuration
* Logging
* Pipeline Dependency Tracking

---

# 🎯 Learning Objectives

Through this project, the following concepts are demonstrated:

1. How to structure an ML project.
2. How to separate ML pipeline stages.
3. How to use DVC for pipeline management.
4. How to manage model parameters using YAML.
5. How to track data and model artifacts.
6. How to reproduce ML experiments.
7. How to integrate Git with an ML workflow.
8. How to build a basic end-to-end MLOps pipeline.

---

# 🔮 Future Improvements

The project can be extended with:

* MLflow experiment tracking
* DVC remote storage
* GitHub Actions CI/CD
* Docker containerization
* Model deployment using FastAPI
* Cloud deployment
* Automated model monitoring
* Data validation
* Model versioning
* Automated testing
* Continuous integration and deployment

---

# 👨‍💻 Author

**Nikhil Kant**

This project was built as a hands-on implementation of an end-to-end MLOps workflow using Python, Scikit-learn, Git, and DVC.

---

## ⭐ If you found this project useful

Feel free to explore the repository, experiment with the pipeline parameters, and extend the project with additional MLOps tools and deployment workflows.
