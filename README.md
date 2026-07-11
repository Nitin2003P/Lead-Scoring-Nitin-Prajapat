# Lead Scoring Project

## Live Demo

[Try the Streamlit App](https://lead-score-by-nitin-prajapat.streamlit.app/)

---

## About Project

This project is based on the Lead Scoring dataset of X Education.

The aim of this project is to find which leads have more chance to become customers.

I trained different machine learning models and compared their performance.

After comparison, **XGBoost** gave the best result, so I selected it as the final model.
I also created a Streamlit app where a user can upload a CSV file and get lead scores.

---

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

---

## Lead Categories

The app gives every lead a score between **0 and 100**.
- Hot Lead → 70 or above
- Warm Lead → 40 to 69
- Cold Lead → Below 40

---

## Project Files

```
Lead-Scoring-Project/

│── app.py
│── lead-scoring.ipynb
│── requirements.txt
│── README.md
│── .gitignore
└── model
    │── xgboost_model.pkl
    │── features.pkl
```

---

## How to Run

Install all libraries.

```bash
pip install -r requirements.txt
```

Run the app.

```bash
streamlit run app.py
```

Upload the CSV file.

The app will generate lead scores.

---

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit

---

## Author

Nitin Prajapat

B.Tech CSE

Poornima College of Engineering
