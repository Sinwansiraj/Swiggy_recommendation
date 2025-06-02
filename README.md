# Swiggy Restaurant Recommendation System

> 🍽️ A lightweight recommendation engine for Indian restaurants, powered by Python, scikit-learn, and Streamlit.

---

## 🚀 Project Overview

This project ingests a raw Swiggy dataset (`swiggy.csv`), cleans and preprocesses the data, and exposes two main functionalities:

1. **Recommendation Methodology (Notebooks)**  
   - **Data Cleaning & Preprocessing**  
     - Strip out messy strings (`"--"`, `"Too Few Ratings"`, currency symbols, etc.)  
     - Convert `rating`, `rating_count`, and `cost` into numeric values (with median imputation)  
     - Drop rows missing critical fields (`name`, `city`, `cuisine`)  
   - **Feature Engineering**  
     - One-hot encode categorical features (`city`, `cuisine`)  
     - Concatenate with numeric features to create a final feature matrix  
   - **Clustering & Similarity**  
     - K-Means clustering for “grouping” similar restaurants  
     - Cosine-based NearestNeighbors for “top-5” recommendations within the same city + cuisine  

2. **Interactive Streamlit App**  
   - Users can filter by **city**, **cuisine**, **rating**, and **cost**  
   - After filtering, pick a restaurant from the list and get **5 similar recommendations** in the same city & cuisine  
   - All heavy lifting (data cleaning + feature prep) happens under the hood the first time you run the app

---

## 📂 Repository Structure
swiggy_project/
├── data/processed
│
│    ├── swiggy.csv 
│    ├── cleaned_data.csv 
│    ├── numerical_data.csv 
│    ├── encoder.pkl # Fitted OneHotEncoder (city + cuisine)
│    ├── kmeans_model.pkl # KMeans clusters
│    └──nn_model.pkl # NearestNeighbors model 
│ 
├── notebook/
│ ├── Generate_Clean_Numerical_Encoder.ipynb
│ └──  Recommendation_Methodology.ipynb
│ 
├── streamlit_app.py 
├── .gitignore
├── README.md
└── requirements.txt 
