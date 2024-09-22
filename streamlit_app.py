import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load dataset
st.title("Activitati economice")
st.write("Analiza Datelor")
# Upload file
uploaded_file = st.file_uploader("Choose file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)  
    print(df.head())

    # Display dataframe info
    print("\
    Dataframe Info:")
    print(df.info())
    
    # Display column names
    print("\
    Column Names:")
    print(df.columns)
    

