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
import io

# Load dataset
st.title("Activitati economice")
st.write("Analiza Datelor")

# Upload file
uploaded_file = st.file_uploader("Choose file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)  
    st.write("Data Preview:")
    st.write(data.head())

    # Display dataframe info
    st.write("Dataframe Info:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    # Display column names
    st.write("Column Names:")
    st.write(data.columns.tolist())

    # Basic statistics
    st.write("Basic Statistics:")
    st.write(data.describe())
    # Correct the column names for sales revenues and other indicators
sales_revenues = df[['Venituri din vinzari, milioane lei_2021', 'Venituri din vinzari, milioane lei_2022', 'Venituri din vinzari, milioane lei_2023']]
profit_before_tax = df[['Rezultatul financiar soldat pina la impozitare. Profit (+) Pierdere (-), milioane lei_2021',
                        '_Rezultatul financiar soldat pina la impozitare. Profit (+) Pierdere (-), milioane lei_2022',
                        'Rezultatul financiar soldat pina la impozitare. Profit (+) Pierdere (-), milioane lei_2023']]

# Create plots
fig, axs = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle('Economic Indicators (2021-2023)', fontsize=16)

# Plot 1: Number of personnel
axs[0, 0].bar(years, personnel.mean(), color='skyblue')
axs[0, 0].set_title('Numărul de personal (thousands)')
axs[0, 0].set_xlabel('Year')
axs[0, 0].set_ylabel('Personnel (thousands)')

# Plot 2: Sales revenues
axs[0, 1].bar(years, sales_revenues.mean(), color='orange')
axs[0, 1].set_title('Venituri din vânzări (billion)')
axs[0, 1].set_xlabel('Year')
axs[0, 1].set_ylabel('Sales Revenues (billion)')

# Plot 3: Profit before tax
axs[1, 0].bar(years, profit_before_tax.mean(), color='green')
axs[1, 0].set_title('Profit înainte de impozitare (billion)')
axs[1, 0].set_xlabel('Year')
axs[1, 0].set_ylabel('Profit (billion)')

# Plot 4: Net Margin
# Assuming net margin data is not available, using synthetic data for demonstration
net_margin = [20, 18, 17]  # in percentage
axs[1, 1].plot(years, net_margin, marker='o', color='purple')
axs[1, 1].set_title('Net Margin (%)')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Net Margin (%)')

# Plot 5: Inflation Rate
# Assuming inflation rate data is not available, using synthetic data for demonstration
inflation_rate = [2.5, 3.0, 3.5]  # in percentage
axs[2, 0].plot(years, inflation_rate, marker='o', color='red')
axs[2, 0].set_title('Inflation Rate (%)')
axs[2, 0].set_xlabel('Year')
axs[2, 0].set_ylabel('Inflation Rate (%)')

# Hide the empty subplot
axs[2, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('economic_indicators_from_dataset.png')
plt.show()
print("Plots created and saved as 'economic_indicators_from_dataset.png'")
    
print("Streamlit app code executed successfully.")
