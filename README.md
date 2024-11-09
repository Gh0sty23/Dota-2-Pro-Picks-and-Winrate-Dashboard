# Dota 2 Team Win Predictor and Team maker

A streamlit app that makes use of a pro tournament dataset of win rates, ban rates, pick rates etc. to come to conclusions on what the best team composition would look like and what the winrate of a team would be depending on its composition.

Made by 
Manguerra, Marius Andre
Perez, Dharl Russel
Lagman, Seth
Panes, Nathan
Metran, Josh

Streamlit Web App Link
https://dota2datascienceproject.streamlit.app/

Google Colab Notebook Link
https://colab.research.google.com/drive/1EbLrdfMkCboJBYdMgTgJatbaDIYFvHkF#scrollTo=y-1SZ4J29Hhr

Dataset 
https://www.kaggle.com/datasets/nihalbarua/dota2-competitive-picks?resource=download

Pages: 

EDA - Exploratory Data Analysis of the Dota 2 hero dataset, showcasing insights such as hero pick frequencies, average win rates, synergy with other heroes, and tournament-specific trends.

Data Cleaning - A process where we clean and preprocess the raw data from Dota 2 tournaments, handling missing values, duplicates, and ensuring the dataset is ready for analysis.

Machine Learning - We used clustering algorithms to analyze and suggest optimal team compositions based on key factors such as win rate, contestation rate, ban rate, and pick rate from professional Dota 2 tournaments. The model also considers the common team structure of 3 carry heroes and 2 support heroes. Additionally, we applied Linear Regression to predict the win rate of a team composition, factoring in contestation rates and the various roles of the heroes in the team.

Prediction - Predicts winrate of a team composition

Conclusion - Summary and conclusion of all the findings within the dataset, EDA, and models
   ```
