#######################
# Import libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

#######################
# Page configuration
st.set_page_config(
    page_title="Dota 2 Pro Meta Team Maker and Win rate predictor", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
#######################


# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Dota 2 Pro Meta Team Maker and Win rate predictor')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Marius Andre Manguerra\n2. Dharl Russel Perez\n3. Seth Lagman\n4. Nathan Panes\n5. Josh Metran")
#######################
# Data

# Load data
dataset = pd.read_csv("Current_Pro_meta.csv")

# 1 SPLIT ROLES
dataset['Roles_List'] = dataset['Roles'].str.split(',')

# 2 PICK RATE
total_picks = dataset['Times Picked'].sum()
dataset['Pick Rate (%)'] = (dataset['Times Picked'] / total_picks) * 100

# 3 BAN RATE
total_bans= dataset['Times Banned'].sum()
dataset['Ban Rate (%)'] = (dataset['Times Banned'] / total_bans) * 100

# 4 CONTESTATION RATE
dataset['Contestation Rate (%)'] = dataset['Pick Rate (%)'] + dataset['Ban Rate (%)']

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.markdown("""
        Just a streamtlit web app that shows some **Exploratory Data Analysis (EDA)**, **Data Pre-processing** and usage of **Clustering and Linear Regression** 
        to make a Dota 2 team compostion and predict a team's win rate based on the type of heroes in the team
    
    """)
    st.markdown("### Contents")
    st.markdown("```Dataset``` - A comprehensive dataset containing information about Dota 2 heroes picked in tournaments, including pick rates, win rates, and various performance metrics.")
    st.markdown("```EDA``` - Exploratory Data Analysis of the Dota 2 hero dataset, showcasing insights such as hero pick frequencies, average win rates, synergy with other heroes, and tournament-specific trends.")
    st.markdown("```Data Cleaning``` - A process where we clean and preprocess the raw data from Dota 2 tournaments, handling missing values, duplicates, and ensuring the dataset is ready for analysis.")
    st.markdown("""
    ```Machine Learning``` - We used clustering algorithms to analyze and suggest optimal team compositions based on key factors such as win rate, contestation rate, ban rate, and pick rate from professional Dota 2 tournaments. 
    The model also considers the common team structure of 3 carry heroes and 2 support heroes. Additionally, we applied Linear Regression to predict the win rate of a team composition, factoring in contestation rates and the 
    various roles of the heroes in the team.
    """)
    st.markdown("```Prediction``` - Predicts winrate of a team composition")
    st.markdown("```Conclusion``` - Summary and conclusion of all the findings within the dataset, EDA, and models")

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.write("Dota 2 Heroes Professional Meta Analysis")
    st.write("")

    ## Your content for your DATASET page goes here

    st.write("""
    This page presents data on Dota 2 heroes based on their professional match performance. Primary attributes, attack types, roles, and match performance metrics like win rate, pick, and ban counts are all included in each hero's statistics.
    """)

    st.write("""
    The dataset offers information on the viability of heroes with a variety of roles, including Carry, Support, Durable, Nuker, and more, in competitive play. Additionally, it contains clues about niche heroes who might have specific applications.

    """)

    st.subheader("Dataset Columns")
    st.write("""
    The columns in this dataset are:
    - **Name**: Hero name
    - **Primary Attribute**: Main attribute (Strength, Agility, or Intelligence)
    - **Attack Type**: Whether the hero is Melee or Ranged
    - **Attack Range**: Distance at which the hero can attack
    - **Roles**: Various roles assigned to the hero
    - **Total Pro Wins**: Number of professional wins
    - **Times Picked**: Number of picks in professional matches
    - **Times Banned**: Number of bans in professional matches
    - **Win Rate**: Win rate percentage in professional matches
    - **Niche Hero?**: Indicates if the hero is considered niche
    """)

    #Load DataSet
    st.header("Dataset Preview")
    dataset = pd.read_csv('Current_Pro_meta.csv')
    dataset

    #Describe
    st.header("Descriptive Statistics")
    dataset.describe()

    st.write("""
    The summary statistics table provides a breakdown of the main performance metrics:

    - **Attack Range**: Indicates whether heroes are melee or ranged, with higher values for ranged heroes.
    - **Total Pro Wins**: Shows the average number of professional wins for each hero, with variability indicating consistency or volatility in performance.
    - **Times Picked**: Reflects hero popularity in professional matches, with high values indicating frequently chosen heroes.
    - **Times Banned**: Points to how often heroes are banned, potentially due to their perceived strength or specific meta relevance.
    - **Win Rate**: Average win rate across matches, showing which heroes have the highest success rates.
    
    This analysis helps highlight which heroes dominate the competitive scene and which are more situational picks.
    """)

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1, 1, 1), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown("Average Pick Rate by Role")
        df_exploded = dataset.explode('Roles_List')

        # Stripping any extra spaces from the roles
        df_exploded['Roles_List'] = df_exploded['Roles_List'].str.strip()

        # Calculating the average pick rate by role
        pick_rate_by_role = df_exploded.groupby('Roles_List')['Times Picked'].mean().reset_index()
        pick_rate_by_role.columns = ['Role', 'Average Pick Rate']

        # Sorting the results by average pick rate
        pick_rate_by_role = pick_rate_by_role.sort_values(by='Average Pick Rate', ascending=False)

        # Plotting the bar chart using seaborn
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Average Pick Rate', y='Role', data=pick_rate_by_role, hue='Role', dodge=False, legend=False, palette='viridis')
        plt.title('Average Pick Rate by Role')
        plt.xlabel('Average Pick Rate')
        plt.ylabel('Role')

        # Display the plot in Streamlit
        st.pyplot(plt)

        st.text("""
        Now we can see here some of the most picked roles within the tournament having Initiator's as the most picked by 204.14 followed by 
        Supports and Disablers. It is something to mention that since Heroes can have multiple different roles that these statistics can be 
        skewed as one hero could have at least 5 different roles at most sometimes 6. But for now we can see that current Pro's really priortized 
        a good Initiator, Support and Disabler as shown here in the bar plot.
        """)

        st.markdown('Average Win Rate by Role')
        # Split roles and calculate the average win rate by role
        df_roles = dataset[['Roles', 'Win Rate']].dropna()

        # Split the Roles column into multiple rows
        df_roles = df_roles.assign(Roles=df_roles['Roles'].str.split(',')).explode('Roles')
        df_roles['Roles'] = df_roles['Roles'].str.strip()  # Remove any leading/trailing whitespace

        win_rate_by_role = df_roles.groupby('Roles')['Win Rate'].mean().reset_index()
        win_rate_by_role.columns = ['Role', 'Average Win Rate']
        win_rate_by_role = win_rate_by_role.sort_values(by='Average Win Rate', ascending=False).reset_index(drop=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Average Win Rate', y='Role', data=win_rate_by_role, hue='Role', dodge=False, legend=False, palette='viridis')
        plt.title('Average Win Rate by Role')
        plt.xlabel('Average Win Rate')
        plt.ylabel('Role')
        st.pyplot(plt)

        st.text("""
        So now with these we can now also find the average win rate of each role in the game. As we can see that Pusher style Heroes have the highest average Win 
        Rate of all but this statistics is skewed by the fact that it has a lower pick rate than the rest of the roles. We can then see that Carries have the lowest average 
        Win rate along with Initiators. Initiators having a lower average win rate can be explained due to them having a higher pick rate meaning they have on average, a higher chance of having 
        more losses. For carries, Since they have both low average win rate and low average pick rate, we could conclude that they were weak in the current meta of the tournament.
        """)

        st.markdown("Hero Win Rates")

        hero_Win_Rate = dataset[['Name', 'Win Rate']].sort_values(by='Win Rate', ascending=True)
        plt.figure(figsize=(6, 25))
        sns.barplot(x='Win Rate', y='Name', data=hero_Win_Rate, hue='Name', dodge=False, legend=False, palette='viridis')

        # Add titles and labels
        plt.title('Hero Win Rates', fontsize=16)
        plt.xlabel('Win Rate', fontsize=14)
        plt.ylabel('Heroes', fontsize=14)

        st.pyplot(plt)

        st.text("""
        We can See here now the different Win rates of different Heroes. Meepo, Chen, Anti-Mage, Queen of Pain and Nature's Prophet being the top 5 winning heroes in the tournament. 
        As for the least winning Heroes we have Riki, Abaddon, Magnus, Tinker and Batrider. But this data is not sufficient enough to make conclusions as we do not know yet how many times 
        these heroes were picked meaning they could have inflated statistics as they didn't play as many games meaning they have less chances of having losing.
        """)


    with col[1]:
        st.markdown('Hero Pick Rates in Dota 2')
        df_sorted = dataset[['Name', 'Times Picked', 'Pick Rate (%)']].sort_values(by='Pick Rate (%)', ascending=True)
        # Create a bar plot
        plt.figure(figsize=(6, 25))
        sns.barplot(x='Pick Rate (%)', y='Name', data=df_sorted, hue='Name', dodge=False, legend=False, palette='viridis')

        # Add titles and labels
        plt.title('Hero Pick Rates in Dota 2', fontsize=16)
        plt.xlabel('Pick Rate (%)', fontsize=14)
        plt.ylabel('Heroes', fontsize=14)
        st.pyplot(plt)

        st.text("""
        We can see here now the the most picked heroes by ascending value. We can see that the top 5 most picked heroes are Vengeful Spirit at 801 picks and 3.51% pick rate, Sven at 591 picks and 
        2.59% pick rate, Phantom Assasin at 542 picks and 2.38% pick rate, Rubick at 539 picks and 2.36% pick rate and lastly Warlock at 517 picks and 2.27% pick rate. As for the least picked heroes we 
        can see that Meepo is the least picked heroes at only 5 picks at 0.02% pick rate, followed by Arc Warden with 14 and 0.06% pick rate, Enigma at 19 picks and 0.08% pick rate, Broodmother at 21 picks and 
        0.09% pick rate and Abaddon at 24 picks and 0.11% pick rate.
        """)

        st.markdown("Contestation Rate")
        WinRate_Vs_ContestationRate = dataset[['Name', 'Contestation Rate (%)','Win Rate']].sort_values(by='Contestation Rate (%)', ascending=True)

        WinRate_Vs_ContestationRate.plot(kind='scatter', x='Contestation Rate (%)', y='Win Rate', s=32, alpha=.8)
        plt.title('Contestation Rate (%) vs Win Rate')
        plt.xlabel('Contestation Rate (%)')
        plt.ylabel('Win Rate')
        plt.gca().spines[['top', 'right',]].set_visible(False)
        st.pyplot(plt)

        st.text("""
        Due to limitations (and for clarity), We cannot show the names of the heroes but we can just cross reference the contestation rates in the previous graphs.
        As we can see here, The most contested heroes actually do have higher win rates on average. Such as the highest contested Hero which is Vengeful spirit having an above 50% win rate which considering that they were picked so much, 
        that's a really good win rate. The 2nd most contested Hero in Invoker as well having also an Above 50% win rate showing why the pros preferred pick or banning these heroes.
        Although there actually is an outlier value here which is the highest win rate in Meepo but only ever being picked 5 times in the whole tournament. Out of 5 games, Meepo won 4 of them.
        As for the least contested Heroes, we can see that some of them do actually have positive win rates but some of them also have negative win rates (Positive being above 50% while Negative being below 50%). Such as Arc Warden 
        who has a contest rate of 0.15% also only having 33% win rate. Or Abaddon, One of the least contested Heroes also only having 33% win rate. We can conlcude that these Heroes are picked and banned less due to their nature of likely being niche 
        counter pick heroes and having better options. Although some least contested heroes have positie win rates, a majority of them do not as we can see that the higher the contest rate, the higher likelihood of them having a positive win rate
        """)

    with col[2]:
        st.markdown("Hero Ban Rates in Dota 2")
        total_bans= dataset['Times Banned'].sum()

        # Calculate the ban rate for each hero
        dataset['Ban Rate (%)'] = (dataset['Times Banned'] / total_bans) * 100

        # Sort the DataFrame by 'ban Rate (%)' in ascending order
        df_sorted = dataset[['Name', 'Times Banned', 'Ban Rate (%)']].sort_values(by='Ban Rate (%)', ascending=True)
        plt.figure(figsize=(6, 25))
        sns.barplot(x='Ban Rate (%)', y='Name', data=df_sorted, hue='Name', dodge=False, legend=False, palette='viridis')

        # Add titles and labels
        plt.title('Hero Ban Rates in Dota 2', fontsize=16)
        plt.xlabel('Ban Rate (%)', fontsize=14)
        plt.ylabel('Heroes', fontsize=14)

        st.pyplot(plt)

        st.text("""
        Now with this we can see the top 5 most and least banned heroes in the tournament. For the top 5 Most banned we have Invoker with 1049 bans and 3.3% ban rate, Followed by Pangolier with 974 bans and 3.08% ban rate, 
        then Brewmaster with 964 and 3.05% ban rate, Dawnbreaker by 963 and 3.04% ban rate and lastly Nature's Prophet at 946 2.99% ban rate. As for the least banned heroes we have Elder Titan at 18 bans 0.06% ban rate, Alchemist 
        with 22 bans 0.07% ban rate, Luna with 26 bans and 0.08% ban rate, Shadow Shaman at 27 and 0.09% ban rate and lastly Disruptor at 28 bans and 0.09% ban rate. Now that we have the ban rate and pick rate of each Hero, We can now 
        find the contest rate of each hero by adding the 2 values and we can see which Hero was the most contested in the Tournament
        """)

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    # Your content for the DATA CLEANING / PREPROCESSING page goes here
    st.dataframe(dataset.head())

    st.write("""
    Since we'll mostly ever be looking only at win rates, pick rates and ban rates, 
             we will be dropping most every other column except for those that we need.
    """)

    roles_split = dataset['Roles'].str.get_dummies(sep=', ')
    dataset = pd.concat([dataset.drop(columns=['Roles']), roles_split], axis=1)

    # Drop unnecessary columns
    columns_to_drop = ['Unnamed: 0', 'Primary Attribute', 'Attack Type', 
                       'Attack Range', 'Roles', 'Total Pro wins', 
                       'Times Picked', 'Times Banned', 'Roles_List',
                       'Niche Hero?'
                       ]
    dataset = dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns])
    st.dataframe(dataset.head())

    features = ['Contestation Rate (%)','Pick Rate (%)', 'Ban Rate (%)', 'Win Rate']

    new = dataset[features]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(new)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    
    st.dataframe(X_scaled_df.head())

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    st.subheader("Linear Regression Model")

    st.markdown("""
    On this page, we will build a **Linear Regression** model to predict the win rate of a Dota 2 team composition.
    We will use features such as **Pick Rate**, **Ban Rate**, and **Contestation Rate** to train the model.
    Linear regression will help us establish a relationship between these features and the win rate of the team.
    """)

    X = dataset[['Pick Rate (%)', 'Ban Rate (%)', 'Contestation Rate (%)']]
    y = dataset['Win Rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Predicted Win Rate: {y_pred.mean():.2f}")

    st.markdown("### Actual vs Predicted Win Rate")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([0, 100], [0, 100], color='red', linestyle='--')
    plt.title("Actual vs Predicted Win Rate")
    plt.xlabel("Actual Win Rate")
    plt.ylabel("Predicted Win Rate")
    st.pyplot(plt)

    st.markdown("""
    We also made use of Linear regression to predict the win rate of a Team composition
    based on the win rates based on contestation rate and the many different roles.
    """)

    st.subheader("K-Means Model")

    st.markdown(""" In this section, we will apply a **K-Means Clustering** algorithm to identify hero clusters based on their performance metrics, 
    including **Contestation Rate**, **Pick Rate**, **Ban Rate**, and **Win Rate**. 
    We will then use these clusters to generate an optimized team composition for Dota 2 by selecting heroes from different clusters while ensuring the usual team structure of 3 carry heroes and 2 supports.""")   

    roles_split = dataset['Roles'].str.get_dummies(sep=', ')
    dataset = pd.concat([dataset.drop(columns=['Roles']), roles_split], axis=1)
    dataset = dataset.drop(columns=['Unnamed: 0', 'Primary Attribute', 'Attack Type', 'Attack Range', 
                                    'Total Pro wins', 'Times Picked', 'Times Banned', 'Roles_List'])

    features = ['Contestation Rate (%)', 'Pick Rate (%)', 'Ban Rate (%)', 'Win Rate']
    X = dataset[features]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)
    dataset['Cluster'] = kmeans.labels_

    def select_team(dataset):
        team = []
        clusters_represented = set()
        num_carries = 0
        num_supports = 0

        for cluster in range(5):
            hero_options = dataset[dataset.Cluster == cluster].sort_values('Contestation Rate (%)', ascending=False)

            for index in hero_options.index:
                hero = hero_options.loc[index]

                if 'Carry' in hero.index and hero['Carry'] and num_carries < 3:
                    team.append(hero['Name'])
                    clusters_represented.add(cluster)
                    num_carries += 1
                    break
                elif 'Support' in hero.index and hero['Support'] and num_supports < 2:
                    team.append(hero['Name'])
                    clusters_represented.add(cluster)
                    num_supports += 1
                    break
            else:
                remaining_heroes = dataset[~dataset.Name.isin(team)].sort_values('Win Rate', ascending=False)
                for index in remaining_heroes.index:
                    hero = remaining_heroes.loc[index]
                    if 'Carry' in hero.index and hero['Carry'] and num_carries < 3:
                        team.append(hero['Name'])
                        num_carries += 1
                        break
                    elif 'Support' in hero.index and hero['Support'] and num_supports < 2:
                        team.append(hero['Name'])
                        num_supports += 1
                        break
                else:
                    hero = dataset[~dataset.Name.isin(team)].sample(1).iloc[0]
                    team.append(hero['Name'])
        return team

    if st.button("Generate Team"):
        team = select_team(dataset)
        team_df = pd.DataFrame(team, columns=['Selected Heroes'])
        st.write("Selected team:")
        st.table(team_df)

        st.markdown("""Here is now the Machine learning models that we came up with using the dataset.
                   Using clustering it comes up with a team composition based on the win rate, contestation rate, 
                   ban rate and pick rate of the pro teams. It also will account for usual team composition of 3 carry heroes and 2 supports.""")
    

#  Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

# Your content for the PREDICTION page goes here

    # 1 SPLIT ROLES
    dataset['Roles_List'] = dataset['Roles'].str.split(',')

    # 2 PICK RATE
    total_picks = dataset['Times Picked'].sum()
    dataset['Pick Rate (%)'] = (dataset['Times Picked'] / total_picks) * 100

    # 3 BAN RATE
    total_bans= dataset['Times Banned'].sum()
    dataset['Ban Rate (%)'] = (dataset['Times Banned'] / total_bans) * 100

    # 4 CONTESTATION RATE
    dataset['Contestation Rate (%)'] = dataset['Pick Rate (%)'] + dataset['Ban Rate (%)']
    roles_split = dataset['Roles'].str.get_dummies(sep=', ')
    dataset = pd.concat([dataset.drop(columns=['Roles']), roles_split], axis=1)

    # Drop unnecessary columns
    columns_to_drop = ['Unnamed: 0', 'Primary Attribute', 'Attack Type', 
                       'Attack Range', 'Roles', 'Total Pro wins', 
                       'Times Picked', 'Times Banned', 'Roles_List',
                       'Niche Hero?'
                       ]
    dataset = dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns])

    features = ['Contestation Rate (%)','Pick Rate (%)', 'Ban Rate (%)', 'Win Rate']

    new = dataset[features]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(new)

    X = dataset[['Contestation Rate (%)','Carry', 'Support', 'Disabler', 'Nuker', 'Pusher','Initiator','Escape','Durable']]
    y = dataset['Win Rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    col_pred = st.columns((1.5, 3, 3), gap='medium')

    # Initialize session state for clearing results
    if 'clear' not in st.session_state:
        st.session_state.clear = False

    with col_pred[0]:
        with st.expander('Options', expanded=True):
            show_dataset = st.checkbox('Show Dataset')
            clear_results = st.button('Clear Results', key='clear_results')

            if clear_results:
                st.session_state.clear = True

    with col_pred[1]:
        st.markdown("#### Linear Regression (Predicting Winrate)")

        contestation_rate = st.number_input('Contestation Rate', min_value=0.0, max_value=5.0, step=1.00, key='contestation_rate', value=0.0 if st.session_state.clear else st.session_state.get('contestation_rate', 0.0))
        carry = st.number_input('Number of Carry', min_value=0.0, max_value=5.0, step=1.00, key='carry', value=0.0 if st.session_state.clear else st.session_state.get('carry', 0.0))
        support = st.number_input('Number of Support', min_value=0.0, max_value=5.0, step=1.00, key='support', value=0.0 if st.session_state.clear else st.session_state.get('support', 0.0))
        disabler = st.number_input('Number of Disabler', min_value=0.0, max_value=5.0, step=1.00, key='disabler', value=0.0 if st.session_state.clear else st.session_state.get('disabler', 0.0))
        nuker = st.number_input('Number of Nuker', min_value=0.0, max_value=5.0, step=1.00, key='nuker', value=0.0 if st.session_state.clear else st.session_state.get('nuker', 0.0))
        pusher = st.number_input('Number of Pusher', min_value=0.0, max_value=5.0, step=1.00, key='pusher', value=0.0 if st.session_state.clear else st.session_state.get('pusher', 0.0))
        initiator = st.number_input('Number of Initiator', min_value=0.0, max_value=5.0, step=1.00, key='initiator', value=0.0 if st.session_state.clear else st.session_state.get('initiator', 0.0))
        escape = st.number_input('Number of Escape', min_value=0.0, max_value=5.0, step=1.00, key='escape', value=0.0 if st.session_state.clear else st.session_state.get('escape', 0.0))
        durable = st.number_input('Number of Durable', min_value=0.0, max_value=5.0, step=1.00, key='durable', value=0.0 if st.session_state.clear else st.session_state.get('durable', 0.0))

        if st.button('Detect', key='dt_detect'):
            # Prepare the input data for prediction
            new_composition = pd.DataFrame({'Contestation Rate (%)': [contestation_rate] ,'Carry': [carry], 'Support': [support],'Disabler' : [disabler], 'Nuker': [nuker], 'Pusher': [pusher], 'Initiator': [initiator], 'Escape': [escape], 'Durable': [durable]})
            predicted_win_rate = model.predict(new_composition)
            st.markdown(f"Predicted Win Rate: {predicted_win_rate}")

        if show_dataset:
            # Display the dataset
            st.subheader("Dataset")
            st.dataframe(dataset, use_container_width=True, hide_index=True)
            
# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")
    st.markdown("""
                    
        Through exploratory data analysis and training of two classification models (`Decision Tree Classifier` and `Random Forest Regressor`) on the **Iris Flower dataset**, the key insights and observations are:

        #### 1. 📊 **Exploratory data analysis**:
        - The dataset showed the many different winrates, pickrates and banrates of each hero during a pro dota 2 tournament. We find that there are some heroes that are picked more often and banned more often than some heroes and the popularly picked heroes actually have positive win rates. 
        - The dataset shows as well the roles of each Hero and their main attributes so that we can learn more about the effectiveness of each role. We find that the most effective role was the Initiator and Support role. 

        #### 2. 📝 **Model Training (Linear Regression)**:
        - The Linear Regression model Reached conclusions with 38% mean squared error which does mean it is quite inaccurate. It predicts a team's winrate based on the winrates of each separate role within the game. 
        - Since the dataset also doesn't account for Counter Matchups and Player knowledge, It also furthers the inaccuracy of this model.

        #### 3. 📈 **Model Training (K-Means Clustering)**:

        - With this model it just takes the dataset and suggests a team composition based on the individual win rate of the hero as well as accounting for having the standard team composition of 3 carry heroes and 2 Support heroes
        - As mentioned before, It does not account for counter matchups and player skill so it is not too accurate. 

        ##### **Final Conclusion** :
        While this Dataset has provided us with a lot of good insights into pro player picks, hero win rates and what hero was good during a tournament patch, It was not a good dataset to use machine learning model with as it was very limited with its data. 

        It did not have any data for Matchup win rates I.e Vengeful spirit vs Earthshaker or Invoker vs Sniper. It also does not account for item meta as well which can also change the game meta due to certain item hero synergies. 

        The models that we have trained for this project is very limited and can only serve as a suggestion and not as full proof win immediate prediction so all in all, We have learnt a lot about Hero metas but was not able to make a confidently good Machine learning model due to the chosen dataset's limited nature.
                    
     """)
    # Your content for the CONCLUSION page goes here
