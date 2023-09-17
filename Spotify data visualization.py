#spotify stuff
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random


def predictPopularity(dataframe, features = ["Energy", "Danceability", "Liveness", "Valence"], r = False, random_song_info = []):      #predicts popularity based on Energy, Danceability, Liveness, and Valence
    X = dataframe[features]
    y = dataframe['Popularity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    if random_song_info:
        model.fit(X, y)
        song_features = pd.DataFrame([random_song_info], columns = features)
        predicted_popularity = model.predict(song_features)
        return predicted_popularity[0]

    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}\n\n")

    userChoice = ""

    while userChoice != "N" and r != True:
        print("Test a random song from the dataset? Y/N")
        userChoice = input()
        if userChoice == "N":
            break
        elif userChoice == "Y":
            num_rows = len(dataframe)
            random_index = random.randint(0, num_rows - 1)
            random_song = dataframe.iloc[random_index]
            random_song_popularity = random_song["Popularity"]
            random_song_energy = random_song["Energy"]
            random_song_danceability = random_song["Danceability"]
            random_song_liveness = random_song["Liveness"]
            random_song_valence = random_song["Valence"]
            random_song_info = [random_song_energy, random_song_danceability, random_song_liveness, random_song_valence]
            print(f"This random song's Energy, Danceability, Liveness, Valence are respectively:\n{random_song_energy},\n{random_song_danceability},\n{random_song_liveness},\n{random_song_valence}.")
            print(f"The actual popularity is: {random_song_popularity}.")
            random_song_popularity_prediction = predictPopularity(dataframe, features, True, random_song_info)
            print(f"The predicted popularity is: {random_song_popularity_prediction}.")
        else:
            print("Enter an appropiate input.\n")

# def detect_anomalies(dataframe, features):
#     X = dataframe[features]
#     scaler = StandardScaler()
#     X_normalized = scaler.fit_transform(X)

#     model = IsolationForest(contamination = 0.01)
#     model.fit(X_normalized)

#     prediction = model.predict(X_normalized)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(X["Beats Per Minute (BPM)"], X["Energy"], X["Danceability"], c=prediction, cmap='viridis')

#     ax.set_xlabel("Beats Per Minute (BPM)")
#     ax.set_ylabel("Energy")
#     ax.set_zlabel("Danceability")
#     plt.title("Anomaly Detection (3D)")
#     plt.show()

def visualize_correlation(dataframe, x, y):
    #correlation between two variables
    correlation_dataframe = dataframe.groupby(x)[y].mean().reset_index()
    # print(correlation_dataframe)
    plt.figure(figsize = (10, 6))
    plt.plot(correlation_dataframe[x], correlation_dataframe[y], marker = "o", linestyle = "-")
    plt.title("Plotchart")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()


def visualize_trend_over_years(dataframe, factor_to_measure): #Currently not working with Length (Duration)
    if factor_to_measure == "Top Genre":
        genres_counts = dataframe["Top Genre"].value_counts()
        genres = genres_counts.index.tolist()
        print (genres)
        genre = input("Which genre would you like to visualize over time?\nThe list above is sorted from most to least popular genres.\n")
        dataframe = dataframe[dataframe['Top Genre'] == genre]
        trend_by_year = dataframe.groupby("Year")["Popularity"].mean().reset_index()
        plt.figure(figsize = (10, 6))
        plt.plot(trend_by_year["Year"], trend_by_year["Popularity"], marker = "o", linestyle = "-")
        plt.title(f"{genre} over the years")
        plt.xlabel("Year")
        plt.ylabel("Popularity")        
        plt.grid(True)
        plt.show()

    else:
        trend_by_year = dataframe.groupby('Year')[factor_to_measure].mean().reset_index()
        plt.figure(figsize = (10, 6))
        plt.plot(trend_by_year["Year"], trend_by_year[factor_to_measure], marker = "o", linestyle = "-")
        plt.title(f"{factor_to_measure} trend over the years")
        plt.xlabel("Year")
        plt.ylabel(factor_to_measure)
        plt.grid(True)
        plt.show()

initial_spotify_dataframe = pd.read_csv("SpotifyAllTime.csv")
features_year_trend = list(initial_spotify_dataframe.columns[5:])
features_year_trend.insert(0, "Top Genre")

features_vc = list(initial_spotify_dataframe.columns[5:])

# data visualization

# trendOverYears(initial_spotify_dataframe, factor_to_measure)
print("Welcome to spotify data visualization.\n select an option below.")

userInput = " "
while userInput != "Quit":
    print("Enter \"Quit\" to exit.")
    print("Enter \"ty\" to visualize a specific trend over years.")
    print("Enter \"vc\" to visualize correlation.")
    # print("Enter \"ad\" for anomaly detection.")
    print("Enter \"pp\" for popularity prediction.")
    userInput = input()
    if userInput == "Quit":
        break
    elif userInput == "ty":
        print("For what factor would you like to see the changes over the years? \n",  features_year_trend)
        factor_to_measure = input() 
        visualize_trend_over_years(initial_spotify_dataframe, factor_to_measure)

    elif userInput == "vc":
        print("What two features would you like to compare?\n Features: ")
        print(features_vc)
        x = input("Enter the x axis.\n")
        y = input("Enter the y axis.\n")
        visualize_correlation(initial_spotify_dataframe, x, y)

    # elif userInput == "ad":
    #     print("Enter 3 ")
    #     detect_anomalies(initial_spotify_dataframe, features)

    elif userInput == "pp":
        predictPopularity(initial_spotify_dataframe)
    else:
        print("Enter a correct choice.\n")
        

    