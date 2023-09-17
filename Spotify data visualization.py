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

    # print(f"\n\nMean Absolute Error: {mae}")
    # print(f"Mean Squared Error: {mse}")
    # print(f"R-squared: {r2}\n\n")

    userChoice = ""

    while userChoice != "N" and r != True:
        print("Would you like to test a random song from the dataset? Y/N")
        userChoice = input()
        if userChoice == "N":
            break
        elif userChoice == "Y":
            num_rows = len(dataframe)
            random_index = random.randint(0, num_rows - 1)
            random_song = dataframe.iloc[random_index]

            random_song_info = []
            for i in range(len(features)):
                random_song_info.append(random_song[features[i]])

            random_song_Title = random_song["Title"]
            random_song_Artist = random_song["Artist"]
            random_song_Beats_Per_Minute = random_song["Beats Per Minute (BPM)"]
            random_song_energy = random_song["Energy"]
            random_song_danceability = random_song["Danceability"]
            random_song_loudness = random_song["Loudness (dB)"]
            random_song_liveness = random_song["Liveness"]
            random_song_valence = random_song["Valence"]
            random_song_acousticness = random_song["Acousticness"]
            random_song_speechiness = random_song["Speechiness"]

            random_song_popularity = random_song["Popularity"]
            
            
            #Beats Per Minute (BPM),Energy,Danceability,Loudness (dB),Liveness,Valence,Acousticness,Speechiness
            random_song_info = [random_song_energy, random_song_danceability, random_song_liveness, random_song_valence]
            print(f"Title {random_song_Title}\nArtist {random_song_Artist}")
            print(f"This random song's Energy, Danceability, Liveness, Valence are respectively:\n{random_song_energy},\n{random_song_danceability},\n{random_song_liveness},\n{random_song_valence}.")
            print(f"The actual popularity is: {random_song_popularity}.")
            random_song_popularity_prediction = predictPopularity(dataframe, features, True, random_song_info)
            print(f"The predicted popularity is: {random_song_popularity_prediction}.")
        else:
            print("Enter an appropiate input.\n")


def detect_anomalies(dataframe, features = ["Beats Per Minute (BPM)", "Energy", "Speechiness"]):
    X = dataframe[features]
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    model = IsolationForest(contamination = 0.01)
    model.fit(X_normalized)

    prediction = model.predict(X_normalized)
    anomaly_scores = model.decision_function(X_normalized)

    anomalies = dataframe.copy()
    anomalies['Anomaly Score'] = anomaly_scores

    # Filter anomalies based on the prediction
    anomalies = anomalies[prediction == -1]

    # Sort the anomalous songs by their anomaly score (most anomalous first)
    anomalies = anomalies.sort_values(by='Anomaly Score', ascending=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[features[0]], X[features[1]], X[features[2]], c=prediction, cmap="viridis")

    #labels for the 3d scatterplot 
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    plt.title("Anomaly Detection (3D)")
    plt.show()
    print("Anomalous Songs:")
    print(anomalies[["Title", "Artist", "Anomaly Score"] + features])
    print("\n\n")

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
        print("Top 10 genres: ")
        print (genres[:10])
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

print("Welcome to spotify data visualization.\n select an option below.")

userInput = " "
while userInput != "Quit":
    print("Enter \"Quit\" to exit.")
    print("Enter \"ty\" to visualize a specific trend over years.")
    print("Enter \"vc\" to visualize correlation.")
    print("Enter \"da\" for anomaly detection.")
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

    elif userInput == "pp":
        
        pred_choice = ""
        while pred_choice != "Y" or pred_choice != "N":
            print("\n Use default features? \"Energy\", \"Danceability\", \"Liveness\", \"Valence\" \n\n Enter Y/N\n\n")
            pred_choice = input()
            if pred_choice == "Y":
                predictPopularity(initial_spotify_dataframe)
            elif pred_choice == "N":
                print ("How many features do you want to use?\nEnter a number no greater than 8")
                print ("\n\n\nThis branch is not finished.\n\n\nrestart and Choose Y")
                num_features = int(input())
                features_list = []
                print("Features you can use: Beats Per Minute (BPM), Energy, Danceability, Loudness (dB), Liveness,Valence,Acousticness, Speechiness,\n You must enter the feature exactly as show here.\n\n")
                for i in range(num_features):
                    print(f"Enter feature {i}. \n")
                    input_feature = input()
                    features_list.append(input_feature)
                predictPopularity(initial_spotify_dataframe, features_list)
            else:
                print("Enter a correct command.")

    elif userInput == "da":
        detect_anomalies(initial_spotify_dataframe)
    else:
        print("Enter a correct choice.\n\n")
        

    