#spotify stuff
import pandas as pd
import matplotlib.pyplot as plt

def trendOverYears(dataframe, factor_to_measure):
    if factor_to_measure == "Top Genre":
        genre = input("Which genre would you like to visualize over time?\n")
        dataframe = dataframe[dataframe['Top Genre'] == genre]
        trend_by_year = dataframe.groupby("Year")["Popularity"].mean().reset_index()
        print(trend_by_year)
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
# print (initial_dataframe)
# print (initial_dataframe.columns)

#data visualization
features = list(initial_spotify_dataframe.columns[5:-1])
features.insert(0, "Top Genre")

print("For what factor would you like to see the changes over the years? \n",  features)
factor_to_measure = input()
trendOverYears(initial_spotify_dataframe, factor_to_measure)


# scikit-learn
# regression for song popularity
# classification for genres