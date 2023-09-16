#spotify stuff
import pandas as pd
import matplotlib.pyplot as plt

def trendOverYears(dataframe, factor_to_measure):
    trend_by_year = dataframe.groupby('Year')[factor_to_measure].mean().reset_index()
    print(trend_by_year)
    plt.figure(figsize = (10, 6))
    plt.plot(trend_by_year["Year"], trend_by_year[factor_to_measure], marker = "o", linestyle = "-")
    plt.title("Danceability Trend over the years")
    plt.xlabel("Year")
    plt.ylabel("Average Danceability")
    plt.grid(True)
    plt.show()

    

initial_spotify_dataframe = pd.read_csv("SpotifyAllTime.csv")
# print (initial_dataframe)
# print (initial_dataframe.columns)

#data visualization
print("For what factor would you like to see the changes over the years? ", initial_spotify_dataframe.columns)
trendOverYears(initial_spotify_dataframe, "Danceability")


# scikit-learn
# regression for song popularity
# classification for genres