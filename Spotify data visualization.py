#spotify stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# creating dataframe
initial_dataframe = pd.read_csv("Spotify 2010 - 2019 Top 100.csv")

# df of independent variables
x = initial_dataframe.drop("top genre", axis=1)
# df of targets
y = initial_dataframe["top genre"]

        
# splitting of data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# dropping unnecessary columns
columns_dropped = ["title", "top genre", "artist", "artist type", "added", "year released", "top year"]
x_train = initial_dataframe.drop(columns_dropped, axis=1)
x_test = initial_dataframe.drop(columns_dropped, axis=1)
y_train = initial_dataframe.drop(columns_dropped, axis=1)
y_test = initial_dataframe.drop(columns_dropped, axis=1)

# Assigning each genre a number
unique_genres = initial_dataframe["top genre"].unique()
genre_to_label = {genre: label for label, genre in enumerate(unique_genres)} 
initial_dataframe["genre_label"] = initial_dataframe["top genre"].map(genre_to_label)

model = RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=7, random_state=42)
model.fit(x_train, y_train) 

#storing the predictions made from the trained model
y_pred = model.predict(x_test) 
y_pred = y_pred.flatten()

predicted_genre = []
numeric_labels = list(genre_to_label.values())

for num_pred in y_pred:
    for label in numeric_labels:
        if label == num_pred:
            # Find the corresponding genre for the numeric label
            predicted_genre.append(next((genre for genre, l in genre_to_label.items() if l == label), None))

print(predicted_genre)

# Create histograms for original numeric columns
plot_df = x_train.select_dtypes(include=['int64', 'float64', 'bool'])

for column in plot_df.columns:
    # Define data
    data = x_train[column]
    bins = np.linspace(min(data), max(data), 20)

    # Create histogram
    plt.hist(data, bins=bins, alpha=0.5, color='green')

    # Add labels and title
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column}')
    plt.show()

# Slice the first 100 elements of y_pred
y_pred_subset = y_pred[:50]

# Create a histogram for the limited subset of predicted genres
plt.hist(y_pred_subset, bins=len(unique_genres))  # Use the number of unique genres as bins
plt.xlabel('Predicted Genre Label')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Genres (First 100)')
plt.xticks(range(len(unique_genres)), unique_genres, rotation=90)  # Assign genre names to x-ticks
plt.show()


