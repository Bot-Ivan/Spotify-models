#spotify stuff
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# creating dataframe
initial_dataframe = pd.read_csv("Spotify 2010 - 2019 Top 100.csv")

# df of independent variables
x = initial_dataframe.drop("top genre", axis=1)
# df of targets
y = initial_dataframe["top genre"]


avg_score = 0.0
n = 30
        
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

# y_pred = list(y_pred)

for num_pred in y_pred:
    for label in numeric_labels:
        if label == num_pred:
            # Find the corresponding genre for the numeric label
            predicted_genre.append(next((genre for genre, l in genre_to_label.items() if l == label), None))

# Now, predicted_genre contains the corresponding genres for the predicted labels
print(predicted_genre)

            
    # # Map each integer label to the corresponding genre string
    # predicted_genre = [genre for genre, label in genre_to_label.items() if label == num_pred]

    # if predicted_genre:
    #     # If there is a predicted genre (non-empty list), print the first one
    #     print("Predicted Genre:", predicted_genre[0])
    # else:
    #     # Handle the case when there is no matching genre label
    #     print("No matching genre found for label:", num_pred)

# Calculate accuracy
#accuracy = accuracy_score(y_test, y_pred)

