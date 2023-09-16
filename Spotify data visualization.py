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

y_pred = model.predict(x_test)

print(y_pred)

# Assuming you have a predicted genre label as an integer
predicted_label = 2  # Replace with the actual integer label you have

# Map the integer label to the corresponding genre string
predicted_genre = next((genre for genre, label in genre_to_label.items() if label == predicted_label), None)

#prints as a String
print("Predicted Genre:", predicted_genre)

# accuracy = accuracy_score(y_test, y_pred)

# print (initial_dataframe)
# print (initial_dataframe.columns)


#data visualization

# scikit-learn
# regression for song popularity 
# classification for genre