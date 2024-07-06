from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


crop_data = "Crop_recommendation.csv"
dataf = pd.read_csv(crop_data)



columns = ['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']

max_values = dataf[columns].max(axis=0)


min_values = dataf[columns].min(axis=0)

X = dataf[columns]
y = dataf["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


classifier = svm.SVC(kernel='poly')


classifier.fit(X_train, y_train)


predictions = classifier.predict(X_test)


accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy * 100)
import random

random_n = random.uniform(min_values[0],max_values[0]+1)
print("Nitrogen Value: \t",random_n)
random_p = random.uniform(min_values[1],max_values[1]+1)
print("Phosphorous Value: \t",random_p)
random_k = random.uniform(min_values[2],max_values[2]+1)
print("Potassium Value: \t",random_k)
random_temp = random.uniform(min_values[3],max_values[3]+1)
print("Temperature Value: \t",random_temp)
random_hum = random.uniform(min_values[4],max_values[4]+1)
print("Humidity Value: \t",random_hum)
random_ph = random.uniform(min_values[5],max_values[5]+1)
print("pH Value: \t\t",random_ph)
random_rain = random.uniform(min_values[6],max_values[6]+1)
print("Rainfall Value: \t",random_rain)
dataex = np.array([[random_n, random_p, random_k, random_temp, random_hum, random_ph, random_rain]])
testprediction = classifier.predict(dataex)
print("Predicted Crop: \t",testprediction[0])
predicted_label = str(testprediction[0])
filtered_data = dataf[dataf["label"].astype(str) == predicted_label]
fil_data = dataf[dataf["label"].astype(str) == predicted_label]
filtered_data.to_csv("filtered_data.csv", index=False)
import pandas as pd

filtered_data = pd.read_csv("filtered_data.csv")


filtered_data.dropna(axis=1, how='all', inplace=True)


filtered_data.to_csv("filtered_data.csv", index=False)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mlxtend.plotting import plot_decision_regions

image_file_path = f"food pics/{predicted_label}.JPG"
try:
    img = plt.imread(image_file_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted Crop: {predicted_label}")
    plt.show()
except FileNotFoundError:
    print(f"Image for crop '{predicted_label}' not found.")

avg_values = fil_data.select_dtypes(include=np.number).mean()

random_values = [random_n, random_p, random_k, random_temp, random_hum, random_ph, random_rain]
features = ['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
min_values = filtered_data[features].min(axis=0).round()
max_values = filtered_data[features].max(axis=0).round()
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(features))
bars = plt.bar(index, max_values[features], color='yellow', alpha=0.6, label='Maximum and Minimum recommended Values', bottom = min_values)
bars = plt.bar(index, min_values[features], color='white', alpha=0.6)

plt.xlabel('Features')
plt.ylabel('Values')
plt.title(f'Statistics for Predicted Crop: {predicted_label}')
plt.xticks(index, features, rotation=45)

for i, val in enumerate(random_values):
    plt.plot([i - bar_width, i + bar_width], [val, val], linestyle='--', color='red', linewidth=2.0)

legend_elements = [Line2D([0], [0], linestyle='--', color='red', label='Current Value')]

handles, labels = plt.gca().get_legend_handles_labels()
handles += legend_elements
labels += ['Current Value']
plt.legend(handles=handles, labels=labels, loc='upper left')
plt.tight_layout()
plt.show()
