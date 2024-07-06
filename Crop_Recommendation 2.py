from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

crop_data = "filtered_data.csv"
dataf = pd.read_csv(crop_data)
crop=dataf["label"].iloc[0]
print("Crop:\t",crop)

columns = ['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']

max_values = dataf[columns].max(axis=0).round()
print("Maximum Values:")
print(max_values)

max_values['label']=crop
min_values = dataf[columns].min(axis=0).round()
print("Minimum Values:\n")
print(min_values)
min_values['label']=crop
max_values = max_values.to_frame().T
file_path1 = "D:/VIT/Courses in Vit/Fall Inter Semester 2023-2024/BCSE308L-Computer Networks/Project/maximum values.csv"
max_values.to_csv(file_path1, index=False)

print("Data saved to", file_path1)
file_path2 = "D:/VIT/Courses in Vit/Fall Inter Semester 2023-2024/BCSE308L-Computer Networks/Project/minimum values.csv"
min_values = min_values.to_frame().T
min_values.to_csv(file_path2, index=False)

print("Data saved to", file_path2)

