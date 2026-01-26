import pandas as pd

df = pd.read_csv('datasets/MELD/train.csv')

emotion_counts = df['Emotion'].value_counts()
print(emotion_counts)