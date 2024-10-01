import pandas as pd

df = pd.read_csv("Data/Data.csv")
a = df.loc[[2], :]
