import pandas as pd
data = {
    'Week': [1, 1, 2, 2, 3, 3],
    'Data_Usage': [500, 450, 520, 480, 550, 530]
}
df = pd.DataFrame(data)
mean_usage = df.groupby('Week')['Data_Usage'].mean()
print(mean_usage)