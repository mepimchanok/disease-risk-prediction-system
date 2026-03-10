import pandas as pd

# โหลด dataset
df = pd.read_csv("data/diabetes.csv")

# แสดงข้อมูล
print(df.head())

# แสดงขนาด dataset
print("\nShape:", df.shape)

# แสดงชื่อ columns
print("\nColumns:", df.columns)