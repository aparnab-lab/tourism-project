import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Performing all the data manipulation inside the script
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)

# Changing  'Fe Male' into 'Female' in Gender Column
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# Changing  Unmarried to Single in Marital Status
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# Convert 'NumberOfChildrenVisiting', 'NumberOfTrips', and 'NumberOfFollowups' to integer types
df['NumberOfChildrenVisiting'] = df['NumberOfChildrenVisiting'].astype(int)
df['NumberOfTrips'] = df['NumberOfTrips'].astype(int)
df['NumberOfFollowups'] = df['NumberOfFollowups'].astype(int)

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
