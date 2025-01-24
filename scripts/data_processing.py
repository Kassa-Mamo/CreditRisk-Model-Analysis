import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths for the dataset
DATA_PATH = r"C:\Users\user\Desktop\10 Academy- Machine-Learning\10 Academy W6\data\data.csv"

def load_data():
    """
    Load data from the specified path.
    """
    return pd.read_csv(DATA_PATH)

def preprocess_data(df):
    """
    Preprocess data:
    - Handle missing values
    - Encode categorical variables
    - Normalize numerical features
    """
    # Fill missing values
    df.fillna(df.median(), inplace=True)

    # Encode categorical columns
    label_encoders = {}
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, label_encoders

def split_data(df, target_column):
    """
    Split data into train and test sets.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
