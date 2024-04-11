import pandas as pd



def process_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna()
    print(df.columns)
    print(df)
    return df


