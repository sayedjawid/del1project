import pandas as pd


def load_data(path:str) -> pd.DataFrame:
    """Läser in CSV-fil och retrnerar en pandas dataframe"""
    df = pd.read_csv(path)
    return df 