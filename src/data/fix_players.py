import pandas as pd
import math
from pathlib import Path
from src.utils.db import get_database

def main():
    db = get_database()
    df = pd.read_csv("data/raw/historical_1930_2018/players.csv")
    
    # Convertir birth_date en string pour éviter le problème NaT
    if 'birth_date' in df.columns:
        df['birth_date'] = df['birth_date'].astype(str).replace('NaT', None).replace('nan', None)

    def clean(record):
        return {k: (None if (isinstance(v, float) and math.isnan(v)) or v == 'NaT' or v == 'nan' else v)
                for k, v in record.items()}

    records = [clean(r) for r in df.to_dict(orient='records')]
    
    db['players'].drop()
    db['players'].insert_many(records)
    print(f"✅ players: {len(records)} documents importés")

if __name__ == "__main__":
    main()
