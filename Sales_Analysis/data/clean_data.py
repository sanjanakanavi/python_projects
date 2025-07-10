# data/clean_data.py

from load_data import load_csv  # ✅ import from sibling module

def clean_data():
    try:
        # Load the data using the imported function
        df = load_csv()
        if df is None:
            print("No data loaded.")
            return None

        # Clean the data
        cleaned_df = df.dropna()
        cleaned_df.reset_index(drop=True, inplace=True)
        return cleaned_df

    except Exception as e:
        print(f"Error cleaning data: {e}")
        return None

if __name__ == "__main__":
    cleaned_data = clean_data()
    if cleaned_data is not None:
        print("Data cleaned successfully.")
        print(cleaned_data.info())
    else:
        print("Data cleaning failed.")