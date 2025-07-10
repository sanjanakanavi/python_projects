def load_csv():
    import pandas as pd
    import os

    try:
        # Construct path: up one folder from 'data', then into 'src'
        base_path = os.path.dirname(__file__)
        print("Base path for CSV loading:", base_path)  # DEBUG LINE
        csv_file_path = os.path.abspath(os.path.join(base_path, '..', 'src', 'username-password-recovery-code.csv'))

        print("Trying to load CSV from:", csv_file_path)  # DEBUG LINE

        df = pd.read_csv(csv_file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
