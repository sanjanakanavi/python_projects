def load_csv():
    import pandas as pd
    import os

    # Define the path to the CSV file
    csv_file_path = os.path.join(os.path.dirname( ), 'username-password-recovery-code.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    return df