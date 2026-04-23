import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from pathlib import Path
import pandas as pd
import os
from sklearn.model_selection import KFold


def save_performances_on_file(output_folder, prefix, output_file_name, true_labels, predictions):
    output_file = output_folder + prefix+ output_file_name.split('/')[1] + '.txt'

    directory = Path(output_folder)
    # Create the directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)

    conf_matrix = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    with open(output_file, "w") as f:
        # Save confusion matrix
        f.write("\nFinal Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix) + "\n")  # Convert to string for writing

        # Save classification report
        f.write("\nClassification Report:\n")
        f.write(report + "\n")

        print(f"Results saved to {output_file}")
    report = classification_report(true_labels, predictions, output_dict=True)
    return report["macro avg"]["f1-score"]



def handle_csv(csv_path, df, indexes):
    """
    If the CSV file exists, reads it into a DataFrame and returns it.
    Otherwise, saves the provided DataFrame to the given path.

    Args:
        csv_path (str): Path to the CSV file.
        df (pd.DataFrame): DataFrame to save if the file does not exist.

    Returns:
        pd.DataFrame: The DataFrame read from the file if it exists, otherwise the provided DataFrame.
    """
    if os.path.exists(csv_path):
        print(f"File '{csv_path}' exists. Reading into DataFrame.")
        selected_df = pd.read_csv(csv_path)
    else:
        print(f"File '{csv_path}' does not exist. Using provided DataFrame.")
        df = df.reindex(indexes)
        selected_df = df
    return selected_df

def save_predictions_on_file(output_folder, data_path, df, column_name, column_values, true_labels, label_column, indexes):
    """
    If the CSV file exists, reads it into a DataFrame.
    Otherwise, saves the provided DataFrame to the given path.
    In both cases, appends an additional column with the given column name and results.

    Args:
        csv_path (str): Path to the CSV file.
        df (pd.DataFrame): DataFrame to save if the file does not exist.
        column_name (str): Name of the additional column.
        column_values (list or pd.Series): Values to add in the new column (must match DataFrame length).
        true_labels (or pd.Series): Used to verify the integrity of the provided df

    Returns:
        pd.DataFrame: The updated DataFrame with the new column.
    """
    data_stem = Path(data_path).stem
    csv_path = os.path.join(output_folder, f'models_prediction_{data_stem}.csv')
    
    df = handle_csv(csv_path, df, indexes)

    # Ensure the new column has the same length as the DataFrame
    if len(column_values) != len(df):
        raise ValueError("Length of column_values must match the number of rows in the DataFrame.")

    # Check for NaN values in the label column and handle them
    if df[label_column].isna().any():
        raise ValueError(f"The label column '{label_column}' contains NaN values. Please check the data preprocessing step.")
    
    if not df[label_column].astype(int).tolist() == list(true_labels):
        raise ValueError("Inconsistency among the provided labels and the ones in the dataset. Verify possible shuffle in the data.")

    # Add the new column
    df[column_name] = column_values

    # Save the updated DataFrame back to CSV
    df.to_csv(csv_path, index=False)

    return df

