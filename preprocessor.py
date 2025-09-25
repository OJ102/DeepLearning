import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    df = pd.read_csv(file_path, encoding='latin1')
    print("Data loaded successfully.")
    return df


def _interval_mean(interval_str: str) -> float:
    """
    Convert an interval string (e.g., "(20000,30000]") into its mean value.

    Parameters
    ----------
    interval_str : str
        Interval in string format.

    Returns
    -------
    float
        Mean of the interval values.
    """
    interval_str = interval_str.strip('()[]')   # remove brackets
    left, right = interval_str.split(',')       # split into boundaries
    return (float(left) + float(right)) / 2     # compute midpoint


def _extract_state(geography: str) -> str:
    """
    Extract the U.S. state name from a geography string.

    Example: "CountyName, State" → "State"

    Parameters
    ----------
    geography : str
        Geography column value.

    Returns
    -------
    str
        Extracted state name.
    """
    return geography.split(',')[-1].strip()


def _encode_state_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the 'State' column and drop the original.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'State' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'State' replaced by one-hot encoded features.
    """
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids dummy trap
    state_encoded = encoder.fit_transform(df[['State']])

    # Create a DataFrame with new encoded columns
    state_encoded_df = pd.DataFrame(
        state_encoded,
        columns=encoder.get_feature_names_out(['State']),
        index=df.index
    )

    # Return combined DataFrame
    return pd.concat([df.drop(columns=['State']), state_encoded_df], axis=1)


def _scale_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation and standard scaling to numeric columns.

    Excludes target column ('TARGET_deathRate') and one-hot encoded 'State_' columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with numeric and categorical features.

    Returns
    -------
    pd.DataFrame
        DataFrame with scaled numeric features.
    """
    # Select numeric columns (exclude target + one-hot encoded categorical features)
    cols_to_scale = [
        col for col in df.columns
        if not col.startswith('State_') and col != 'TARGET_deathRate'
    ]

    # Apply log transformation to reduce skewness
    log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
    df[cols_to_scale] = log_transformer.fit_transform(df[cols_to_scale])

    # Standardize to mean=0, variance=1
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline:
      1. Drop irrelevant features
      2. Fill missing values
      3. Extract 'State' and convert income intervals to numeric
      4. One-hot encode categorical features
      5. Apply log transform + scaling to numeric features

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for modeling.
    """
    # Drop columns with little predictive power
    df.drop(columns=['PctSomeCol18_24'], inplace=True)

    # Fill missing values in specific columns with their mean
    for col in ['PctEmployed16_Over', 'PctPrivateCoverageAlone']:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)

    # Extract state from geography and compute average income from intervals
    df['State'] = df['Geography'].apply(_extract_state)
    df['avgInc'] = df['binnedInc'].apply(_interval_mean)

    # Drop original non-numeric or redundant columns
    df.drop(columns=['Geography', 'binnedInc'], inplace=True)

    # One-hot encode categorical column(s)
    df_encoded = _encode_state_column(df)

    # Scale numerical features
    df_scaled = _scale_numeric_columns(df_encoded)

    return df_scaled
