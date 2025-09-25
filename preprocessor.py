import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    df = pd.read_csv(file_path, encoding='latin1')
    print("Data loaded successfully.")
    return df

def _interval_mean(interval_str: str) -> float:
    """Convert a string interval to its mean value."""
    interval_str = interval_str.strip('()[]')
    left, right = interval_str.split(',')
    return (float(left) + float(right)) / 2

def _extract_state(geography: str) -> str:
    """Extract the state from the Geography column."""
    return geography.split(',')[-1].strip()

def _encode_state_column(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the 'State' column."""
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    state_encoded = encoder.fit_transform(df[['State']])
    state_encoded_df = pd.DataFrame(
        state_encoded,
        columns=encoder.get_feature_names_out(['State']),
        index=df.index
    )
    return pd.concat([df.drop(columns=['State']), state_encoded_df], axis=1)

def _scale_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric columns (excluding one-hot encoded columns)."""
    cols_to_scale = [col for col in df.columns if not col.startswith('State_') and not col in ['TARGET_deathRate']]
    log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
    df[cols_to_scale] = log_transformer.fit_transform(df[cols_to_scale])
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input DataFrame."""
    df.drop(columns=['PctSomeCol18_24'], inplace=True)
    # Fill missing values with column mean
    for col in ['PctEmployed16_Over', 'PctPrivateCoverageAlone']:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)

    # Extract state and average income
    df['State'] = df['Geography'].apply(_extract_state)
    df['avgInc'] = df['binnedInc'].apply(_interval_mean)

    # Drop original columns
    df.drop(columns=['Geography', 'binnedInc'], inplace=True)

    # One-hot encode and scale
    df_encoded = _encode_state_column(df)
    df_scaled = _scale_numeric_columns(df_encoded)

    return df_scaled