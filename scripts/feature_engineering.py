def create_new_features(df):
    """
    Create new features based on existing data.
    """
    df['loan_to_income_ratio'] = df['loan_amount'] / df['income']
    df['age_at_loan'] = 2025 - df['birth_year']  # Example: Calculate age at loan application
    return df

def select_important_features(df, target_column):
    """
    Select important features using correlation or feature importance scores.
    """
    correlation = df.corr()[target_column].abs()
    selected_features = correlation[correlation > 0.1].index.tolist()
    return df[selected_features]
