import logging

import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split


def create_target(
    data: pd.DataFrame
) -> pd.DataFrame:
    """Cria target 'FLAG_MAU' para a base de modelagem.
    
    Args:
        data: Base de pagamentos com colunas 'DATA_PAGAMENTO' e 'DATA_VENCIMENTO'
    Returns:
        df_target: Base com target criado
    """

    df_target = (
        data
        .assign(
            DATA_PAGAMENTO = lambda df: pd.to_datetime(df['DATA_PAGAMENTO'], format='%Y-%m-%d'),
            DATA_VENCIMENTO = lambda df: pd.to_datetime(df['DATA_VENCIMENTO'], format='%Y-%m-%d'),
            ATRASO = lambda df: (df.DATA_PAGAMENTO - df.DATA_VENCIMENTO).dt.days,
            FLAG_MAU = lambda df: (df['ATRASO'] >= 5).astype(int)
        )
    )

    return df_target

# def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
#     """Splits data into features and targets training and test sets.

#     Args:
#         data: Data containing features and target.
#         parameters: Parameters defined in parameters/data_science.yml.
#     Returns:
#         Split data.
#     """
#     X = data[parameters["features"]]
#     y = data["price"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
#     )
#     return X_train, X_test, y_train, y_test


# def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
#     """Trains the linear regression model.

#     Args:
#         X_train: Training data of independent features.
#         y_train: Training data for price.

#     Returns:
#         Trained model.
#     """
#     regressor = LinearRegression()
#     regressor.fit(X_train, y_train)
#     return regressor


# def evaluate_model(
#     regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
# ):
#     """Calculates and logs the coefficient of determination.

#     Args:
#         regressor: Trained model.
#         X_test: Testing data of independent features.
#         y_test: Testing data for price.
#     """
#     y_pred = regressor.predict(X_test)
#     score = r2_score(y_test, y_pred)
#     logger = logging.getLogger(__name__)
#     logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
