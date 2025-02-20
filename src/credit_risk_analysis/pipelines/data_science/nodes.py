import logging

import pandas as pd
from typing import List, Dict
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier


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

def split_data(data: pd.DataFrame, 
               split_column: str, split_cohort: str, 
               features: List, target: str,
               random_state: int = 42) -> tuple:
    """Separa os dados em features e targets e conjuntos de treino e validação.
    
    Args:
        data: Data containing features and target.
        split_cohort: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    df_dev = data[data[split_column] < split_cohort].copy()
    df_oot = data[data[split_column] >= split_cohort].copy()

    df_treino, df_teste = train_test_split(df_dev, test_size=0.3, random_state=random_state)
    
    X_train, y_train = df_treino[features], df_treino[target]
    X_test, y_test = df_teste[features], df_teste[target]
    X_val_oot, y_val_oot = df_oot[features], df_oot[target]

    return X_train, X_test, X_val_oot, y_train, y_test, y_val_oot


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                parameters: Dict, cat_features: List, random_state: int) -> CatBoostClassifier:
    """Treina o modelo catboost.

    Args:
        X_train: Dados de treinamento de pagamento.
        y_train: Target do modelo 'FLAG_MAU', que representa um atraso acima de 5 dias.

    Returns:
        Trained model.
    """
    model_final = CatBoostClassifier(**parameters,
                                     cat_features = cat_features, 
                                     verbose=False, 
                                     random_state = random_state)
    model_final.fit(X_train, y_train)
    return model_final


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
