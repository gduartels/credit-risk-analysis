import logging

import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from credit_risk_analysis.utils.utils import ks_score


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
        Modelo treinado.
    """
    model_final = CatBoostClassifier(**parameters,
                                     cat_features = cat_features, 
                                     verbose=False, 
                                     random_state = random_state)
    model_final.fit(X_train, y_train)
    return model_final


def _return_metrics(
    df: pd.DataFrame
) -> Tuple:
    auc = roc_auc_score(df['TARGET'], df['PROB'])
    ks = ks_score(df['TARGET'], df['PROB'])
    acc = accuracy_score(df['TARGET'], df['PRED'])
    pr = precision_score(df['TARGET'], df['PRED'])
    rec = recall_score(df['TARGET'], df['PRED'])
    f1 = f1_score(df['TARGET'], df['PRED'])
    return auc, ks/100, acc, pr, rec, f1
    


def evaluate_model(
    model: CatBoostClassifier, 
    features: List, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    X_val_oot: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    y_val_oot: pd.Series
):
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """

    df_train = X_train.reset_index(drop=True).assign(TARGET = y_train.copy())
    df_test = X_test.reset_index(drop=True).assign(TARGET = y_test.copy())
    df_val_oot = X_val_oot.reset_index(drop=True).assign(TARGET = y_val_oot.copy())

    df_train = df_train.assign(
        PRED = lambda df: model.predict(df[features]),
        PROB = lambda df: model.predict_proba(df[features])[:, 1]
    )
    df_test = df_test.assign(
        PRED = lambda df: model.predict(df[features]),
        PROB = lambda df: model.predict_proba(df[features])[:, 1]
    )
    df_val_oot = df_val_oot.assign(
        PRED = lambda df: model.predict(df[features]),
        PROB = lambda df: model.predict_proba(df[features])[:, 1]
    )
    auc_train, ks_train, acc_train, pr_train, rec_train, f1_train = _return_metrics(df_train)
    auc_test, ks_test, acc_test, pr_test, rec_test, f1_test = _return_metrics(df_test)
    auc_val, ks_val, acc_val, pr_val, rec_val, f1_val = _return_metrics(df_val_oot)
    

    df_metrics = pd.DataFrame({
        "AUC": [auc_train, auc_test, auc_val],
        "KS": [ks_train, ks_test, ks_val],
        "Accuracy": [acc_train, acc_test, acc_val],
        "Precision": [pr_train, pr_test, pr_val],
        "Recall": [rec_train, rec_test, rec_val],
        "F1": [f1_train, f1_test, f1_val]
    })
    df_metrics.index = ['Treino', 'Teste', 'Validação Out-of-Time']

    return df_metrics