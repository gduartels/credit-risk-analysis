from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from typing import List


def scoring(
    model: CatBoostClassifier,
    inference_data: pd.DataFrame,
    id_columns: List[str],
    features: List[str]
) -> pd.DataFrame:
    """
    Realiza a escoragem de dados de entrada utilizando um modelo treinado.

    Esta função aplica um modelo de classificação previamente treinado sobre um conjunto de dados de entrada
    e retorna um DataFrame contendo as previsões de probabilidade de inadimplência, pontuação e classificação,
    além de identificar as colunas de ID.

    Parameters
    ----------
    trained_model : CatBoostClassifier
        O modelo de classificação treinado, que deve ser uma instância da classe CatBoostClassifier.
        
    inference_data : pd.DataFrame
        Um DataFrame contendo as features de entrada que serão utilizadas para a escoragem. 
        Deve conter todas as colunas necessárias que o modelo espera.
        
    id_columns : List[str]
        Uma lista de strings que representa os nomes das colunas de identificação que devem ser mantidas no resultado.

    features: List[str]
        Uma lista de strings contendo as features do modelo.

    Returns
    -------
    pd.DataFrame
        Um DataFrame contendo as colunas de ID, as features de entrada, as probabilidades de inadimplência,
        a pontuação calculada e a classificação, além da data da escoragem.
    """

    X = inference_data.filter(features)
    df_id = inference_data[id_columns]

    result = (
        pd.concat([df_id, X], axis=1)
        .assign(PROBABILIDADE_INADIMPLENCIA=model.predict_proba(X)[:, 1],
                SCORE=lambda df: (1-df.PROBABILIDADE_INADIMPLENCIA)*1000,
                RATING=lambda df: pd.cut(df['SCORE'], bins=np.linspace(0,1000,11)).astype(str),
                DT_ESCORAGEM=pd.Timestamp.now().strftime("%Y-%m-%d"))
    )

    return result
