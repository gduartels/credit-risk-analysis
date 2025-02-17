import pandas as pd
import numpy as np
import re

def _check_2DIG(x:str):
    if x and re.match(r"^[0-9]{2}$", x):
        return x
    else:
     return None

def join_data(
    base_pagamentos: pd.DataFrame,
    base_cadastral: pd.DataFrame,
    base_info: pd.DataFrame
) -> pd.DataFrame:
    """Junta as bases de dados em uma única tabela.

    Args:
        base_pagamentos: Base bruta com informações sobre transações (empréstimos) passados
        base_cadastral: Base contendo informações cadastrais dos clientes
        base_info: Base com informações adicionais dos clientes
    Returns:
        df_completo: Tabela unindo todas as informações separadas
    """

    df_completo = (
        base_pagamentos
        .merge(base_cadastral, on=['ID_CLIENTE'], how='left')
        .merge(base_info, on=['ID_CLIENTE', 'SAFRA_REF'], how='left')
    )

    return df_completo

def filter_pj_data(
    raw_data: pd.DataFrame
) -> pd.DataFrame:
    """Mantém apenas os registros de clientes PJ da base.

    Args:
        raw_data: Tabela unindo todas as informações separadas
    Returns:
        df_pj: Tabela contendo apenas documentos de clientes PJ
    """
    df_pj = (
        raw_data
        .query('FLAG_PF.isna()')
        .reset_index(drop=True)
    )

    return df_pj

def clean_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Realiza os tratamentos para limpeza da base.

    Args:
        df: Tabela contendo as informações de pagamento
    Returns:
        df_cleaned: Tabela limpa 
    """

    df_cleaned = (
        df
        .assign(
            DDD = lambda df: df['DDD'].astype(str).apply(_check_2DIG).fillna('NA'),
            SEGMENTO_INDUSTRIAL = lambda df: df['SEGMENTO_INDUSTRIAL'].map(
                {'Serviços':'SERVICOS', 'Comércio':'COMERCIO', 'Indústria':'INDUSTRIA'}).fillna('NA'),
            DOMINIO_EMAIL = lambda df: df['DOMINIO_EMAIL'].map(
                {'YAHOO':'YAHOO', 'HOTMAIL':'HOTMAIL', 'OUTLOOK':'OUTLOOK', 
                 'GMAIL':'GMAIL','BOL':'BOL','AOL':'AOL'}).fillna('NA'),
            PORTE = lambda df: df['DOMINIO_EMAIL'].map(
                {'PEQUENO':'PEQUENO', 'MEDIO':'MEDIO', 'GRANDE':'GRANDE'}).fillna('NA'),
            CEP_2_DIG = lambda df: df['CEP_2_DIG'].astype(str).apply(_check_2DIG).fillna('NA')
        )
    )

    return df_cleaned

def create_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Cria novas variáveis para o modelo.

    Args:
        df: Tabela contendo as informações de pagamento
    Returns:
        df_features: Tabela com features criadas
    """

    df_features = (
        df
        .assign(
            RZ_RENDA_FUNC = lambda df: df[['RENDA_MES_ANTERIOR','NO_FUNCIONARIOS']].apply(
                lambda x: x[0]/x[1] if x[1]> 0 else np.nan, axis=1
            ),
            VL_TAXA = lambda df: df[['TAXA','VALOR_A_PAGAR']].apply(
                lambda x: (x[0]/100)*x[1] if x[1]> 0 else np.nan, axis=1
            )
        )
    )

    return df_features
