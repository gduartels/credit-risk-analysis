import pandas as pd


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
