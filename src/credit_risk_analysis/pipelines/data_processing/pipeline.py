from kedro.pipeline import Pipeline, node, pipeline

from .nodes import join_data, filter_pj_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=join_data,
                inputs=["pagamentos_desenvolvimento", "cadastral", "info"],
                outputs="raw_data",
                name="join_data_node",
            ),
            node(
                func=filter_pj_data,
                inputs="raw_data",
                outputs="pj_data",
                name="filter_data_node",
            ),
        ]
    )
