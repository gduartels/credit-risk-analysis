from kedro.pipeline import Pipeline, node, pipeline

from .nodes import join_data, filter_pj_data, clean_data, create_features


def create_pipeline(**kwargs) -> Pipeline:
    pipe_template = pipeline(
        [
            node(
                func=join_data,
                inputs=["pagamentos", "cadastral", "info"],
                outputs="raw_data",
                name="join_data_node",
            ),
            node(
                func=filter_pj_data,
                inputs="raw_data",
                outputs="pj_data",
                name="filter_data_node",
            ),
            node(
                func=clean_data,
                inputs="pj_data",
                outputs="clean_data",
                name="clean_data_node",
            ),
            node(
                func=create_features,
                inputs="clean_data",
                outputs="features_data",
                name="create_features_node",
            )
        ]
    )

    modeling_pipe = pipeline(
        pipe_template,
        namespace="modeling",
    )
    scoring_pipe = pipeline(
        pipe_template,
        namespace="scoring"
    )

    return (
        modeling_pipe +
        scoring_pipe
    )
