from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import scoring


def create_pipeline(**kwargs) -> Pipeline:
    pipe_template = pipeline(
        [
            node(
                func=scoring,
                inputs=["model",
                        "features_data",
                        "params:scoring.id_columns",
                        "params:modeling.features_selected"],
                outputs="output",
                name="scoring_node"
            )
        ]
    )

    return pipeline(
        pipe_template,
        namespace="scoring",
        parameters={
            "params:scoring.id_columns",
            "params:modeling.features_selected"
        }
    )
