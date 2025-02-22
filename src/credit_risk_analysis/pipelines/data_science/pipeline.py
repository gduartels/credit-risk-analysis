from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_target, split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_target,
                inputs="features_data",
                outputs="spine",
                name="create_target_node"
            ), 
            node(
                func=split_data,
                inputs=["spine", 
                        "params:modeling.split_column", "params:modeling.split_cohort",
                        "params:modeling.features_selected", "params:modeling.target_name",
                        "params:modeling.random_state_split"],
                outputs=["X_train", "X_test", "X_val_oot",
                         "y_train", "y_test", "y_val_oot"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train",
                        "params:modeling.hyperparameters", "params:modeling.categorical_features",
                        "params:modeling.random_state_model"],
                outputs="model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "params:modeling.features_selected", 
                        "X_train", "X_test", "X_val_oot",
                        "y_train", "y_test", "y_val_oot"],
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
        ]
    )
