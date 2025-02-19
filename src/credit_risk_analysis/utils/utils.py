import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
import shap

def separa_colunas(df):
    numbers = df.select_dtypes(include=["number"]).columns.tolist()
    texts = df.select_dtypes(exclude=["number"]).columns.tolist()
    return numbers, texts

def check_2DIG(x:str):
  if x and re.match(r"^[0-9]{2}$", x):
    return x
  else:
    return None
  
def ks_score(y_true, y_pred):
    """_summary_

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    df2 = pd.DataFrame({"bad": y_true, "score": y_pred})

    df2["good"] = 1 - df2.bad
    df2["bucket"] = pd.qcut(df2["score"], 10, duplicates="drop")

    grouped = df2.groupby("bucket")
    agg1 = (
        pd.DataFrame(grouped.min()["score"])
        .rename(columns={"score": "min_scr"})
        .copy()
    )
    agg1["max_scr"] = grouped.max()["score"]
    agg1["bads"] = grouped.sum().bad
    agg1["goods"] = grouped.sum().good
    agg1["total"] = agg1.bads + agg1.goods

    agg2 = (agg1.sort_values(by="min_scr")).reset_index(drop=True)
    agg2["odds"] = (agg2.goods / agg2.bads).apply("{0:.2f}".format)
    agg2["bad_rate"] = (agg2.bads / agg2.total).apply("{0:.2%}".format)
    agg2["ks"] = (
        np.round(
            (
                (agg2.bads / df2.bad.sum()).cumsum()
                - (agg2.goods / df2.good.sum()).cumsum()  # noqa: W503
            ),
            4,
        )  # noqa: W503
        * 100  # noqa: W503
    )

    flag = lambda x: "<----" if x == agg2.ks.max() else ""  # noqa: E731
    agg2["max_ks"] = agg2.ks.apply(flag)
    return max(abs(agg2.ks))

def get_metrics(
    model,
    X_test,
    y_test,
    to_print: bool = False,
    output_dict: bool = False,
):
    """
    Calculate the AUC, classification report and KS-statistics for a model.

    Parameters:
        - model: Classifier with predict and predict_proba methods
        - X_test (pandas Dataframe): Dataframe ready for prediction
        - y_test (pandas Series): Series with true target values for the
          samples in X_test
        - to_print (bool): Whether to print or not the results

    Returns:
        - ks_test: the value of the KS statistic
        - auc_test: the value of the AUC
    """
    X_test = X_test.copy()
    _, texts = separa_colunas(X_test)
    X_test[texts] = X_test[texts].fillna("NA")

    y_prob = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_prob)
    ks_test = ks_score(y_test, y_prob)

    cr = classification_report(
        y_true=y_test, y_pred=model.predict(X_test), output_dict=output_dict
    )
    if to_print:
        print(cr)
        print("AUC Test", auc_test)
        print("KS Test", ks_test)
    if output_dict:
        return auc_test, ks_test, cr
    else:
        return auc_test, ks_test


def get_shap_importances(model, X_test):
    """
    Calculate shap values and importances for a model.

    Parameters:
        - model: Classifier compatible with SHAP
        - X_test (pandas Dataframe): Dataframe ready for prediction

    Returns:
        - pandas Dataframe: Dataframe with the variables importances
    """
    explainer = shap.TreeExplainer(model)
    try:
        shap_values = explainer.shap_values(X_test)[0]
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(X_test.columns, vals)),
            columns=["col_name", "feature_importance_vals"],
        )
    except:  # noqa: E722
        shap_values = explainer.shap_values(X_test)
        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(X_test.columns, vals)),
            columns=["col_name", "feature_importance_vals"],
        )

    return feature_importance.sort_values(
        by=["feature_importance_vals"], ascending=False, inplace=False
    )

def get_shap_summary_plot(
    model,
    X_test,
    **kwargs
):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    shap.summary_plot(
        shap_values[::20], X_test[::20], plot_size=None, show=False
    )
    plt.title(f"Shap Values Summary Plot")
    return fig

def evaluate_model(model,model_features,df,target,to_plot=True):
    df=df.copy()
    probs = model.predict_proba(df[model_features])[:,1]
    df['SCORE'] = (1-probs)*1000

    get_metrics(model,df[model_features],df[target],to_print=True)
    if to_plot:
        get_shap_summary_plot(model,df[model_features])
        plt.show()
        plt.figure()
        df['RATING'] = pd.cut(df['SCORE'],bins=np.linspace(0,1000,11))
        df['RATING'].value_counts().sort_index().plot.bar()
        plt.ylabel('Frequência')
        plt.show()
        plt.figure()
        df.groupby('RATING')[target].mean().sort_index().plot.bar()
        plt.ylabel('Inadimplência')
        plt.show()

def plot_shap_dependence(model, df, model_features, target):
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(df[model_features+[target]])
  for var in model_features:
    plt.figure(figsize=(16,10))
    if df[var].dtype in (int,float):
      shap.dependence_plot(ind = var,shap_values = shap_values, features = df[model_features+[target]],interaction_index=target,xmax=df[var].quantile(0.99))
    else:
      shap.dependence_plot(ind = var,shap_values = shap_values, features = df[model_features+[target]],interaction_index=target)
    plt.show()
