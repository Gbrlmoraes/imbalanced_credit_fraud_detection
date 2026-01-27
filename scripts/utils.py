import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def log_model_results(y_true, y_hat):
    """Calculate, print, and log metrics and artifacts to MLflow."""

    recall = recall_score(y_true, y_hat)
    f1 = f1_score(y_true, y_hat)
    precision = precision_score(y_true, y_hat, zero_division=1)
    accuracy = accuracy_score(y_true, y_hat)

    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

    metrics_dict = {
        'test_recall': recall,
        'test_f1_score': f1,
        'test_precision': precision,
        'test_accuracy': accuracy,
    }
    mlflow.log_metrics(metrics_dict)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Fraud (P)', 'Fraud (P)'],
        yticklabels=['No Fraud (T)', 'Fraud (T)'],
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.plot()

    confusion_matrix_path = 'confusion_matrix.png'
    plt.savefig(confusion_matrix_path)
    plt.close()

    mlflow.log_artifact(confusion_matrix_path, 'plots')


def evaluate_distribution(
    dataframe: pd.DataFrame,
    column: str,
    show_plot: bool = True,
    ax=None,
) -> dict:
    """
    Plots the distribution of a specified column in the dataframe,
    separated by the 'Class' column. It also calculates and displays
    statistical metrics such as mean, median, KS statistic, KS p-value,
    and AUC-ROC.
    Args:
        dataframe (pd.DataFrame): The input dataframe containing the data.
        column (str): The column name for which the distribution is to be plotted.
        show_plot (bool): If True, plots the distribution.
        output_metrics (bool): If True, returns a dictionary of calculated metrics.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new
            figure is created.
    Returns:
        dict: A dictionary containing the calculated metrics if output_metrics is True.
    """
    mean_val = dataframe[column].mean()
    median_val = dataframe[column].median()

    ks_stat, ks_pval = ks_2samp(
        dataframe.loc[dataframe['Class'] == 0, column],
        dataframe.loc[dataframe['Class'] == 1, column],
    )

    auc_roc = roc_auc_score(dataframe['Class'], dataframe[column])

    if show_plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
            should_show = True
        else:
            should_show = False

        q3, q1 = np.percentile(dataframe[column], [75, 25])

        ax.set_title(
            f'Distribution of {column} by Class'
            f'\nMean: {mean_val:.3f}, Median: {median_val:.3f}'
            f'\nKS stat: {ks_stat:.3f}, KS p-value: {ks_pval:.3f}'
            f'\nAUC-ROC: {auc_roc:.3f}'
            ''
        )

        sns.histplot(
            data=dataframe,
            x=column,
            hue='Class',
            bins=150,
            kde=True,
            stat='density',
            common_norm=False,
            alpha=0.6,
            ax=ax,
        )

        ax.axvline(mean_val, color='k', linestyle='--', alpha=0.5, label='Mean')
        ax.axvline(median_val, color='r', linestyle=':', alpha=0.5, label='Median')
        ax.axvline(
            q1, color='b', linestyle='-.', alpha=0.5, label='Q1 (25th percentile)'
        )
        ax.axvline(
            q3, color='g', linestyle='-.', alpha=0.5, label='Q3 (75th percentile)'
        )
        ax.legend()

        ax.set_xlabel('')
        ax.set_ylabel('Density')

        if should_show:
            plt.tight_layout()
            plt.show()

    return {
        'feature': column,
        'mean': mean_val,
        'median': median_val,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'auc_roc': auc_roc,
    }


def print_dataframe_info(dataframe: pd.DataFrame, name: str) -> None:
    """
    Prints the total number of rows in the dataframe,
    as well as the number of each class.
    """
    total_rows = dataframe.shape[0]
    rows_class_0 = dataframe[dataframe['Class'] == 0].shape[0]
    rows_class_1 = dataframe[dataframe['Class'] == 1].shape[0]
    print(
        f'The {name} dataframe has {total_rows} rows'
        f'\n- Rows with class 0: {rows_class_0 / total_rows:.2%} ({rows_class_0})'
        f'\n- Rows with class 1: {rows_class_1 / total_rows:.2%} ({rows_class_1})'
    )
