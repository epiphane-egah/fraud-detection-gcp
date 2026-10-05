import copy 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import roc_curve, auc
import miceforest as mf


def pearson_skewness(data:pd.Series):
    p_sk = 3*(data.mean() - data.median())/data.std()
    return p_sk


def density_estimator(x_series:pd.Series)->None:
    d_estimator = scipy.stats.gaussian_kde(x_series)
    d_estimator.set_bandwidth(0.1)
    x = d_estimator.evaluate(np.linspace(0, 140, 40))
    plt.plot(np.linspace(0, 140, 40), x, label="kde estimated PDF", color="r")


def plot_categorical(df):
    # Sélection des colonnes catégorielles uniquement
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    n_cols = 2  # nombre de colonnes dans la grille
    n_rows = int(np.ceil(len(cat_cols) / n_cols))
    
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(cat_cols):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        sns.countplot(
            y=col,
            hue=col,
            data=df,
            ax=ax,
            palette="Dark2",
            order=df[col].value_counts().index
        )
        
        ax.set_title(col)
        
        total = df[col].dropna().shape[0]
        
        for p in ax.patches:
            percentage = f'{100 * p.get_width() / total:.1f}%'
            x = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax.annotate(
                percentage,
                (x, y),
                xytext=(5, 0),
                textcoords='offset points',
                ha='left',
                va='center'
            )
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
    
    plt.tight_layout()
    plt.show()


def skew_data(X_pdf):
    for col in X_pdf.select_dtypes(include=["int", "float"]).columns.tolist():
        X_pdf[col] = scipy.stats.boxcox(X_pdf[col]+1)[0]
    return X_pdf


def borne_inf_sup(x: pd.Series)->pd.Series:
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    born_inf = q1 - 1.5*(q3-q1)
    born_sup = q1 + 1.5*(q3-q1)
    conditions = (x > born_sup) | (x < born_inf)
    return conditions


def carre_rapport_de_correlation(df, groupe, col):
    inertie_inter_groupe = (((df[col].mean() - df.groupby(groupe)[col].mean())**2)*(df.groupby(groupe)[col].count())).sum()
    intertie_intra_groupe = (df.groupby(groupe)[col].count()*df.groupby(groupe)[col].var(ddof=0)).sum()
    inertie_total = inertie_inter_groupe + intertie_intra_groupe
    return np.round(inertie_inter_groupe / inertie_total, 2)


def eda(df):
    plt.rcParams["figure.figsize"] = (12, 10)

    plt.subplot(221)

    df.groupby("isFraud")["amount"].median().plot(kind="bar")
    texte = f"Les transactions frauduleuses sont {df.groupby('isFraud')['amount'].mean()[1] / (df.groupby('isFraud')['amount'].mean()[0]): .2f} plus élevées en moyenne"
    texte = texte + f"\n Mais en volume elles ne représentent que {df.groupby('isFraud')['amount'].sum()[1]*100 / (df.groupby('isFraud')['amount'].sum()[0] + df.groupby('isFraud')['amount'].sum()[1]):.2f} % des transactions"
    plt.figtext(0.5, 1, texte,
         fontsize=12,
         color="red",
         ha="center",   # alignement horizontal
         va="bottom",   # alignement vertical
         rotation=0)   # rotation en degrés
    plt.title("Montant médian des transactions vs isFraud ")

    plt.subplot(222)
    # Group and calculate mean
    grouped = df.groupby(["isFraud", "type"])["amount"].mean()

    # Convert to wide format (types become columns)
    pivot_df = grouped.unstack(fill_value=0)

    bottom = None
    for column in pivot_df.columns:
        if bottom is None:
            plt.bar(pivot_df.index.astype(str), pivot_df[column])
            bottom = pivot_df[column]
        else:
            plt.bar(pivot_df.index.astype(str), pivot_df[column], bottom=bottom)
            bottom = bottom + pivot_df[column]

    plt.xlabel("isFraud")
    plt.ylabel("Mean Amount")
    plt.title("Mean Transaction Amount by isFraud and Type")
    plt.xticks(rotation=0)
    plt.legend(pivot_df.columns, title="Transaction Type")

    ax = plt.subplot(223)
    texte = f"Rapport de corrélation est de : {carre_rapport_de_correlation(df, 'type', 'amount'):.2f}"
    ax.text(0.5, 2, texte,
         fontsize=12,
         color="red",
         ha="center",   # alignement horizontal
         va="bottom",   # alignement vertical
         rotation=0)   # rotation en degrés
    fig = df.groupby("type")["amount"].sum().plot(kind="barh", ax=ax)

    ax = plt.subplot(224)
    fig = df.groupby("type")["amount"].median().plot(kind="barh", ax=ax)

    plt.show()


def plot_roc_curve(y_test, y_scores):

    # 2. ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # 3. Interactive ROC with Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines+markers",
            name=f"ROC (AUC = {roc_auc:.2f})",
            # custom data to show threshold (probabilité) on hover
            customdata=np.stack([thresholds], axis=-1),
            hovertemplate=(
                "FPR: %{x:.3f}<br>"
                "TPR: %{y:.3f}<br>"
                "Threshold (proba): %{customdata[0]:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="ROC Curve (interactive)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=600,
        height=400,
    )

    fig.show(renderer="browser")


def preprocess(train_pdf: pd.DataFrame) -> pd.DataFrame:
    
    # type casting
    train_pdf["type"] = train_pdf["type"].astype("category")

    # traitement outlier
    for col in train_pdf.select_dtypes(
     exclude=["category", "object", "str"]).columns.tolist():
        train_pdf.loc[borne_inf_sup(train_pdf[col])==True, col] = np.nan

    # traitement des valeurs manquantes
    original_index = train_pdf.index

    train_pdf = train_pdf.reset_index(drop=True)
    kernel = mf.ImputationKernel(
        train_pdf, num_datasets=1, save_all_iterations_data=True, random_state=42
    )

    kernel.mice(iterations=3)

    train_pdf = kernel.complete_data(dataset=0)

    train_pdf.index = original_index

    return train_pdf
