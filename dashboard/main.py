import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

PRECISION1 = 0.016291193579710775
CSV_PATH = "transactions.csv"
REFRESH_SECONDS = 30          # relecture automatique du fichier
COLORS = {0: "#4B2EE8", 1: "#DF8355"}
LABELS = {0: "0 - Pas Fraude", 1: "1 - Fraude"}


# Données
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["temps"] = pd.to_datetime(df["temps"].str.replace(" UTC", "", regex=False))
    df["predictions"] = df["predictions"].astype(int)
    # Un "lot" = un horodatage arrondi à la seconde
    df["lot"] = df["temps"].dt.floor("s")
    return df


def fmt(n: float) -> str:
    """Format"""
    for seuil, suffixe in [(1e9, "Md"), (1e6, "M"), (1e3, "k")]:
        if abs(n) >= seuil:
            return f"{n / seuil:,.2f}{suffixe}".replace(",", " ").replace(".", ",")
    return f"{n:,.0f}".replace(",", " ")


def base_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        font=dict(family="Segoe UI, Arial", size=12),
    )
    return fig


# Graphiques
def fig_total_par_lot(df: pd.DataFrame) -> go.Figure:
    """Histogramme empilé pour observer la 
    distribution des transactions dans le temps"""
    g = df.groupby(["lot", "predictions"])["amount"].sum().reset_index()
    fig = go.Figure()
    for p in (0, 1):
        d = g[g["predictions"] == p]
        fig.add_bar(x=d["lot"], y=d["amount"], name=LABELS[p], marker_color=COLORS[p])
    fig.update_layout(barmode="stack", xaxis_title="Date",
                      yaxis_title="Montant total")
    fig.update_xaxes(type="category")

    fig = base_layout(fig)

    fig.update_layout(
        legend=dict(
            orientation="v",
            x=0.99, xanchor="right",
            y=1.5, yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        )
    )
    return fig


def fig_moyen_par_type(df: pd.DataFrame) -> go.Figure:
    g = df.groupby(["type", "predictions"])["amount"].mean().reset_index()
    fig = go.Figure()
    for p in (0, 1):
        d = g[g["predictions"] == p]
        fig.add_bar(y=d["type"], x=d["amount"], name=LABELS[p],
                    marker_color=COLORS[p], orientation="h")
    fig.update_layout(barmode="stack", xaxis_title="Montant moyen")
    return base_layout(fig)


def fig_donut(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("predictions")["amount"].sum().reindex([0, 1], fill_value=0)
    fig = go.Figure(go.Pie(
        labels=[LABELS[i] for i in g.index],
        values=g.values,
        hole=0.78,
        marker_colors=[COLORS[i] for i in g.index],
        texttemplate="%{percent:.2%}",
        sort=False,
    ))
    fig.add_annotation(text=f"<b>{fmt(g.sum())}</b>", showarrow=False,
                       font=dict(size=26))
    return base_layout(fig)


def fig_taux_fraude(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("lot").agg(taux=("predictions", "mean"), n=("predictions", "size"))
    fig = go.Figure(go.Scatter(x=g.index, y=g["taux"] * 100, mode="lines+markers",
                               line=dict(color=COLORS[1]), name="Taux de fraude"))
    moyenne = df["predictions"].mean() * 100
    fig.add_hline(y=moyenne, line_dash="dash", line_color="grey",
                  annotation_text=f"moyenne {moyenne:.1f}%")
    fig.update_layout(yaxis_title="% de transactions flaguées")
    return base_layout(fig, height=300)


# Mise en page
CARD = {"background": "white", "padding": "14px 18px", "borderRadius": "6px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.12)"}


def card(title: str, subtitle: str, graph_id: str, flex: str = "1") -> html.Div:
    return html.Div(
        [html.H4(title, style={"margin": 0}),
         html.Div(subtitle, style={"color": "#777", "fontSize": 13}),
         dcc.Graph(id=graph_id, config={"displayModeBar": False})],
        style={**CARD, "flex": flex},
    )


def kpi(title: str, value_id: str, color: str = "#222") -> html.Div:
    return html.Div(
        [html.Div(title, style={"color": "#777", "fontSize": 14}),
         html.Div(id=value_id, style={"fontSize": 32, "fontWeight": 600, "color": color})],
        style={**CARD, "flex": "1", "textAlign": "left"},
    )


app = Dash(__name__)
app.title = "Détection de fraude (monitoring)"

app.layout = html.Div(
    style={"background": "#f3f3f3", "padding": "16px", "fontFamily": "Segoe UI, Arial"},
    children=[
        html.H2("Détection de fraude (monitoring)", style={"marginTop": 0}),
        dcc.Interval(id="tick", interval=REFRESH_SECONDS * 1000),

        # Ligne de KPI
        html.Div([
            kpi("Montant économisé (fraudes bloquées)", "k_saved", COLORS[1]),
            kpi("Montant économisé / Montant total de fraude (%)", "k_total"),
            kpi("Nb transactions", "k_n"),
            kpi("Taux de fraude détectée", "k_rate"),
            kpi("Dernier lot", "k_last"),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        # Graphique principal
        html.Div([card("Montant total des transactions", "par Date et étiquettes",
                       "g_total")], style={"marginBottom": "12px"}),

        # Ligne du bas
        html.Div([
            card("Montant moyen d'une transaction", "par type de transaction et étiquettes",
                 "g_moyen"),
            card("Distribution des transactions", "par étiquettes", "g_donut"),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        # Santé du modèle
        html.Div([card("Taux de fraude détectée par lot",
                       "dérive possible si la courbe s'écarte fortement de la moyenne",
                       "g_rate")]),
    ],
)


@app.callback(
    Output("g_total", "figure"), Output("g_moyen", "figure"),
    Output("g_donut", "figure"), Output("g_rate", "figure"),
    Output("k_saved", "children"), Output("k_total", "children"),
    Output("k_n", "children"), Output("k_rate", "children"),
    Output("k_last", "children"),
    Input("tick", "n_intervals"),
)
def refresh(_):
    df = load_data()
    saved = df.loc[df["predictions"] == 1, "amount"].sum()*PRECISION1
    return (
        fig_total_par_lot(df), fig_moyen_par_type(df), fig_donut(df), fig_taux_fraude(df),
        fmt(saved), f"{((saved)/(0.001*len(df)*1_456_248)) * 100:.2f} %", f"{len(df):,}".replace(",", " "),
        f"{df['predictions'].mean() * 100:.2f} %",
        df["lot"].max().strftime("%d/%m %H:%M:%S"),
    )


if __name__ == "__main__":
    app.run(debug=True)