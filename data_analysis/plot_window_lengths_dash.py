# dash_app.py
import re
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# -----------------------
# Config
CSV_PATH = "./outputs/rep_samples_per_test.csv"  # change if needed
# -----------------------

# Load data
df = pd.read_csv(CSV_PATH)
df["test_id"] = df["test_id"].astype(str)
df["num_samples"] = pd.to_numeric(df["num_samples"], errors="coerce")

# Derive 'activity' from rep_id (strip trailing _<digits>)
def extract_activity(s: str) -> str:
    m = re.match(r"^(.*)_(\d+)$", str(s))
    return m.group(1) if m else str(s)

df["activity"] = df["rep_id"].apply(extract_activity)

# Category orders (for consistent axis ordering)
activity_order = sorted(df["activity"].unique().tolist())
test_order = sorted(
    df["test_id"].unique().tolist(),
    key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x
)

app = Dash(__name__)
app.title = "Window Length Variability"

def make_figure(df_plot: pd.DataFrame):
    # jittered categorical scatter (strip plot)
    fig = px.strip(
        df_plot,
        x="activity",
        y="num_samples",
        category_orders={"activity": activity_order},
        hover_data={"test_id": True, "rep_id": True, "activity": False},  # test_id on hover
        stripmode="overlay",   # all points overlaid
        #jitter=0.35,           # horizontal jitter to avoid overlap
    )

    fig.update_traces(
        marker=dict(size=6, opacity=0.7),
        selector=dict(type="scatter")
    )

    fig.update_layout(
        height=600,
        margin=dict(l=60, r=20, t=60, b=120),
        title="Per-Repetition Window Lengths (num_samples) by Activity",
        xaxis_title="Activity",
        yaxis_title="Number of Samples (window length)",
        xaxis=dict(tickangle=45),
    )

    # Add light grid
    fig.update_yaxes(showgrid=True, gridwidth=1, griddash="dot")
    fig.update_xaxes(showgrid=False)

    # Cleaner hover
    fig.update_traces(
        hovertemplate=(
            "activity=%{x}<br>"
            "num_samples=%{y}<br>"
            "test_id=%{customdata[0]}<br>"
            "rep_id=%{customdata[1]}<extra></extra>"
        ),
        customdata=df_plot[["test_id", "rep_id"]]
    )

    return fig

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "sans-serif"},
    children=[
        html.H2("Window Length Variability (Scatter by Activity)"),
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"minWidth": "260px", "flex": "1"},
                    children=[
                        html.Label("Filter by Activity"),
                        dcc.Dropdown(
                            id="activity-filter",
                            options=[{"label": a, "value": a} for a in activity_order],
                            value=activity_order,  # all selected by default
                            multi=True,
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "260px", "flex": "1"},
                    children=[
                        html.Label("Filter by Test"),
                        dcc.Dropdown(
                            id="test-filter",
                            options=[{"label": t, "value": t} for t in test_order],
                            value=test_order,  # all selected by default
                            multi=True,
                        ),
                    ],
                ),
            ],
        ),
        dcc.Graph(id="scatter-graph", figure=make_figure(df)),
    ],
)

@app.callback(
    Output("scatter-graph", "figure"),
    Input("activity-filter", "value"),
    Input("test-filter", "value"),
)
def update_figure(selected_activities, selected_tests):
    dff = df.copy()
    if selected_activities:
        dff = dff[dff["activity"].isin(selected_activities)]
    if selected_tests:
        dff = dff[dff["test_id"].isin(selected_tests)]
    return make_figure(dff)

if __name__ == "__main__":
    app.run(debug=True)