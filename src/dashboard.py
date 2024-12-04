import pandas as pd
import dash
from dash import dcc, html

from src.constants import UPDATE_DASHBOARD_EVERY_S


def get_signal_options(csv_path='train_data.csv'):
    df = pd.read_csv(csv_path)
    df[pd.isna(df)] = None

    # Create a list of signal options for the dropdown
    signal_options = [{
        'label': f"{i}: {row['ID_SIGNAL']}",
        'value': row['ID_SIGNAL']
    } for i, row in df.iterrows()]

    return signal_options


def get_app(signal_options):

    app = dash.Dash(__name__, title='Anomaly Detection Dashboard')

    app.layout = html.Div([
        html.Div(
            dcc.Dropdown(
                id='signal-dropdown',
                options=signal_options,
                value=signal_options[0]['value'],  # Default value
                style={'font-family': 'Verdana', 'font-size': '14px'},
                clearable=False
            ),
            style={'width': '30%'}
        ),
        dcc.Graph(id='live-update-graph'),
        html.Div([
            dcc.Textarea(
                id='log-console',
                style={'width': '69%', 'height': '220px', 'margin-right': '1%'},
                readOnly=True
            ),
            dcc.Textarea(
                id='second-log-console',
                style={'width': '30%', 'height': '220px'}
            )
        ], style={'display': 'flex', 'width': '100%'}),
        dcc.Interval(
            id='interval-component',
            interval=UPDATE_DASHBOARD_EVERY_S * 1000,  # in milliseconds
            n_intervals=0
        ),
        dcc.Store(id='zoom_info')
    ])

    return app


app = get_app(get_signal_options())
