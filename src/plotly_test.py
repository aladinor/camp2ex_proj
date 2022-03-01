#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from src.backend import dt_sensor, dt_day, dt_aircraft
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("CAMP2Ex cloud probes", className="ml-2")
                    ),
                ],
                align="center",
            ),
            href="https://plot.ly",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

LEFT_COLUMN = dbc.Col(
    [
        html.H4(children="Filter option", className="display-5"),
        html.Hr(className="my-2"),

        html.Label("Date", className="lead"),
        dcc.Dropdown(
            id="drop-days",
            clearable=True,
            multi=False,
            style={"marginBottom": 10, "font-size": 12},
            options=dt_day,
            placeholder="Day",
            searchable=True,
        ),

        html.Label("Aircraf", style={"marginTop": 20}, className="lead"),
        dcc.Dropdown(
            id="drop-aircraft",
            clearable=True,
            multi=False,
            style={"marginBottom": 10, "font-size": 12},
            options=dt_aircraft,
            placeholder="Aircraft",
            searchable=True
        ),

        html.Label("Sensor", style={"marginTop": 20}, className="lead"),
        dcc.Dropdown(
            id="drop-sensor",
            clearable=True,
            multi=True,
            style={"marginBottom": 10, "font-size": 12},
            options=dt_sensor,
            placeholder="Cloud probe",
            searchable=True
        ),

        dbc.Button(
            id="submit-dpto",
            n_clicks=0,
            children='Send',
            size="sm",
        ),
        html.Br(),
        html.Br(),
    ]
)

MIDDLE_COLUMN = [
    dbc.CardHeader(html.H5("Resultados de la consulta")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="table-res",
                type='circle',
                # type="default",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-bank",
                        color="warning",
                        style={"display": "none"},
                    ),
                    # dcc.Graph(id='plot-map', style=dict(aling='centered')),
                    # dcc.Graph(id="plot-table"),
                ],
            )
        ],
        style={"marginTop": 0, "marginBottom": 0, 'display': 'flex'},
    ),
]


BODY = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=4),
                dbc.Col(dbc.Card(MIDDLE_COLUMN), md=8),
                # dbc.Col(dbc.Card(RIGHT_COLUMN), md=3),
            ],
            style={"marginTop": 30},
        ),
    ],
    className="mt-12",
)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(children=[NAVBAR, BODY])


def wait_for():
    return dbc.CardBody(
        [
            dcc.Loading(
                id="table-res",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-bank",
                        color="warning",
                        style={"display": "none"},
                    ),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0, 'display': 'flex'},
    )


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8051', debug=True)
    pass
