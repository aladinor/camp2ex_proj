#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from src.backend import dt_aircraft, get_sensors, get_hour, get_minutes, get_seconds, plot_nsd
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from datetime import date

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="80px")),
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

        html.H1(children="Filter options", className="display-4"),
        # html.Hr(className="my-2"),

        html.Label("Aircraft", style={"marginTop": 20}, className="lead"),
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
            placeholder="Cloud probe",
            searchable=True
        ),

        html.Label("Date", className="lead"),
        dcc.Dropdown(
            id="drop-days",
            clearable=True,
            multi=False,
            style={"marginBottom": 10, "font-size": 12},
            placeholder="Day",
            searchable=True,
        ),
        html.Label("Hour", className="lead"),
        dcc.Dropdown(
            id="drop-hour",
            clearable=True,
            multi=False,
            style={"marginBottom": 10, "font-size": 12},
            placeholder="Hour",
            searchable=True,
        ),
        html.Label("Minute", className="lead"),
        dcc.Dropdown(
            id="drop-minute",
            clearable=True,
            multi=False,
            style={"marginBottom": 10, "font-size": 12},
            placeholder="Hour",
            searchable=True,
        ),
        html.Label("Second slider", className="lead"),
        html.Div([
            dcc.Slider(
                min=0,
                max=10,
                step=1,
                id='time-slider',
                tooltip={"placement": "bottom", "always_visible": True},
                marks=None
                ),
            ]
        ),
    ]
)

MIDDLE_COLUMN = [
    dbc.CardHeader(html.H5("Results")),
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
                    dcc.Graph(id='plot-map', style=dict(aling='centered')),
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


@app.callback(
    Output("drop-sensor", "options"),
    Output("drop-days", "options"),
    [
        Input("drop-aircraft", "value")
    ],
)
def update_sensor_day(aircraft):
    if not aircraft:
        raise PreventUpdate
    else:
        sensor, _date = get_sensors(aircraft)
        return sensor, _date


@app.callback(
    Output('drop-hour', "options"),
    [
        Input("drop-days", "value")
    ],
    [
        State("drop-aircraft", "value"),
        State("drop-sensor", "value"),
    ]
)
def update_hour(_date=None, aircraft=None, sensor=None):
    if not _date:
        raise PreventUpdate
    else:
        return get_hour(aircraft=aircraft, ls_sensor=sensor, day=_date)


@app.callback(
    Output('drop-minute', "options"),
    [
        Input("drop-hour", "value")
    ],
    [
        State("drop-aircraft", "value"),
        State("drop-sensor", "value"),
        State("drop-days", "value"),
    ]
)
def update_minutes(hour=None, aircraft=None, sensor=None, date=None, ):
    if not hour:
        raise PreventUpdate
    else:
        return get_minutes(aircraft=aircraft, ls_sensor=sensor, day=date, _hour=hour)


@app.callback(
    Output('time-slider', 'min'),
    Output('time-slider', 'max'),
    [
        Input("drop-minute", "value"),
    ],
    [
        State("drop-aircraft", "value"),
        State("drop-sensor", "value"),
        State("drop-days", "value"),
        State("drop-hour", "value"),
    ]
)
def update_slider(minute=None, aircraft=None, sensor=None, date=None, hour=None, ):
    if not minute:
        raise PreventUpdate
    else:
        return get_seconds(aircraft=aircraft, ls_sensor=sensor, day=date, _hour=hour, minute=minute)


@app.callback(
    Output('plot-map', 'figure'),
    [
        Input("time-slider", "value")
    ],
    [
        State("drop-aircraft", "value"),
        State("drop-sensor", "value"),
        State("drop-days", "value"),
        State("drop-hour", "value"),
        State("drop-minute", "value"),
    ]
)
def update_figure(second=None, aircraft=None, sensor=None, date=None, hour=None, minute=None):
    if not second:
        raise PreventUpdate
    else:
        if (sensor is None) or (date is None) or (aircraft is None) or (hour is None) or (minute is None) \
                or (second is None):
            raise PreventUpdate
        else:
            return plot_nsd(aircraft=aircraft, ls_sensor=sensor, day=date, _hour=hour, minute=minute, second=second)


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
