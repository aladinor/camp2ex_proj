#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
import sys
import os
import base64
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from src.backend import dt_aircraft, get_sensors, get_hour, get_minutes, get_seconds, plot_nsd, plot_map
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from re import split

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(campaign='loc')[location]['path_data']

PLOTLY_LOGO = f"{path_data}/data/CAMPEX_Logo_Lg.png"
img = base64.b64encode(open(PLOTLY_LOGO, 'rb').read())

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src='data:image/png;base64,{}'.format(img.decode()), height="100px"
                                     # style={'height': '40%'}
                                     )),
                    dbc.Col(
                        dbc.NavbarBrand("CAMP2Ex - UIUC Dashboard", className="ms-2")
                    ),
                ],
                # align="center",
            ),
            href="https://www-air.larc.nasa.gov/missions/camp2ex/index.html",
        )
    ],
    color="dark",
    dark=True,
    # sticky="top",
)

LEFT_COLUMN = dbc.Col(
    [
        html.H6(children="Filter options", className="display-4"),
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
        dcc.Checklist(id='select-all',
                      options=[{'label': 'Select All', 'value': 1}], value=[],
                      style={"marginBottom": 10, "font-size": 11.5},
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
            placeholder="Minute",
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
            ),
        ]
        ),
    ]
)

MIDDLE_COLUMN = [
    dbc.CardHeader(html.H5("Results")),
    dbc.CardBody(
        [
            html.Div([
                dcc.Graph(id='plot-cop',
                          style={'align': 'left', 'width': '49%'}
                          ),
            ]
            ),
            html.Div([
                dcc.Graph(id='plot-map',
                          style={'align': 'left', 'width': '49%'}
                          ),
            ]
            ),
        ],
        style={"marginTop": 0, "marginBottom": 0, 'display': 'flex'},
    ),

]

BODY = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, lg=3, md=12),
                dbc.Col(dbc.Card(MIDDLE_COLUMN), lg=9, md=12),
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
    Output("drop-sensor", 'value'),
    [Input('select-all', 'value')],
    [State("drop-aircraft", "value"),
     State('drop-sensor', 'options')])
def test(selected, options_1, sensor):
    if len(selected) > 0:
        return [i['value'] for i in sensor]
    else:
        return []


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
    [Output('plot-cop', 'figure'),
     Output('plot-map', 'figure')],
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
    return plot_nsd(aircraft=aircraft, ls_sensor=sensor, _hour=hour, minute=minute, second=second), \
           plot_map(aircraft=aircraft, date=date)


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
    app.run_server(host='127.0.0.1', port=8053, debug=True)
    pass
