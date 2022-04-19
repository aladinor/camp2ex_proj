#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
import sys
import os
import base64
import pandas as pd
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from re import split

sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini
from src.backend import dt_aircraft, get_sensors, get_hour, get_minutes, get_seconds, plot_map, title, \
    psd_fig, lear_df, p3_df, ls_p3_merged, plot_temp, z_table

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(campaign='loc')[location]['path_data']

PLOTLY_LOGO = f"{path_data}/data/CAMPEX_Logo_Lg.png"
img = base64.b64encode(open(PLOTLY_LOGO, 'rb').read())

NAVBAR = dbc.Navbar(
    html.Div(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(src='data:image/png;base64,{}'.format(img.decode()), height="60px"
                                     )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand("  CAMP2Ex - UIUC - CSRG Dashboard", className="ms-2",
                                            style=dict(size='10px'))
                        ),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://www-air.larc.nasa.gov/missions/camp2ex/index.html",
                style={"textDecoration": "none"},
            )
        ],
        style={"marginLeft": 30},
    ),
    color="dark",
    dark=True,
)

LEFT_COLUMN = dbc.Card(
    [
        dbc.CardHeader(html.H4("Filter Options")),
        dbc.CardBody(
            [
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
            ],
        )
    ],
)

MIDDLE_COLUMN = dbc.Card(
    [
        dbc.CardHeader(html.H4("Results")),
        dbc.CardBody(
            [
                dbc.Row(
                    dbc.Col(
                        html.H5(
                            html.Div(id='title-output', className='text-center')
                        ),
                    )
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row([dcc.Graph(id='plot-cop')], justify="center",
                                        style={"height": "100%"},  # , "background-color": "yellow"},
                                        className="h-50", ),
                                dbc.Row([dcc.Graph(id='plot-temp-alt')], justify="center",
                                        style={"height": "100%"},  # , "background-color": "green"},
                                        className="h-50", )
                            ],
                            align='center',
                            width=6
                        ),
                        dbc.Col(
                            [
                                dbc.Row([dcc.Graph(id='plot-table')], justify="center",
                                        style={"height": "100%"},  # , "background-color": "red"},
                                        className="h-50",
                                        ),
                                dbc.Row([dcc.Graph(id='plot-map')], justify="center",
                                        style={"height": "100%"},  # , "background-color": "blue"},
                                        className="h-50", ),

                            ],

                            align='center',
                            width=6
                        ),

                    ],
                )

            ],
            style={"height": "1000vh"},
        )
    ]
)
BODY = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=3),
                dbc.Col(MIDDLE_COLUMN, md=9),
            ],
            style={"marginTop": 30, "marginLeft": 30, "marginRight": 30},
        )
    ]
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
    if (sensor is None) or (options_1 is None) or (sensor is None):
        raise PreventUpdate
    elif len(selected) > 0:
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
    if (_date is None) or (aircraft is None) or (sensor is None):
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
    if (hour is None) or (aircraft is None) or (sensor is None) or (date is None):
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
    if (hour is None) or (aircraft is None) or (sensor is None) or (date is None) or (minute is None):
        raise PreventUpdate
    else:
        return get_seconds(aircraft=aircraft, ls_sensor=sensor, day=date, _hour=hour, minute=minute)


@app.callback(
    [Output('plot-cop', 'figure'),
     Output('plot-map', 'figure'),
     Output('title-output', 'children'),
     Output('plot-temp-alt', 'figure'),
     Output('plot-table', 'figure')],
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
    if (aircraft is None) or (hour is None) or (minute is None) or (second is None):
        df = lear_df[-1]
        idx = pd.Timestamp(year=2019, month=9, day=7, hour=10, minute=32, second=21, tz='Asia/Manila')
        df = df.groupby(by=df['local_time'].dt.floor('d')).get_group(pd.Timestamp(idx.date(), tz='Asia/Manila'))
        _lear_df = [i.groupby(by=i['local_time'].dt.floor('d')).get_group(pd.Timestamp(idx.date(), tz='Asia/Manila'))
                    for i in lear_df]
        return psd_fig(_idx=idx, ls_df=_lear_df), plot_map(idx, df), title(aircraft, idx), \
               plot_temp(idx, df, aircraft=aircraft), z_table(_idx=idx, ls_df=_lear_df[:-1])
    else:
        idx = pd.to_datetime(minute) + pd.to_timedelta(int(second), unit='s')
        if aircraft == 'P3B':
            ls_df = [i for i in p3_df if i.attrs['type'] in sensor]
            df = pd.read_pickle(ls_p3_merged[0])  # temperature and other variables
            df = df.groupby(by=df['local_time'].dt.floor('d')).get_group(pd.Timestamp(idx.date(), tz='Asia/Manila'))
            df.rename(columns={' Latitude_YANG_MetNav': 'Lat', ' Longitude_YANG_MetNav': 'Long',
                               ' Pressure_Altitude_YANG_MetNav': 'Palt', ' Total_Air_Temp_YANG_MetNav': 'Temp',
                               ' Dew_Point_YANG_MetNav': 'Dew', ' LWC_gm3_LAWSON': 'NevLWC'}, inplace=True)
            # df = df[['Lat', 'Long', 'Temp', 'NevLWC', 'local_time', 'Dew']]
        else:
            df = lear_df[-1]  # temperature and other variables
            df = df.groupby(by=df['local_time'].dt.floor('d')).get_group(pd.Timestamp(idx.date(), tz='Asia/Manila'))
            ls_df = [i for i in lear_df if i.attrs['type'] in sensor]
        df = df.replace(-999999.0, pd.NA)
        return psd_fig(_idx=idx, ls_df=ls_df), plot_map(idx, df), title(aircraft, idx), \
               plot_temp(idx, df, aircraft=aircraft), z_table(_idx=idx, ls_df=ls_df)


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
    app.run_server(host='127.0.0.1', port=8054, debug=True)
    pass
