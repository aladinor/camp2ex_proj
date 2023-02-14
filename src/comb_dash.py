import glob
import sys
import os
import dash
import plotly.graph_objs as go
import numpy as np
import xarray as xr
from re import split
from dash import dcc
from dash import html
from scipy.special import gamma
from numpy import arange

app = dash.Dash()
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini
from src.pds_retrievals import eq_funct

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']

ds_files = glob.glob(f'{path_data}/cloud_probes/zarr/combined_psd_Lear*.zarr')
ds_probes = glob.glob(f'{path_data}/cloud_probes/zarr/*Learjet.zarr')
ds_probes = [i for i in ds_probes if i.replace("\\", '/').split('/')[-1].split('_')[0] in ['2DS10', 'HVPS']]
dm = xr.open_zarr(f'{path_data}/cloud_probes/zarr/dm_estimation.zarr')
_dm = arange(0.1, 4, 0.001)
dms = xr.DataArray(data=_dm,
                   dims=['dm'],
                   coords=dict(dm=(['dm'], _dm)))

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='zarr_file',
                options=[{'label': i.split('/')[-1], 'value': i} for i in ds_files],
                value=ds_files[0]
            ),
        ],
            style={'width': '49%', 'display': 'inline-block'}),

    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='dm_nw',
            hoverData={'points': [{'hovertext': '2019-09-07 2:32:08'}]}
        )
    ],
        style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='psd_series'),
    ],
        style={'display': 'inline-block', 'width': '48%'}
    ),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='dm_var',
                options=[{'label': i.split('/')[-1], 'value': i} for i in list(dm.keys())],
                value='dm_dfr'
            ),
        ],
            style={'width': '49%', 'display': 'inline-block'}),
    ]),
    html.Div([
        dcc.Graph(id='dm_dmest',
                  hoverData={'points': [{'hovertext': '2019-09-07 2:31:52'}]}
                  ),
    ],
        style={'display': 'inline-block', 'width': '48%'}),
    html.Div([
        dcc.Graph(id='dfr_ib'),
    ],
        style={'display': 'inline-block', 'width': '48%'},
    ),
])


@app.callback(
    dash.dependencies.Output('dm_nw', 'figure'),
    [dash.dependencies.Input('zarr_file', 'value')])
def update_graph(file):
    ds = xr.open_zarr(file)
    ds = ds.where(ds.mu < 10)
    return {
        'data': [go.Scatter(
            x=ds.dm,
            y=ds.log10_nw,
            hovertext=ds.time,
            mode='markers',
            marker={
                'size': 8,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'},
                'color': ds.mu,
                'colorscale': 'jet',
                'colorbar': dict(thickness=5, outlinewidth=0),
                'cmin': -2,
                'cmax': 10
            },
        )],
        'layout': go.Layout(
            xaxis={
                'title': 'Dm (mm)',
                'type': 'linear'
            },
            yaxis={
                'title': 'log10(Nw)',
                'type': 'linear'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }


@app.callback(
    dash.dependencies.Output('dm_dmest', 'figure'),
    [dash.dependencies.Input('dm_var', 'value')])
def update_dm(_var):
    print(1)
    return {
        'data': [go.Scatter(
            x=dm.dm,
            y=dm[_var],
            hovertext=dm.time,
            mode='markers',
            marker={
                'size': 8,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'},
                'colorscale': 'jet',
                'colorbar': dict(thickness=5, outlinewidth=0),
                'cmin': -2,
                'cmax': 10
            },
        )],
        'layout': go.Layout(
            xaxis={
                'title': 'Dm (mm)',
                'type': 'linear'
            },
            yaxis={
                'title': 'Dm GPM',
                'type': 'linear'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }


def norm_gamma(d, nw, dm, mu):
    return nw * (6 * (mu + 4) ** (mu + 4)) / (4 ** 4 * gamma(mu + 4)) * (d / dm) ** mu * np.exp(-(mu + 4) * d / dm)


def create_time_series(date, xr_comb):
    ds_2ds = xr.open_zarr(ds_probes[0])
    x_2ds = ds_2ds.diameter / 1e3
    y_2ds = ds_2ds.psd.sel(time=date) * 1e6
    y_2ds = np.where(y_2ds > 0, y_2ds, np.nan)

    ds_hvps = xr.open_zarr(ds_probes[1])
    x_hvps = ds_hvps.diameter / 1e3
    y_hvps = ds_hvps.psd.sel(time=date) * 1e6
    y_hvps = np.where(y_hvps > 0, y_hvps, np.nan)

    x_comb = xr_comb.diameter / 1e3
    y_comb = xr_comb.psd * 1e6
    y_comb = np.where(y_comb > 0, y_comb, np.nan)
    nd = norm_gamma(d=xr_comb.diameter / 1e3, dm=xr_comb.dm, mu=xr_comb.mu, nw=xr_comb.nw)
    nd = np.where(nd > 0, nd, np.nan)
    nd_2 = norm_gamma(d=xr_comb.diameter / 1e3, dm=xr_comb.dm, mu=xr_comb.new_mu, nw=xr_comb.nw)
    nd_2 = np.where(nd > 0, nd, np.nan)
    return {
        'data': [go.Scatter(
            x=x_2ds,
            y=y_2ds,
            mode="lines",
            line_shape="vh",
            name='2DS',
            line=dict(width=2)
        ),

            go.Scatter(
                x=x_hvps,
                y=y_hvps,
                mode="lines",
                line_shape="vh",
                name='HVPS',
                line=dict(width=2)
            ),
            go.Scatter(
                x=x_comb,
                y=y_comb,
                mode="lines",
                line_shape="vh",
                name='COMB.',
                line=dict(color='black', width=1),
            ),
            go.Scatter(
                x=x_comb,
                y=nd,
                mode="lines",
                line_shape="vh",
                name='Norm. Gamma',
                line=dict(color='red', width=1)
            ),
            go.Scatter(
                x=x_comb,
                y=nd_2,
                mode="lines",
                line_shape="vh",
                name='Norm. Gamma (new mu)',
                line=dict(color='blue', width=1)
            )
        ],
        'layout': go.Layout(
            xaxis=dict(title="D (mm)", type='log', range=[-3, 1]),
            yaxis=dict(title="N(d) (mm-3 m-3)", type='log', range=[0, 10]),
            annotations=[{
                'x': 0, 'y': 1.05, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': f'{date}'},
                {
                    'x': 0.5, 'y': 0.9, 'xanchor': 'left', 'yanchor': 'bottom',
                    'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                    'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                    'text': f'dm={xr_comb.dm.values:.2f}, mu={xr_comb.mu.values:.2f}, '
                            f'log10(nw)={xr_comb.log10_nw.values:.2f}',
                }]
        )
    }


@app.callback(
    dash.dependencies.Output('psd_series', 'figure'),
    [
        # dash.dependencies.Input('dm_nw', 'hoverData'),
        dash.dependencies.Input('dm_dmest', 'hoverData'),
        dash.dependencies.Input('zarr_file', 'value'),
    ])
def update_y_timeseries(hoverData, file):
    date = hoverData['points'][0]['hovertext']
    xr_comb = xr.open_zarr(file).sel(time=date)
    return create_time_series(date, xr_comb)


def dfr_plot(xr_comb):
    dfr_ib = eq_funct(dm=dms, xr_comb=xr_comb, mu=xr_comb.mu)
    return {
        'data': [go.Scatter(
            x=dfr_ib.dm,
            y=dfr_ib.values,
            mode="lines",
            line=dict(width=2)
        )
        ],
        'layout': go.Layout(
            xaxis={
                'title': 'Dm (mm)',
                'type': 'linear'
            },
            yaxis={
                'title': 'DFR - Ib(Ku) + Ib(Ka)',
                'type': 'linear'
            },
        )
    }


@app.callback(
    dash.dependencies.Output('dfr_ib', 'figure'),
    [
        # dash.dependencies.Input('dm_nw', 'hoverData'),
        dash.dependencies.Input('dm_dmest', 'hoverData'),
        dash.dependencies.Input('zarr_file', 'value'),
    ])
def update_dfr(hoverData, file):
    date = hoverData['points'][0]['hovertext']
    xr_comb = xr.open_zarr(file).sel(time=date)
    return dfr_plot(xr_comb)


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8055)
