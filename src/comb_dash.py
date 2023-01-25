import glob
import sys
import os
import dash
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import xarray as xr
from re import split
from dash import dcc

app = dash.Dash()
sys.path.insert(1, f"{os.path.abspath(os.path.join(os.path.abspath(''), '../'))}")
from src.utils import get_pars_from_ini

location = split(', |_|-|!', os.popen('hostname').read())[0].replace("\n", "")
path_data = get_pars_from_ini(file_name='loc')[location]['path_data']

ds_files = glob.glob(f'{path_data}/cloud_probes/zarr/combined_psd_Lear*.zarr')
ds_probes = glob.glob(f'{path_data}/cloud_probes/zarr/*Learjet.zarr')

df = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/'
    'cb5392c35661370d95f300086accea51/raw/'
    '8e0768211f6b747c0db42a9ce9a0937dafcbd8b2/'
    'indicators.csv')

available_indicators = df['Indicator Name'].unique()


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

        # html.Div([
        #     dcc.Dropdown(
        #         id='crossfilter-yaxis-column',
        #         options=[{'label': i, 'value': i} for i in available_indicators],
        #         value='Life expectancy at birth, total (years)'
        #     ),
        #     dcc.RadioItems(
        #         id='crossfilter-yaxis-type',
        #         options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
        #         value='Linear',
        #         labelStyle={'display': 'inline-block'}
        #     )
        # ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='dm-nw',
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    # html.Div([
    #     dcc.Graph(id='x-time-series'),
    #     dcc.Graph(id='y-time-series'),
    # ], style={'display': 'inline-block', 'width': '49%'}
    # ),

    # html.Div(dcc.Slider(
    #     id='crossfilter-year--slider',
    #     min=df['Year'].min(),
    #     max=df['Year'].max(),
    #     value=df['Year'].max(),
    #     step=None,
    #     marks={str(year): str(year) for year in df['Year'].unique()}
    # ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


@app.callback(
    dash.dependencies.Output('dm-nw', 'figure'),
    [dash.dependencies.Input('zarr_file', 'value')])
def update_graph(file):
    ds = xr.open_zarr(file)
    print(1)
    return {
        'data': [go.Scatter(
            x=ds.dm,
            y=ds.log10_nw,
            mode='markers',
            marker={
                'size': 10,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': 'Dm (mm)',
                'type': 'linear'
            },
            yaxis={
                'title': 'log10(Nw)',
                'type': 'log'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }


def create_time_series(dff, axis_type, title):
    return {
        'data': [go.Scatter(
            x=dff['Year'],
            y=dff['Value'],
            mode='lines+markers'
        )],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear' if axis_type == 'Linear' else 'log'},
            'xaxis': {'showgrid': False}
        }
    }


#
# def psd_fig(_idx, ls_df):
#     layout = go.Layout(autosize=True)
#     fig = go.Figure(layout=layout)
#     for i in ls_df:
#         x = i.attrs['sizes']
#         try:
#             y = i.loc[i['local_time'] == _idx, i.columns].filter(like='nsd').values[0] # * 1e6
#             y = np.where(y > 0, y, np.nan)
#             fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=i.attrs['type']))
#         except IndexError:
#             pass
#     fig.update_traces(mode="lines", line_shape="vh")
#     fig.update_yaxes(title_text="Concentration (# / L um)", type="log", showgrid=False, exponentformat='power',
#                      showexponent='all')
#     fig.update_xaxes(title_text="Diameter (um)", type="log", exponentformat='power', showexponent='all')
#     fig.update_layout(legend=dict(y=0.99, x=0.7), margin=dict(l=20, r=20, t=20, b=20))
#     return fig
# #
# @app.callback(
#     dash.dependencies.Output('x-time-series', 'figure'),
#     [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#      dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
# def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
#     country_name = hoverData['points'][0]['customdata']
#     dff = df[df['Country Name'] == country_name]
#     dff = dff[dff['Indicator Name'] == xaxis_column_name]
#     title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
#     return create_time_series(dff, axis_type, title)
#
#
# @app.callback(
#     dash.dependencies.Output('y-time-series', 'figure'),
#     [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#      dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
# def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
#     dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
#     dff = dff[dff['Indicator Name'] == yaxis_column_name]
#     return create_time_series(dff, axis_type, yaxis_column_name)


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8054)

