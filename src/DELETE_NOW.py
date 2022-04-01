import dash
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash()

app.layout = html.Div([

    dcc.Dropdown(id='dropdown', multi=True,
                 options=[{'label': i, 'value': i} for i in range(10)], value=[1]),
    dcc.Checklist(id='select-all',
                  options=[{'label': 'Select All', 'value': 1}], value=[])
])

@app.callback(
    Output('dropdown', 'value'),
    [Input('select-all', 'value')],
    [State('dropdown', 'options'),
     State('select-all', 'options')])
def test(selected, options_1):
    if len(selected) > 0:
        return [i['value'] for i in options_1]
    else:
        return []


@app.callback(
    Output('select-all', 'values'),
    [Input('dropdown', 'value')],
    [State('dropdown', 'options')])
def tester(selected, options_1):
    print(selected)
    if len(selected) < len(options_1):
        return []


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)