import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

# 從 Excel 檔案中匯入資料
table1 = pd.read_excel("Table 1.xlsx")

# 將值轉換為百萬單位，除了稅率
for i in table1.columns[1:]:
    if table1[i].dtype in ['float64', 'int64']:  # 確保是數值型別
        table1[i] = table1[i].apply(lambda x: round(x / 1000000, 2) if x != 0.12 else x)

def DCF(FCF, WACC_lo, WACC_high, g):
    WACC_lo /= 100
    WACC_high /= 100
    g /= 100
    
    discounted_lo = sum([fcf / ((1 + WACC_lo) ** (i + 1)) for i, fcf in enumerate(FCF)])
    discounted_high = sum([fcf / ((1 + WACC_high) ** (i + 1)) for i, fcf in enumerate(FCF)])
    
    terminal_val_lo = FCF[-1] * (1 + g) / (WACC_lo - g)
    terminal_val_high = FCF[-1] * (1 + g) / (WACC_high - g)
    
    discounted_terminal_lo = terminal_val_lo / ((1 + WACC_lo) ** len(FCF))
    discounted_terminal_high = terminal_val_high / ((1 + WACC_high) ** len(FCF))
    
    total_lo = discounted_lo + discounted_terminal_lo
    total_high = discounted_high + discounted_terminal_high
    
    return total_lo, total_high

# 定義 multiple 資料
multiple = pd.DataFrame(
    [['PE Ratio','EV/EBITDA','EV/Revenue','PB'],
     [1428480,1512680,1279362,2320812],
     [873158,945425,852908,1160406]],
    index = ('Method','Hi', 'Lo'),
)

temp = multiple.T
temp['Hi'] = multiple.T['Hi'] - multiple.T['Lo']

# 定義 melted 資料
melted = temp.melt(id_vars='Method', var_name='Range')
melted.loc[8] = ['DCF', 'Hi', 4]
melted.loc[9] = ['DCF', 'Lo', 4]

app = dash.Dash(__name__)

#for deployment
server = app.server

WACC = [5, 6, 7, 8, 9, 10]
g = [0, 1, 2, 3, 4, 5, 6, 7, 8]

color_map = {
    0: '#72b206',
    1: '#aad361',
    2: '#5c940e',
    3: '#7c7c7b',
    4: '#44642c'
}

app.layout = html.Div([
    # 左側區域
    html.Div([
        html.H1("NVIDIA Business Valuation Dashboard", style={'textAlign': 'left'}),
        html.H3("Key Financial Table", style={'textAlign': 'left'}),
        # 滑桿
        html.Div([
            html.Div([
                html.Label('Data Center (g)'),
                dcc.Slider(
                    id='datacenter-slider',
                    min=10,
                    max=40,
                    step=0.1,
                    value=28,
                    marks={i: f'{i}%' for i in range(0, 36, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 10px'}),
            html.Div([
                html.Label('Gaming (g)'),
                dcc.Slider(
                    id='gaming-slider',
                    min=0,
                    max=20,
                    step=0.1,
                    value=10,
                    marks={i: f'{i}%' for i in range(0, 21, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 10px'}),
            html.Div([
                html.Label('Low Growth Income (g)'),
                dcc.Slider(
                    id='low-g-slider',
                    min=0,
                    max=10,
                    step=0.1,
                    value=3,
                    marks={i: f'{i}%' for i in range(0, 11, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 10px'})
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
        # 表格
        dcc.Graph(id='table-graph', style={'margin-top': '20px'})
    ], style={'width': '50%', 'padding': '20px', 'display': 'inline-block', 'vertical-align': 'top'}),
    
    # 右側區域
    html.Div([
        # 複選框
        html.H3("Projected Future Revenue by Category"),
        html.Div([
            dcc.Checklist(
                id='row-selector',
                options=[{'label': table1.iloc[i, 0], 'value': i} for i in range(5)],
                value=[0, 1, 2, 3, 4],
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            )
        ]),
        # 堆疊條形圖
        dcc.Graph(id='stacked-bar-chart', style={'margin-top': '0px'}),
        html.H3("NVIDIA Business Value Range", style={'textAlign': 'left'}),
        # NVIDIA Business Value Range 滑桿
        html.Div([
            html.Label('WACC'),
            dcc.RangeSlider(
                id='WACC-slider',
                min=min(WACC),
                max=max(WACC),
                marks={w: str(w) for w in WACC},
                value=[6, 8],
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 10px'}),
        html.Div([
            html.Label('Terminal Growth Rate'),
            dcc.Slider(
                id='g-slider',
                min=min(g),
                max=max(g),
                marks={i: str(i) for i in g},
                value=3,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 10px'}),
        # NVIDIA Business Value Range bar_chart
        dcc.Graph(id='bar-chart', style={'margin-top': '20px'})
    ], style={'width': '50%', 'padding': '20px', 'display': 'inline-block', 'vertical-align': 'top'})
], style={'display': 'flex'})

@app.callback(
    [Output('table-graph', 'figure'), Output('stacked-bar-chart', 'figure')],
    [Input('datacenter-slider', 'value'), Input('gaming-slider', 'value'), Input('low-g-slider', 'value'), Input('row-selector', 'value')]
)
def update_table(d_g, g_g, l_g, selected_rows):
    d_g = d_g / 100  
    g_g = g_g / 100
    l_g = l_g / 100
    
    # 更新特定行和列
    for year in range(6, len(table1.columns)):
        table1.iloc[0, year] = round(table1.iloc[0, year-1] * (1 + d_g), 2)
        table1.iloc[1, year] = round(table1.iloc[1, year-1] * (1 + g_g), 2)
        table1.iloc[2, year] = round(table1.iloc[2, year-1] * (1 + l_g), 2)
        table1.iloc[3, year] = round(table1.iloc[3, year-1] * (1 + l_g), 2)
        table1.iloc[4, year] = round(table1.iloc[4, year-1] * (1 + l_g), 2)
        table1.iloc[5, year] = round(table1.iloc[0:5, year].sum(), 2)
        table1.iloc[7, year] = round(table1.iloc[5, year] - table1.iloc[6, year], 2)
        table1.iloc[11, year] = round(table1.iloc[7, year] - table1.iloc[10, year], 2)
        table1.iloc[13, year] = round(table1.iloc[11, year] * (1 - table1.iloc[12, year]), 2)
        table1.iloc[15, year] = round(table1.iloc[13, year] - table1.iloc[14, year], 2)

    # 更新表格資料
    header = dict(values=table1.columns,
                  fill_color='#44642c',
                  font=dict(color='white'),
                  align='left')

    cells = dict(values=[table1[col] for col in table1.columns],
                 fill_color='#f2f2f2',
                 align='left')

    table_fig = go.Figure(data=[go.Table(header=header, cells=cells)])

    table_fig.update_layout(
        autosize=True,
        height=800,
        margin=dict(l=0, r=0, t=0, b=0),
        title_text='Key Financial Table',
    )

    # 建立堆疊條形圖
    bar_fig = go.Figure()
    for row in sorted(selected_rows):
        bar_fig.add_trace(go.Bar(
            x=table1.columns[1:],  # 顯示所有年份
            y=table1.iloc[row, 1:],
            name=table1.iloc[row, 0],
            marker=dict(color=color_map[row])  # 使用 color_map 中的顏色
        ))

    bar_fig.update_layout(
        barmode='stack',
        yaxis_title="Sales (in million USD)",
        xaxis=dict(
            title="Year",  # 添加 X 軸標籤
            tickmode='linear',  # 使用線性刻度
            dtick=1  # 設定刻度間隔
        ),
        height=400
    )
    
    return table_fig, bar_fig

@app.callback(
    Output('bar-chart', 'figure'),
    [Input('WACC-slider', 'value'), Input('g-slider', 'value')]
)
def update_bar_chart(WACC_range, g):
    FCF = [table1.iloc[15, 6], table1.iloc[15, 7], table1.iloc[15, 8]]
    wacc_low, wacc_high = WACC_range
    lo, hi = DCF(FCF, wacc_low, wacc_high, g)
    hi_sub_low = hi - lo
    
    melted.loc[8] = ['DCF', 'Hi', hi_sub_low]
    melted.loc[9] = ['DCF', 'Lo', lo]
    
    fig = px.bar(
        melted, x='value', y='Method', color='Range', orientation='h',
        color_discrete_map={
            'Hi': 'rgba(0, 0, 0, 0)',
            'Lo': '#44642c'
        }
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Value (in million USD)",
        height=400
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)