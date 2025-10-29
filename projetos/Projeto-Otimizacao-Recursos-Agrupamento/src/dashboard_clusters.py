import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import warnings
warnings.filterwarnings('ignore')

# 1. CARREGAR E PREPARAR OS DADOS
# Definição das colunas numéricas que serão usadas no gráfico
FEATURES_NUMERICAS = ['JobSatisfaction', 'PerformanceRating', 'YearsAtCompany', 
                      'TotalWorkingYears', 'YearsInCurrentRole', 'YearsSinceLastPromotion']

# Simulação da criação do DataFrame df (AGORA COM TODAS AS COLUNAS)
# NOTA: No seu ambiente real, você deve carregar o 'alocacao_colaboradores_clusters.csv'
data_simulada = {
    'Cluster': [0, 0, 0, 1, 1, 1, 2, 2, 2],
    'JobSatisfaction': [2.69, 2.68, 2.70, 2.74, 2.75, 2.73, 2.74, 2.73, 2.75],
    'PerformanceRating': [3.08, 3.07, 3.09, 3.00, 3.01, 3.00, 4.00, 3.99, 4.00],
    'YearsAtCompany': [15.11, 15.00, 15.20, 4.37, 4.40, 4.30, 5.47, 5.50, 5.45],
    'TotalWorkingYears': [18.77, 18.70, 18.80, 8.85, 8.90, 8.80, 9.81, 9.70, 9.90],
    
    # NOVAS COLUNAS INCLUÍDAS PARA CORRIGIR O KEYERROR:
    'YearsInCurrentRole': [10.0, 10.1, 9.9, 2.5, 2.6, 2.4, 3.0, 3.1, 2.9], # C0: alto; C1/C2: baixo
    'YearsSinceLastPromotion': [6.3, 6.2, 6.4, 0.8, 0.9, 0.7, 1.4, 1.3, 1.5], # C0: alto; C1/C2: baixo

    'Perfil_Recomendado': ['Veteranos Estagnados', 'Veteranos Estagnados', 'Veteranos Estagnados', 
                            'Novos Colaboradores', 'Novos Colaboradores', 'Novos Colaboradores', 
                            'Alta Performance', 'Alta Performance', 'Alta Performance']
}

df = pd.DataFrame(data_simulada)
df['Cluster'] = df['Cluster'].astype(str) # Garante que o Cluster é tratado como string

# Cálculo da média corrigido: AGORA FUNCIONARÁ, pois todas as colunas estão no DF
df_medias = df.groupby('Cluster')[FEATURES_NUMERICAS].mean().reset_index()

# Mapeamento do Perfil para exibição
df_medias['Perfil'] = df_medias['Cluster'].map({
    '0': 'Veteranos Estagnados (341)',
    '1': 'Novos Colaboradores (931)',
    '2': 'Alta Performance (198)'
})

# Lista de métricas numéricas para o dropdown e o gráfico
metricas = df_medias.columns.drop(['Cluster', 'Perfil']) 


# 2. INICIALIZAÇÃO DO DASHBOARD
app = Dash(__name__)


# 3. LAYOUT DO DASHBOARD
app.layout = html.Div(style={'backgroundColor': '#f8f9fa', 'padding': '20px'}, children=[
    html.H1(
        children='Dashboard de Análise de Clusters de Colaboradores',
        style={'textAlign': 'center', 'color': '#007bff', 'marginBottom': '30px'}
    ),

    # CONTROLES DE FILTRO
    html.Div(style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,.1)'}, children=[
        html.Label('Selecione o Cluster para Visualizar as Métricas:', style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
        
        dcc.Dropdown(
            id='cluster-dropdown',
            options=[
                {'label': f'Cluster {i} - {df_medias[df_medias["Cluster"] == str(i)]["Perfil"].iloc[0].split(" (")[0]}', 'value': str(i)} 
                for i in df_medias['Cluster'].unique()
            ],
            value='2', # Valor inicial: Alta Performance (Cluster 2)
            clearable=False
        ),
    ]),

    html.Hr(style={'margin': '40px 0'}),

    # GRÁFICO DE BARRAS (RESULTADO FILTRADO)
    html.Div(id='grafico-medias-cluster', style={'maxWidth': '1000px', 'margin': '0 auto'})
])


# 4. CALLBACKS (Funções de interatividade com os filtros)
@app.callback(
    Output('grafico-medias-cluster', 'children'),
    [Input('cluster-dropdown', 'value')]
)
def update_graph(selected_cluster):
    # Filtrar o DataFrame para o cluster selecionado
    df_filtrado = df_medias[df_medias['Cluster'] == selected_cluster]
    
    # Transformar o formato wide para long (ideal para Plotly)
    df_long = df_filtrado.melt(
        id_vars=['Cluster', 'Perfil'], 
        value_vars=metricas,
        var_name='Métrica', 
        value_name='Média'
    )

    # Definir o título com o nome do perfil
    perfil_nome = df_filtrado['Perfil'].iloc[0].split(" (")[0]
    
    # Criar o gráfico interativo de barras
    fig = px.bar(
        df_long, 
        x='Métrica', 
        y='Média', 
        color='Métrica',
        text='Média', # Mostrar o valor da barra
        title=f'Médias das Métricas para o Perfil: {perfil_nome}',
        height=500
    )
    
    # Ajustes finos no layout do gráfico
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        uniformtext_minsize=8, 
        uniformtext_mode='hide',
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=20, r=20)
    )

    return dcc.Graph(figure=fig)


# 5. EXECUÇÃO DO APLICATIVO
if __name__ == '__main__':
    app.run(debug=True)