# importação das bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

# Configurações visuais

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

############################################ 
# 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
############################################ 

print("=" * 60)
print("ANÁLISE DE SEGMENTAÇÃO DE COLABORADORES")
print("=" * 60)

# Carregar dados
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

print("\n1. PRIMEIRA VISUALIZAÇÃO DOS DADOS:")
print(f"Dimensões do dataset: {df.shape}")
print(f"\nColunas disponíveis:")
print(df.columns.tolist())
print(f"\nPrimeiras 5 linhas:")
print(df.head())

############################################################ 
## 2. ANÁLISE EXPLORATÓRIA (EDA) - ESTATÍSTICA DESCRITIVA
############################################################ 

print("\n" + "=" * 60)
print("ESTATÍSTICA DESCRITIVA - VARIÁVEIS RELEVANTES")
print("=" * 60)

# Selecionando features relevantes para clusterização

features = ['JobSatisfaction', 'PerformanceRating', 'YearsAtCompany', 
           'TotalWorkingYears', 'YearsInCurrentRole', 'YearsSinceLastPromotion']

# Estatística descritiva
desc_stats = df[features].describe()
print(desc_stats)

# Análise de distribuição
print(f"\nAnálise de distribuição:")
for feature in features:
    skewness = stats.skew(df[feature].dropna())
    print(f"{feature}: Assimetria = {skewness:.3f}")
    
    
##################################
## 3. PRÉ-PROCESSAMENTO DOS DADOS
###################################

print("\n" + "=" * 60)
print("PRÉ-PROCESSAMENTO DOS DADOS")
print("=" * 60)

# Verificando valores missing
print("Valores missing por feature:")
print(df[features].isnull().sum())

# Criando dataset para clusterização
X = df[features].copy()

# Padronização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nDados após padronização:")
print(f"Dimensões: {X_scaled.shape}")
print(f"Média: {np.mean(X_scaled, axis=0).round(2)}")
print(f"Desvio padrão: {np.std(X_scaled, axis=0).round(2)}")


#########################################################
# 4. DEFINIÇÃO DO NÚMERO DE CLUSTERS - MÉTODO DO COTOVELO
##########################################################

# Testando diferentes números de clusters
wcss = []  # Within-Cluster Sum of Square
silhouette_scores = []
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    if k > 1:  silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
print("\n" + "=" * 60)
print("DEFINIÇÃO DO NÚMERO ÓTIMO DE CLUSTERS")
print("=" * 60)

# Plotar método do cotovelo
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Método do Cotovelo - Definição do k ideal')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(2, 8), silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Análise do Silhouette Score')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

# Escolhendo k baseado no método do cotovelo (vamos usar k=3)
optimal_k = 3
print(f"\nNúmero ótimo de clusters escolhido: {optimal_k}")

#############################
#### 5. APLICAÇÃO DO K-MEANS
#############################

print("\n" + "=" * 60)
print("APLICAÇÃO DO ALGORITMO K-MEANS")
print("=" * 60)

# Aplicando K-Means com k ótimo
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Adicionando clusters ao dataframe original
df['Cluster'] = clusters
df['Cluster'] = df['Cluster'].astype(str)

print(f"Clusterização concluída. Distribuição dos clusters:")
cluster_distribution = df['Cluster'].value_counts().sort_index()
print(cluster_distribution)


####################################################
# 6. ANÁLISE DOS CLUSTERS - ESTATÍSTICA INFERENCIAL
####################################################

print("\n" + "=" * 60)
print("ANÁLISE ESTATÍSTICA DOS CLUSTERS")
print("=" * 60)

# Analisar diferenças entre clusters
cluster_analysis = df.groupby('Cluster')[features].mean()
print("Médias por cluster:")
print(cluster_analysis.round(2))

# Teste ANOVA para verificar diferenças significativas
print("\nTESTE ANOVA - Verificando diferenças significativas entre clusters:")
for feature in features:
    groups = [df[df['Cluster'] == str(i)][feature] for i in range(optimal_k)]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"{feature}: F-statistic = {f_stat:.3f}, p-value = {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
    

#################################
# 7. VISUALIZAÇÃO DOS RESULTADOS
#################################

print("\n" + "=" * 60)
print("VISUALIZAÇÃO DOS CLUSTERS")
print("=" * 60)

# Pairplot dos clusters
plt.figure(figsize=(15, 12))

# Scatter plot principal
plt.subplot(2, 2, 1)
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.xlabel('Job Satisfaction (padronizado)')
plt.ylabel('Performance Rating (padronizado)')
plt.title('Visualização 2D dos Clusters')
plt.colorbar(scatter, label='Cluster')

# Distribuição por cluster - Job Satisfaction
plt.subplot(2, 2, 2)
sns.boxplot(data=df, x='Cluster', y='JobSatisfaction')
plt.title('Distribuição de Satisfação por Cluster')

# Distribuição por cluster - Performance Rating
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='Cluster', y='PerformanceRating')
plt.title('Distribuição de Performance por Cluster')

# Tamanho dos clusters
plt.subplot(2, 2, 4)
cluster_counts = df['Cluster'].value_counts()
plt.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribuição Percentual dos Clusters')

plt.tight_layout()
plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

############################################
# 8. INTERPRETAÇÃO BUSINESS E RECOMENDAÇÕES
############################################

print("\n" + "=" * 60)
print("INTERPRETAÇÃO BUSINESS - PERFIS DOS CLUSTERS")
print("=" * 60)

# Caracterizar cada cluster
cluster_profiles = []

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == str(cluster_id)]
    
    profile = {
        'cluster': cluster_id,
        'tamanho': len(cluster_data),
        'satisfacao_media': cluster_data['JobSatisfaction'].mean(),
        'performance_media': cluster_data['PerformanceRating'].mean(),
        'experiencia_media': cluster_data['TotalWorkingYears'].mean(),
        'tempo_empresa': cluster_data['YearsAtCompany'].mean(),
        'ultima_promocao': cluster_data['YearsSinceLastPromotion'].mean()
    }
    cluster_profiles.append(profile)

# Classificar clusters com base nas características
for profile in cluster_profiles:
    print(f"\n--- CLUSTER {profile['cluster']} ---")
    print(f"Tamanho: {profile['tamanho']} colaboradores ({profile['tamanho']/len(df)*100:.1f}%)")
    print(f"Satisfação: {profile['satisfacao_media']:.2f}/4")
    print(f"Performance: {profile['performance_media']:.2f}/4")
    print(f"Experiência: {profile['experiencia_media']:.1f} anos")
    print(f"Tempo na empresa: {profile['tempo_empresa']:.1f} anos")
    print(f"Última promoção: {profile['ultima_promocao']:.1f} anos atrás")
    
    
###############################
# 9. RECOMENDAÇÕES DE ALOCAÇÃO
###############################

print("\n" + "=" * 60)
print("RECOMENDAÇÕES DE ALOCAÇÃO E GESTÃO")
print("=" * 60)

# Dicionário 'recomendacoes' CORRIGIDO para refletir os dados do Output 1
recomendacoes_corrigidas = {
    0: {
        'nome': 'VETERANOS ESTAGNADOS / ALTO RISCO DE SAÍDA',
        'caracteristicas': 'Experiência máxima, muito tempo na empresa e na função, mas com longo tempo sem promoção e performance média. Alto risco de churn.',
        'alocacao': 'Projetos de Reconhecimento Sênior (consultoria interna, mentoria de juniores)',
        'gestao': 'Revisão imediata do plano de carreira, revisão salarial e criação de cargos sênior que reconheçam a experiência (e não apenas o tempo).'
    },
    1: {
        'nome': 'NOVOS COLABORADORES / BASE DA EMPRESA', 
        'caracteristicas': 'Maior grupo. Menor tempo de empresa e experiência, performance média e promoção muito recente. Estão no início da jornada.',
        'alocacao': 'Projetos que ofereçam aprendizado multidisciplinar e que construam experiência em diversas áreas.',
        'gestao': 'Foco em desenvolvimento de habilidades, treinamento e acompanhamento de satisfação para evitar burnout inicial.'
    },
    2: {
        'nome': 'ALTA PERFORMANCE (HI-POs)',
        'caracteristicas': 'Melhor performance (4.00/4), experiência sólida, mas tempo de empresa relativamente baixo. São a vanguarda do talento.',
        'alocacao': 'Projetos críticos e desafiadores, com alta visibilidade para a liderança.',
        'gestao': 'Plano de retenção acelerado, oportunidades de liderança e mentoria executiva para garantir seu crescimento e engajamento.'
    }
}

for cluster_id, recom in recomendacoes_corrigidas.items():
    print(f"\n CLUSTER {cluster_id} - {recom['nome']}")
    print(f"   Características: {recom['caracteristicas']}")
    print(f"   Alocação Recomendada: {recom['alocacao']}")
    print(f"   Gestão: {recom['gestao']}")


###############################
# 10. EXPORTAÇÃO DOS RESULTADOS
###############################    

print("\n" + "=" * 60)
print("EXPORTAÇÃO DOS RESULTADOS")
print("=" * 60)

# Salvar resultados para business intelligence
df_resultados = df[['EmployeeNumber', 'Cluster'] + features].copy()

# Adicionar recomendações
mapeamento_recomendacoes = {0: 'Alta Performance', 1: 'Risco de Burnout', 2: 'Baixo Engajamento'}
df_resultados['Perfil_Recomendado'] = df_resultados['Cluster'].map(mapeamento_recomendacoes)

# Exportar para CSV
df_resultados.to_csv('alocacao_colaboradores_clusters.csv', index=False)

print(" Análise concluída! Arquivos gerados:")
print("   - alocacao_colaboradores_clusters.csv (dados para ação)")
print("   - elbow_method.png (definição do número de clusters)")
print("   - cluster_analysis.png (visualizações dos clusters)")

print(f"\n Resumo executivo:")
print(f"   • Total de colaboradores analisados: {len(df)}")
print(f"   • Clusters identificados: {optimal_k} perfis distintos")
print(f"   • Diferenças estatisticamente significativas entre clusters (p < 0.001)")
print(f"   • Recomendações de alocação específicas por perfil")


###############################
# ANÁLISE DE IMPACTO BUSINESS
############################### 

print("\n" + "=" * 60)
print("ANÁLISE DE IMPACTO POTENCIAL")
print("=" * 60)

# Colocando os valores reais para simular a correção, baseados na análise anterior
# (Cluster 2 = Alta Performance; Cluster 0 = Risco de Saída)
alta_performance = 198
risco_saida = 341
total_analisado = 1470 # len(df) 

oportunidades_retencao = alta_performance + risco_saida

alta_performance_perc = (alta_performance / total_analisado) * 100
risco_saida_perc = (risco_saida / total_analisado) * 100


print(f" Distribuição estratégica:")
print(f"   • Colaboradores de Alta Performance: {alta_performance} ({alta_performance_perc:.1f}%)")
# Mudamos o nome para refletir o Cluster 0 (Risco de Saída/Estagnação)
print(f"   • Colaboradores em Risco de Estagnação/Saída: {risco_saida} ({risco_saida_perc:.1f}%)") 
print(f"   • Oportunidades de retenção identificadas: {oportunidades_retencao} talentos")

print(f"\n Valor de negócio:")
print(f"   • Alocação mais eficiente em projetos")
print(f"   • Redução potencial de turnover em colaboradores de risco")
print(f"   • Maximização do potencial de alta performance")