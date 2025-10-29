# Otimização de Alocação de Recursos com Clusterização

## Sobre o Projeto
Análise de segmentação de colaboradores utilizando algoritmos de clusterização para otimizar a alocação estratégica em projetos. Este projeto demonstra como técnicas de Machine Learning não-supervisionado podem ser aplicadas para gestão de recursos humanos.

## Objetivo de Negócio
Automatizar e otimizar o processo de alocação de colaboradores em projetos, segmentando-os com base em:
- Satisfação no trabalho
- Performance
- Experiência profissional
- Tempo na empresa
- Desenvolvimento profissional

## Stack Tecnológico
- **Python 3.8+**
- **Bibliotecas:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SciPy
- **Algoritmo:** K-Means Clustering
- **Análise Estatística:** Estatística Descritiva, Teste ANOVA

## Dataset Utilizado
**Nome**: IBM HR Analytics Employee Attrition & Performance - [Download link](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

**Por que este dataset?**
- Contém métricas reais de desempenho de funcionários
- Tem features similares às que você analisava (satisfação, avaliação, projetos)
- Publico e disponível no Kaggle, coontendo mais de 267 mil downloads e 1,76 milhões de views.

## Estrutura do Projeto

```markdown
Projeto-Otimizacao-Recursos-Agrupamento/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv # Dados brutos
│
├── notebooks/
│   └── 01_EDA_e_Modelagem_Inicial.ipynb      # Notebooks de Análise Exploratória e Modelagem
│
├── src/
│   └── 01_modelagem_clustering.py             # Script principal para Modelagem e Geração de Outputs
│   └── dashboard_clusters.py                  # Código da aplicação do Dashboard Interativo (Plotly Dash)
│
├── outputs/
│   ├── alocacao_colaboradores_clusters.csv    # Resultados: DataFrame com a alocação final de Cluster ID
│   ├── elbow_method.png                     # Visualização: Validação do número de Clusters (Método do Cotovelo)
│   ├── cluster_analysis.png                 # Visualização: Gráfico de comparação dos perfis de Cluster
│   └── dashboards/
│       └── dashboard_clusters.pdf             # Exportação estática do Dashboard em PDF
│
├── .gitignore                               # Arquivo para ignorar caches, ambientes virtuais e arquivos temporários
├── README.md                                # Documentação detalhada do projeto, metodologia e conclusões
└── requirements.txt                         # Lista de dependências Python para o projeto
```


## Metodologia

### 1. Análise Exploratória (EDA)
- Estatística descritiva das variáveis relevantes
- Análise de distribuição e assimetria
- Identificação de padrões iniciais

### 2. Pré-processamento
- Padronização dos dados (StandardScaler)
- Tratamento de valores missing
- Seleção de features para clusterização

### 3. Definição do Número de Clusters
- Método do Cotovelo (Elbow Method)
- Análise do Silhouette Score
- Definição de k=3 clusters

### 4. Clusterização com K-Means
- Aplicação do algoritmo K-Means
- Validação estatística das diferenças entre clusters
- Análise de significância (Teste ANOVA)

### 5. Interpretação Business
- Caracterização de cada cluster
- Recomendações de alocação específicas
- Estratégias de gestão por perfil

## Resultados - Perfis Identificados

###  CLUSTER 0 - VETERANOS ESTAGNADOS / ALTO RISCO DE SAÍDA
- **23.2% dos colaboradores**
- Performance: 3.08 e Satisfação: 2.69
- **Alocação Recomendada:** Projetos de Reconhecimento Sênior (consultoria interna, mentoria de juniores)

### CLUSTER 1 - NOVOS COLABORADORES / BASE DA EMPRESA
- **63.3%% dos colaboradores**
- Performance: 3.00 e Satisfação: 2.74
- **Alocação Recomendada:** Projetos que ofereçam aprendizado multidisciplinar e que construam experiência em diversas áreas.

###  CLUSTER 2 - ALTA PERFORMANCE (HI-POs)
- **13.5% dos colaboradores** 
- Aalta performance (4.0/4) e satisfação (2.74/4)
- **Alocação Recomendada:** Projetos críticos e desafiadores, com alta visibilidade para a liderança.

## Impacto Business

### Distribuição estratégica:

- Colaboradores de Alta Performance: 198 (13.5%)
- Colaboradores em Risco de Estagnação/Saída: 341 (23.2%)
- Oportunidades de retenção identificadas: 539 talentos

### Valor Gerado

1. Alocação mais eficiente em projetos
2. Redução potencial de turnover em colaboradores de risco
3. Maximização do potencial de alta performance


## Como Executar

1. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

2. **Executar análise:**

```python
python src/employee_clustering_analysis.py
```

3. **Ver resultados:**

- `outputs/alocacao_colaboradores_clusters.csv` - Dados para ação
- `outputs/cluster_analysis.png - Visualizações dos clusters

4. **Próximos passos::**

- Implementar sistema de recomendação em tempo real
- Integrar com API de gestão de projetos
- Desenvolver dashboard interativo
- Adicionar análise de turnover prediction

---

*Desenvolvido por* **Ivan Ajala**
*Projeto demonstrando aplicação de Data Science para otimização de gestão de recursos*

