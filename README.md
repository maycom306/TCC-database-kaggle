# Análise da Evasão Escolar no IFRN: Uma Abordagem Baseada em Machine Learning

## Descrição

Este projeto visa analisar os fatores que contribuem para a evasão escolar no Instituto Federal do Rio Grande do Norte (IFRN) utilizando técnicas de machine learning. O objetivo principal é desenvolver um modelo preditivo que identifique alunos em risco de evasão, possibilitando a implementação de intervenções preventivas.

O dataset utilizado para as fases introdutórias do projeto é público e pode ser acessado em [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e6/data).


## Tabela de Conteúdos

- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Dataset](#dataset)
- [Pré-processamento dos Dados](#pré-processamento-dos-dados)
- [Análise Exploratória](#análise-exploratória)
- [Modelagem](#modelagem)
- [Resultados](#resultados)
- [Conclusão](#conclusão)
- [Como Executar o Projeto](#como-executar-o-projeto)
- [Contribuições](#contribuições)
- [Licença](#licença)

## Tecnologias Utilizadas

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset

## Dataset

O dataset utilizado para esta análise é público e pode ser acessado em [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e6/data). O conjunto de dados contém informações sobre alunos, incluindo as seguintes variáveis:

- **Idade**
- **Gênero**
- **Nota**
- **Frequência**
- **Estado Civil**
- **Modo de Inscrição**
- **Ordem de Inscrição**
- **Curso**
- **Atendimento diurno/noturno**
- **Qualificação Anterior**
- **Nota da Qualificação Anterior**
- **Nacionalidade**
- **Qualificação da Mãe**
- **Qualificação do Pai**
- **Profissão da Mãe**
- **Profissão do Pai**
- **Nota de Admissão**
- **Deslocado**
- **Necessidades Educacionais Especiais**
- **Devedor**
- **Taxas de Matrícula em Dia**
- **Bolsa de Estudo**
- **Idade na Matrícula**
- **Internacional**
- **Unidades Curriculares 1º Semestre (credenciadas)**
- **Unidades Curriculares 1º Semestre (matriculadas)**
- **Unidades Curriculares 1º Semestre (avaliações)**
- **Unidades Curriculares 1º Semestre (aprovadas)**
- **Unidades Curriculares 1º Semestre (nota)**
- **Unidades Curriculares 1º Semestre (sem avaliações)**
- **Unidades Curriculares 2º Semestre (credenciadas)**
- **Unidades Curriculares 2º Semestre (matriculadas)**
- **Unidades Curriculares 2º Semestre (avaliações)**
- **Unidades Curriculares 2º Semestre (aprovadas)**
- **Unidades Curriculares 2º Semestre (nota)**
- **Unidades Curriculares 2º Semestre (sem avaliações)**
- **Taxa de Desemprego**
- **Taxa de Inflação**
- **PIB**
- **Target**


## Pré-processamento dos Dados

No início do projeto, os dados foram carregados e analisados para verificar a presença de valores duplicados e estatísticas descritivas. As variáveis do conjunto de dados foram avaliadas em relação ao número de valores únicos, permitindo a identificação de variáveis categóricas e numéricas. Gráficos de contagem foram gerados para visualizar a distribuição de variáveis categóricas.


## Como Executar o Projeto

Para executar este projeto em sua máquina local, siga as etapas abaixo:

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/maycom306/TCC-database-kaggle.git
2. **Navegue até o diretório:**
   ```bash
   cd TCC-database-kaggle
3. **Instale as dependencias**
    ```bash
    pip install -r requirements.txt
