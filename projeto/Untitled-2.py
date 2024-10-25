# %%
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("../data/train.csv")
teste = pd.read_csv("../data/test.csv")
df.duplicated().sum()

# %%

df.describe().T

# %%
carac_inicial = list(teste.columns)
carac_inicial

# %%
for coluna in df.columns:
    print(f"{coluna} tem {df[coluna].nunique()} valores unicos")

# %%
poucos_valores = [coluna for coluna in df.columns if df[coluna].nunique() <= 8]

muitos_valores = [coluna for coluna in df.columns if df[coluna].nunique() >= 9]
len(muitos_valores)

# %%
plt.figure(figsize=(18, 24))
plotnumber = 1

for coluna in poucos_valores:
    if plotnumber <= len(poucos_valores):
        ax = plt.subplot(4, 3, plotnumber)
        sns.countplot(x=df[coluna], data=df, palette='pastel')
        
        # Add labels to each bar in the plot
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 3, f'{int(p.get_height())}', ha="center")
        
        plt.xlabel(coluna)
        # plt.xticks(rotation=45)
        plt.xlabel(coluna)
        
    plotnumber += 1

plt.suptitle('Distribution of Categorical Variables', fontsize=40, y=1)
plt.tight_layout()
plt.show()

# %%
from sklearn.preprocessing import LabelEncoder

categorias = ['dropout', 'enrolled', 'graduate']
encoder = LabelEncoder()

# converter o target para valores numericos
df['Target'] = encoder.fit_transform(df['Target'])

# %%
matriz_correlação = df.corr()

# Aplica o filtro para manter apenas as correlações > 0.5 ou < -0.5
filtro_correlação = matriz_correlação[(matriz_correlação > 0.5) | (matriz_correlação < -0.5)]

# Cria o mapa de calor com as correlações filtradas
plt.figure(figsize=(21, 18))
sns.heatmap(filtro_correlação, annot=True, cmap='coolwarm', fmt='.1f', linewidths=2, linecolor='lightgrey', vmin=-1, vmax=1)
plt.suptitle('Mapa de calor', fontsize=30, y=1)
plt.show()

# %%
matriz_correlação

# %%
X_train = df[carac_inicial]
Y_train = df["Target"]
X_test = teste[carac_inicial]

# %% [markdown]
# # Criar uma função para fazer as validações cruzadas
# 

# %%
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# criar uma função para

def validação_cruzada(modelo, X_train, Y_train, parametros, n_splits=10): 

        cv = KFold( n_splits=n_splits, shuffle=True, random_state=0)
        score_validação = []

        for fold, (treinamento, validação) in enumerate(cv.split(X_train)):
                
                X_fold_train = X_train.iloc[treinamento]
                Y_fold_train = Y_train.iloc[treinamento]
                X_val = X_train.iloc[validação]
                y_val = Y_train.iloc[validação]

                classificador = modelo(**parametros)
                classificador.fit(X_fold_train, Y_fold_train)

                y_previ_treino = classificador.predict(X_fold_train)
                y_previ_teste = classificador.predict(X_val)
                Acura_score = accuracy_score(Y_fold_train, y_previ_treino)
                Acura_validação = accuracy_score(y_val, y_previ_teste)

                print(f"Fold: {fold}, Train Accuracy: {Acura_score:.5f}, Val Accuracy: {Acura_validação:.5f}")
                print("-" * 50)

                score_validação.append(Acura_validação)

        Media_acuracia_validation = np.mean(score_validação)
        print("Media da acuraica da validação:", Media_acuracia_validation)
        return classificador,Media_acuracia_validation


# %%
from sklearn.ensemble import RandomForestClassifier

print('Random Forest Cross-Validation Results:\n')
rf_model, rf_mean_accuracy = validação_cruzada(RandomForestClassifier, X_train, Y_train, parametros={})
print(f"\nFinal Mean Validation Accuracy: {rf_mean_accuracy:.5f}")


