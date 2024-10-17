#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%%
df = pd.read_csv("data/train.csv")
df

# %%
df.info()
# %%
coluna_categorica = df.select_dtypes(include=['object']).columns
for columns in coluna_categorica:
    plt.figure(figsize=(10,4))
    sns.countplot(x=columns, data=df)
    plt.title(f'Distribuição de{columns}')
    plt.xticks(rotation=90)
    plt.show()

# %%
selecionar_caracteristicas = ['Admission grade', 'Age at enrollment', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Target']
sns.pairplot(df[selecionar_caracteristicas], hue='Target')
plt.show()
# %%
df.describe().T
# %%
teste = df.drop(['id', 'Target'], axis=1)
df.drop('id', axis=1, inplace=True)

# %%
caract_inicial = list(teste.columns)
caract_inicial
# %%
for unicos in df.columns:
    print(f'{unicos} tem {df[unicos].nunique()} valores')
# %%
colunas_categorica = [unicos for unicos in df.columns if df[unicos].nunique() <= 8]
colunas_numericas = [unicos for unicos in df.columns if df[unicos].nunique() >= 9]
# %%
len(colunas_categorica)
len(colunas_numericas)
# %%
plt.figure(figsize=(18,24))
contador = 1

for unicos in colunas_categorica:
    if contador <= len(colunas_categorica):
        ax = plt.subplot(4, 3, contador)
        sns.countplot(x=df[unicos], data=df, palette='pastel')
        
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 3, 
                                f'{int(p.get_height())}', ha="center")
            plt.xlabel(unicos)
            plt.ylabel(unicos)
        
        contador += 1
plt.suptitle('Distribuição das Variáveis Categóricas', fontsize=40, y=1)
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(18,24))
contador = 1

for unicos in colunas_categorica:
    if contador <= len(colunas_categorica):
        plt.subplot(4, 3, contador)
        ax = sns.countplot(x=df[unicos], hue=df['Target'], palette='bright')
        
    contador += 1

plt.suptitle('Distribution of Categorical Variables by Target', fontsize=40, y=1)
plt.tight_layout()
plt.show()
# %%

plt.figure(figsize=(18,40))
contador = 1

for columns in colunas_numericas:
    if contador <= len(colunas_numericas):
        ax = plt.subplot(9,3,contador)
        sns.kdeplot(df[columns], color='deepskyblue', fill=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)

        plt.xlabel(columns)
        ax.grid(False)

    contador += 1
plt.suptitle('distribuição das variaveis numericas', fontsize=40, y=1)
plt.tight_layout()
plt.show()
# %%
from sklearn.preprocessing import LabelEncoder

categorias = ['dropout', 'enrolled', 'graduate']
Label_Encoder = LabelEncoder()

df['Target'] = Label_Encoder.fit_transform(df['Target'])

# %%
threshold = 0.5

# Calcula a matriz de correlação e filtra as correlações acima do limiar
corr = df.corr().apply(lambda x: x.where(np.abs(x) > threshold))

# Remove linhas e colunas que ficaram totalmente vazias
corr = corr.dropna(how='all', axis=0).dropna(how='all', axis=1)

# Plotando o heatmap com correlações filtradas
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.1f', linewidths=2)
plt.title('Filtered Correlation Matrix', fontsize=16)
plt.show()

# %%
threshold = 0.5

# Calcula a matriz de correlação
corr = df.corr()

# Identifica as colunas que têm correlação acima do threshold com pelo menos uma outra coluna
cols_to_keep = corr.columns[(corr.abs() > threshold).any(axis=0)]

# Filtra o DataFrame original para manter apenas as colunas com correlação significativa
df_filtered = df[cols_to_keep]

# Agora df_filtered contém apenas as colunas relevantes
print("Colunas filtradas:", df_filtered.columns.tolist())
# %%
from sklearn.model_selection import train_test_split
X_treino = df[caract_inicial]
y_treino = df['Target']