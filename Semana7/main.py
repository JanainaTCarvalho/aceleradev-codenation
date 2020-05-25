#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[11]:


# Algumas configurações para o matplotlib.
'''%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()'''


# In[59]:


countries = pd.read_csv("countries.csv")


# In[60]:


countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[68]:


#Lendo arquivo novamente, trocando o separador decimal por ponto
countries = pd.read_csv("countries.csv", decimal = ",")

#Trocando nomes das colunas
new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]
countries.columns = new_column_names

#Removendo espaços em branco no início e final da string nas variáveis Country e Region
countries['Country'] = countries['Country'].str.strip()
countries['Region'] = countries['Region'].str.strip()
countries.head(5)


# In[32]:


#Dataframe auxiliar na análise
df_aux = pd.DataFrame({'Type': countries.dtypes,
                      'Missing': countries.isna().sum(),
                      'Size': countries.shape[0],
                       'Unique': countries.nunique()
                     })
df_aux['Missing_%']= df_aux.Missing/df_aux.Size * 100
df_aux


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[14]:


def q1():
    # Convertendo variável Region em um lista de itens únicos
    region_list = (countries['Region'].unique()).tolist()
    
    #Ordenando a lista em ordem alfabética
    region_list = sorted(region_list)
    
    return region_list
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[34]:


#Importando o pacote
from sklearn.preprocessing import KBinsDiscretizer


# In[35]:


def q2():
    # Criando variável discretizer com 10 intervalos, encode ordinal e estratégia quantile
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    
    #Discretizando a variável Pop_density
    discretizer.fit(countries[["Pop_density"]])
    
    #Retorno: quantos países se encontram acima do percentil 90    
    return len(countries.query('Pop_density > @discretizer.bin_edges_[0][9]'))
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[37]:


#Importando o pacote
from sklearn.preprocessing import OneHotEncoder


# In[38]:


def q3():
    #Hot enconding para np.int
    one_hot_encoder = OneHotEncoder(sparse=False)
    
    #Codificando as variáveis
    region_climate_encoded = one_hot_encoder.fit(countries[['Region', 'Climate']].fillna('0').astype('str'))
    
    #Pegando as novas features geradas
    new_attributes = region_climate_encoded.get_feature_names()
    
    return len(new_attributes)
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[39]:


#Importando pacotes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# In[40]:


#Criando pipeline
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# In[41]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

#Tranformando test_country em Dataframe
test_country = pd.DataFrame([test_country], columns = countries.columns)
test_country


# In[46]:


#Selecionando colunas numéricas
num_countries = countries.select_dtypes(include=[np.number])
num_test_country = test_country.select_dtypes(include=[np.number])

#Aplicando o pipeline nas colunas numéricas de Countries
pipeline.fit(num_countries)


# In[47]:


def q4():
    #Transformando test_country
    transformed_test_country = pipeline.transform(num_test_country)

    #Retorno: item Arable no formato solicitado
    return float(np.round(transformed_test_country[:, num_countries.columns.get_loc('Arable')], 3))
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[74]:


def q5():
    #Separando o valores de quartis e iqr
    q1 = countries.Net_migration.quantile(0.25)
    q3 = countries.Net_migration.quantile(0.75)
    iqr = q3 - q1

    #Setando intervalo de valor não considerado outlier
    non_outlier_interval = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
    
    #Coletando os outliers
    lower_outliers = countries.Net_migration[(countries.Net_migration < non_outlier_interval[0])]
    upper_outliers = countries.Net_migration[(countries.Net_migration > non_outlier_interval[1])]
    
    #Analisando resultados: vendo os avalores encontrados como outliers
    print(lower_outliers)
    print(upper_outliers)
    '''Estamos diante da taxa de migração do país. Desta forma, os valores encontrados como outliers não são valores que
    estão fora do que se espera de "normal" para essa taxa'''
    
    #verificando a porcentagem de valores encontrados como outliers
    print(((len(lower_outliers)+len(upper_outliers))*100)/len(countries.Net_migration))
    '''22% dos dados dessa variável são considerados outliers por essa lógica, o que é uma taxa bem alta de itens.
    Assim, caso sejam removidos, podem comprometer nossa análise. Entretanto, como já visto que os valores encontrados
    como outliers fazem sentido para a taxa apresentada, não devemos removê-los.
    Portanto:'''
    answer = False
    
    return (len(lower_outliers), len(upper_outliers), answer)
q5()


# In[80]:


#Analisando graficamente os outliers para confirmção da análise feita
sns.boxplot(countries.Net_migration);


# In[81]:


sns.distplot(countries.Net_migration);


# ## Questão 6
# 
# Para as questões 6 e 7 utilize a biblioteca fetch_20newsgroups de datasets de test do sklearn
# 
# Considere carregar as seguintes categorias e o dataset newsgroups:
# 
#     categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
#     newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
#     
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[84]:


#Importando o pacote
from sklearn.feature_extraction.text import CountVectorizer

#Importando a biblioteca e carregando as categorias e newsgroups
from sklearn.datasets import fetch_20newsgroups

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[85]:


def q6():
    #Setando matriz dos grupos selecionados para contagem
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
    
    #Retornando: quantidade de vezes que palavra phone aparece
    return int(newsgroups_counts[:, count_vectorizer.vocabulary_['phone']].sum())
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[86]:


#Importando o pacote, para analisar a relevância do termo 'phone' no documento
from sklearn.feature_extraction.text import TfidfVectorizer


# In[88]:


def q7():
    #Setando matriz tf-idf dos grupos selecionados
    tfidf_vectorizer = TfidfVectorizer()
    newsgroups_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroups.data)
    
    #Retornando tf-idf da palavra 'phone': 
    return float(round(newsgroups_tfidf_vectorized[:, tfidf_vectorizer.vocabulary_['phone']].sum(), 3))
q7()


# In[ ]:




