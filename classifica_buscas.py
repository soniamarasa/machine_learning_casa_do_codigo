from sklearn.naive_bayes import MultinomialNB
from collections import Counter

import pandas as pd

df = pd.read_csv("buscas.csv")  # data frame

# print(df)

X_df = df[["home", "busca", "logado"]]
Y_df = df["comprou"]

Xdummies_df = pd.get_dummies(X_df).astype(
    int
)  # precisa do .astype(int) para converter para int
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

# a eficacia do algoritmo que chuta tudo sim ou tudo nao
# binario (comprou 1 ou 0):
# acerto_de_um = len(Y[Y==1])
# acerto_de_zero = len(Y[Y==0])
acerto_base = max(Counter(Y).values())
# acerto_de_um = list(Y).count('sim')
# acerto_de_zero = list(Y).count('nao')
taxa_de_acerto_base = 100.0 * acerto_base / len(Y)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

# print(Y)

porcentagem_de_treino = 0.9

tamanho_de_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_teste = len(Y) - tamanho_de_treino
treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]

# CRIAR MODELO

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

# binario: diferencas = resultado - teste_marcacoes
# acertos = [d for d in diferencas if d == 0]
# total_de_acertos = len(acertos)
# com sim/nao a comparacao e direta, porque nao da para subtrair texto
acertos = resultado == teste_marcacoes
total_de_acertos = sum(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos


# print(resultado)
print("Taxa de acerto do algoritmo: %f" % taxa_de_acerto)
print("Total de elementos: %d" % total_de_elementos)

acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(teste_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

