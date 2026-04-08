from sklearn.naive_bayes import MultinomialNB

import pandas as pd

df = pd.read_csv("buscas.csv")  # data frame

# print(df)

X_df = df[["home", "busca", "logado"]]
Y_df = df["comprou"]

Xdummies_df = pd.get_dummies(X_df).astype(
    int
)  # precisa do .astype(int) para converter para int
Ydummies_df = Y_df

# print(type(Ydummies_df))

X = Xdummies_df.values
Y = Ydummies_df.values


#a eficacia do algoritmo que chuta tudo 0 ou 1
acerto_de_um =sum(Y)
acerto_de_zero= len(y) - acerto_de_um
max(acerto_de_um,acerto_de_zero)

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

diferencas = resultado - teste_marcacoes
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos


print(resultado)
print(diferencas)
print(taxa_de_acerto)
print(total_de_elementos)