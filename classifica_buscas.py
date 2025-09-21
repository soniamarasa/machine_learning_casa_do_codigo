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

# print(Y)

tamanho_de_treino = 0.9 * len(Y)

print(tamanho_de_treino)
