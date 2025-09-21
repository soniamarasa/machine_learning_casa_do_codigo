from sklearn.naive_bayes import MultinomialNB

from dados import carregar_acessos

X, Y = carregar_acessos()

# print(Y)
# print(X)

modelo = MultinomialNB()
modelo.fit(X, Y)
misterioso1 = [[1, 0, 1]]
misterioso2 = [[0, 1, 0]]
misterioso3 = [[1, 0, 0]]
misterioso4 = [[1, 1, 0]]
misterioso5 = [[1, 1, 1]]

resultado = modelo.predict(X) # [misterioso1[0], misterioso2[0], misterioso3[0], misterioso4[0], misterioso5[0]]

print(resultado)
print(Y)

diferencas = resultado - Y
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(Y)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

# saída
print(resultado)
print(diferencas)
print(taxa_de_acerto)
print(total_de_elementos)
