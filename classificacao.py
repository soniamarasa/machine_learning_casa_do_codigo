from sklearn.naive_bayes import MultinomialNB

# [é gordinho?, tem perninha curta?, faz auau?]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro1 = [1, 1, 1]
cachorro2 = [0, 1, 1]
cachorro3 = [0, 1, 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# 1 = porco, -1 = cachorro
marcacoes = [1, 1, 1, -1, -1, -1]

# precisa estar dentro de outra lista
misterioso1 = [[1, 1, 1]]
misterioso2 = [[1, 0, 0]]
misterioso3 = [[0, 0, 1]]
teste = [misterioso1[0], misterioso2[0], misterioso3[0]]
marcacoes_teste = [-1, 1, -1]

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)
resultado = modelo.predict(teste)
diferencas = resultado - marcacoes_teste
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

# saída
print(resultado)
print(diferencas)
# print(acertos)
# print(total_de_acertos)
# print(total_de_elementos)
print(taxa_de_acerto)
