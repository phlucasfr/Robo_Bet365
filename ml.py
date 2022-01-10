import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier


arquivo = pd.read_csv('C:/ML/escanteios.csv')

y = arquivo['terminou_over']
x = arquivo.drop('terminou_over', axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.001, shuffle=False)

modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

resultado = modelo.score(x_teste, y_teste)
print("Acurácia:", resultado)

print(y_teste)
print(x_teste)

previsao = modelo.predict(x_teste)
print(previsao)

