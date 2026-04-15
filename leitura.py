import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("teste.csv")

X = df.drop("movimento", axis=1)
y = df["movimento"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2
)

modelo = RandomForestClassifier()
modelo.fit(X_treino, y_treino)

print("Acurácia:", modelo.score(X_teste, y_teste))

novo_dado = pd.DataFrame(
    [[0.54, 0.70, 0.08]],
    columns=X.columns
)

resultado = modelo.predict(novo_dado)

print("Movimento:", resultado[0])
print(novo_dado)