import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =========================
# TREINAMENTO DO MODELO
# =========================

df = pd.read_csv("teste.csv")

X = df.drop("movimento", axis=1)
y = df["movimento"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2
)

modelo = RandomForestClassifier()
modelo.fit(X_treino, y_treino)

print("Acurácia:", modelo.score(X_teste, y_teste))

# =========================
# MONITORAMENTO EM TEMPO REAL
# =========================

ultima_quantidade = len(df)

print("Monitorando novas leituras...\n")

while True:
    # lê o csv novamente
    df_tempo_real = pd.read_csv("teste.csv")

    # verifica se entrou nova linha
    if len(df_tempo_real) > ultima_quantidade:

        # pega somente a última linha no formato de tabela
        nova_linha = df_tempo_real[
            ["sensor1", "sensor2", "sensor3"]
        ].iloc[-1:]

        # faz a previsão do movimento
        resultado = modelo.predict(nova_linha)

        print("Nova leitura detectada:")
        print(nova_linha)

        print("Movimento detectado:", resultado[0])
        print("-" * 30)

        # atualiza a quantidade de linhas
        ultima_quantidade = len(df_tempo_real)

    # espera 1 segundo
    time.sleep(1)