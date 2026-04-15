import pandas as pd
import time
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ======================================
# CAMINHO UNIVERSAL DO CSV
# ======================================

arquivo_csv = Path(__file__).parent / "teste.csv"

# ======================================
# CRIAR CSV CASO NÃO EXISTA
# ======================================

if not arquivo_csv.exists():
    df_inicial = pd.DataFrame(
        columns=["sensor1", "sensor2", "sensor3", "movimento"]
    )
    df_inicial.to_csv(arquivo_csv, index=False)

# ======================================
# FUNÇÃO PARA ADICIONAR NOVA LEITURA
# ======================================

def adicionar_leitura(sensor1, sensor2, sensor3, movimento):
    nova_linha = pd.DataFrame(
        [[sensor1, sensor2, sensor3, movimento]],
        columns=["sensor1", "sensor2", "sensor3", "movimento"]
    )

    nova_linha.to_csv(
        arquivo_csv,
        mode="a",
        header=False,
        index=False
    )

# ======================================
# FUNÇÃO PARA TREINAR O MODELO
# ======================================

def treinar_modelo():
    df = pd.read_csv(arquivo_csv)

    # precisa ter pelo menos 2 linhas
    if len(df) < 2:
        return None, None

    X = df.drop("movimento", axis=1)
    y = df["movimento"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    modelo = RandomForestClassifier()
    modelo.fit(X_treino, y_treino)

    acuracia = modelo.score(X_teste, y_teste)

    print(f"Acurácia do modelo: {acuracia:.2f}")

    return modelo, X.columns

# ======================================
# FUNÇÃO DE MONITORAMENTO EM TEMPO REAL
# ======================================

def monitorar_csv(modelo, colunas):
    df = pd.read_csv(arquivo_csv)
    ultima_quantidade = len(df)

    print("\nMonitorando novas leituras...\n")

    while True:
        df_tempo_real = pd.read_csv(arquivo_csv)

        if len(df_tempo_real) > ultima_quantidade:
            nova_linha = df_tempo_real[
                list(colunas)
            ].iloc[-1:]

            resultado = modelo.predict(nova_linha)

            print("Nova leitura detectada:")
            print(nova_linha)

            print("Movimento detectado:", resultado[0])
            print("-" * 40)

            ultima_quantidade = len(df_tempo_real)

        time.sleep(1)

# ======================================
# SIMULAÇÃO DE BIOSINAIS
# ======================================

def simular_sensores():
    movimentos = {
        "direita": (0.6, 0.7, 0.1),
        "esquerda": (0.3, 0.4, 0.2),
        "repouso": (0.1, 0.1, 0.1)
    }

    for movimento, valores in movimentos.items():
        sensor1, sensor2, sensor3 = valores

        for _ in range(5):
            s1 = round(sensor1 + random.uniform(-0.05, 0.05), 2)
            s2 = round(sensor2 + random.uniform(-0.05, 0.05), 2)
            s3 = round(sensor3 + random.uniform(-0.05, 0.05), 2)

            adicionar_leitura(s1, s2, s3, movimento)

# ======================================
# PROGRAMA PRINCIPAL
# ======================================

# gera dados iniciais
simular_sensores()

# treina o modelo
modelo, colunas = treinar_modelo()

if modelo is not None:
    # adiciona novas leituras simuladas continuamente
    while True:
        movimento = random.choice(["direita", "esquerda", "repouso"])

        if movimento == "direita":
            adicionar_leitura(0.62, 0.71, 0.09, movimento)

        elif movimento == "esquerda":
            adicionar_leitura(0.28, 0.39, 0.22, movimento)

        else:
            adicionar_leitura(0.10, 0.12, 0.11, movimento)

        # monitora e prevê
        monitorar_csv(modelo, colunas)