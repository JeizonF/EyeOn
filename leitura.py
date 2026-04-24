import pandas as pd
import keyboard
import time
from sklearn.ensemble import RandomForestClassifier

ARQUIVO = "teste.csv"

try:
    df = pd.read_csv(ARQUIVO)
except:
    df = pd.DataFrame(columns=["sensor1", "sensor2", "sensor3", "sensor4", "movimento"])

modelo = None

while True:
    print(df.columns.tolist())
    print("\n=== ESCOLHA O MODO ===")
    print("F10 = Treinamento | F4 = Previsão | F1 = Sair")

    # espera tecla
    while True:
        if keyboard.is_pressed('f10'):
            modo = "treino"
            break
        elif keyboard.is_pressed('f4'):
            modo = "prever"
            break
        elif keyboard.is_pressed('f1'):
            print("Encerrando...")
            exit()

    time.sleep(0.3)  # evita múltiplas leituras

    print(f"\nModo selecionado: {modo.upper()}")

    # ---------------- TREINAMENTO ----------------
    if modo == "treino":
        print("\n--- NOVA LEITURA (TREINO) ---")

        s1 = float(input("Sensor 1: "))
        s2 = float(input("Sensor 2: "))
        s3 = float(input("Sensor 3: "))
        s4 = float(input("Sensor 4: "))

        print("Pressione: D=direita | E=esquerda | R=repouso")

        while True:
            if keyboard.is_pressed('d'):
                movimento = "direita"
                break
            elif keyboard.is_pressed('e'):
                movimento = "esquerda"
                break
            elif keyboard.is_pressed('r'):
                movimento = "repouso"
                break

        time.sleep(0.3)

        novo = pd.DataFrame([[s1, s2, s3, s4, movimento]],
                            columns=df.columns)

        df = pd.concat([df, novo], ignore_index=True)
        df.to_csv(ARQUIVO, index=False)

        print("✔ Dado salvo!")

    # ---------------- PREVISÃO ----------------
    elif modo == "prever":
        print("\n--- PREVISÃO ---")

        if len(df) < 5:
            print("⚠️ Poucos dados para treinar!")
            continue

        # treina modelo
        X = df.drop("movimento", axis=1)
        y = df["movimento"]

        modelo = RandomForestClassifier()
        modelo.fit(X, y)

        print("✔ Modelo treinado!")

        s1 = float(input("Sensor 1: "))
        s2 = float(input("Sensor 2: "))
        s3 = float(input("Sensor 3: "))
        s4 = float(input("Sensor 4: "))

        novo_dado = pd.DataFrame([[s1, s2, s3, s4]], columns=X.columns)

        resultado = modelo.predict(novo_dado)[0]

        print("Movimento previsto:", resultado)