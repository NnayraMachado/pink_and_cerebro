import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    norm, lognorm, expon, pareto,
    kstest, anderson
)
import math

# =========================================================
# CONFIGURAÇÃO
# =========================================================

CAMINHO_ARQUIVO = "Resultados.csv"
COLUNA_VALOR = "valor"

st.set_page_config(
    page_title="Sistema de Probabilidade e Decisão",
    layout="wide"
)

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def criar_kde_percentil(dados, p, pontos=150):
    dados = np.array([float(v) for v in dados if not pd.isna(v)])
    n = len(dados)

    limite = np.percentile(dados, p * 100)
    xmin = np.min(dados)
    xmax = limite

    sd = np.std(dados, ddof=1)
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    h = 0.9 * min(sd, iqr / 1.34) * (n ** -0.2)

    x_vals = np.linspace(xmin, xmax, pontos)
    kde_vals = []

    for x in x_vals:
        densidade = np.sum(norm.pdf(x, loc=dados, scale=h)) / (n * h)
        kde_vals.append(densidade)

    return x_vals, np.array(kde_vals), limite


def log_likelihood(dist, params, data):
    try:
        pdf_vals = dist.pdf(data, *params)
        pdf_vals[pdf_vals <= 0] = 1e-12
        return np.sum(np.log(pdf_vals))
    except:
        return -np.inf


# =========================================================
# 🧠 CAMADA DE DECISÃO
# =========================================================

def avaliar_jogo(sim):
    sim = np.array(sim)

    prob_perda = np.mean(sim < 0)
    valor_esperado = np.mean(sim)

    p5 = np.percentile(sim, 5)
    p1 = np.percentile(sim, 1)

    ganho_medio = np.mean(sim[sim > 0]) if np.any(sim > 0) else 0
    perda_media = np.mean(sim[sim < 0]) if np.any(sim < 0) else 0

    if valor_esperado < 0 and prob_perda > 0.5:
        decisao = "❌ NÃO VALE A PENA JOGAR"
        explicacao = "Valor esperado negativo e alta probabilidade de perda."
    elif valor_esperado > 0 and prob_perda < 0.3:
        decisao = "✅ VALE A PENA JOGAR"
        explicacao = "Valor esperado positivo com risco controlado."
    elif valor_esperado > 0 and prob_perda >= 0.3:
        decisao = "⚠️ JOGO ARRISCADO"
        explicacao = "Lucro esperado positivo, porém com alta volatilidade."
    else:
        decisao = "⚠️ CENÁRIO INDEFINIDO"
        explicacao = "Risco e retorno próximos do limite."

    return {
        "Probabilidade de perda": prob_perda,
        "Valor esperado": valor_esperado,
        "Pior perda (P5)": p5,
        "Pior perda extrema (P1)": p1,
        "Ganho médio": ganho_medio,
        "Perda média": perda_media,
        "Decisão": decisao,
        "Explicação": explicacao
    }


# =========================================================
# CARREGAMENTO DOS DADOS
# =========================================================

df = pd.read_csv(CAMINHO_ARQUIVO)
dados = df[COLUNA_VALOR]
dados_validos = dados.dropna().astype(float)

# =========================================================
# LAYOUT COM ABAS
# =========================================================

st.title("📊 Sistema de Probabilidade, Simulação e Decisão")

aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Visão Geral",
    "📈 Distribuições & Ajustes",
    "🎲 Simulação Monte Carlo",
    "🧠 Decisão do Jogador"
])

# =========================================================
# ABA 1 — VISÃO GERAL
# =========================================================

with aba1:
    st.subheader("Estatísticas Descritivas")
    st.write(dados_validos.describe())

    p55 = np.percentile(dados_validos, 55)
    p95 = np.percentile(dados_validos, 95)
    p99 = np.percentile(dados_validos, 99)

    st.subheader("Percentis Importantes")
    st.write(pd.DataFrame({
        "Percentil": ["P55", "P95", "P99"],
        "Valor": [p55, p95, p99]
    }))

    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(dados_validos, bins=60, kde=True, ax=ax)
    ax.axvline(p55, color="green", linestyle="--", label="P55")
    ax.axvline(p95, color="orange", linestyle="--", label="P95")
    ax.axvline(p99, color="red", linestyle="--", label="P99")
    ax.legend()
    ax.set_title("Histograma com Percentis")
    st.pyplot(fig)

# =========================================================
# ABA 2 — DISTRIBUIÇÕES E AJUSTES
# =========================================================

with aba2:
    st.subheader("Teste de Aderência (KS, AD, AIC, BIC)")

    distros = {
        "Normal": norm,
        "Lognormal": lognorm,
        "Exponencial": expon,
        "Pareto": pareto
    }

    resultados = []

    for nome, dist in distros.items():
        params = dist.fit(dados_validos)

        D, p_ks = kstest(dados_validos, dist.cdf, params)

        try:
            ad = anderson(dados_validos, dist='norm')
            AD = ad.statistic
        except:
            AD = np.nan

        ll = log_likelihood(dist, params, dados_validos)
        k = len(params)
        n = len(dados_validos)

        AIC = 2 * k - 2 * ll
        BIC = k * math.log(n) - 2 * ll

        resultados.append([nome, D, p_ks, AD, AIC, BIC, params])

    df_aderencia = pd.DataFrame(
        resultados,
        columns=["Distribuição", "KS", "KS p-valor", "AD", "AIC", "BIC", "Parâmetros"]
    ).sort_values("KS")

    st.write(df_aderencia)

    melhor = df_aderencia.iloc[0]
    st.success(f"Melhor ajuste: **{melhor['Distribuição']}**")

# =========================================================
# ABA 3 — SIMULAÇÃO MONTE CARLO
# =========================================================

with aba3:
    st.subheader("Simulação Monte Carlo (Distribuição Mista)")

    p95 = np.percentile(dados_validos, 95)
    x_comum = dados_validos[dados_validos < p95]
    x_raro = dados_validos[dados_validos >= p95]

    params_lognorm = lognorm.fit(x_comum)
    params_pareto = pareto.fit(x_raro)

    p_comum = len(x_comum) / len(dados_validos)

    def sorteio():
        return (
            lognorm(*params_lognorm).rvs()
            if np.random.rand() < p_comum
            else pareto(*params_pareto).rvs()
        )

    n_sim = st.slider("Número de simulações", 1000, 30000, 5000, step=1000)
    sim = np.array([sorteio() for _ in range(n_sim)])

    fig, ax = plt.subplots(figsize=(10,4))
    sns.kdeplot(dados_validos, ax=ax, label="Real")
    sns.kdeplot(sim, ax=ax, label="Simulado")
    ax.legend()
    ax.set_title("Distribuição Real vs Simulada")
    st.pyplot(fig)

# =========================================================
# ABA 4 — 🧠 DECISÃO DO JOGADOR
# =========================================================

with aba4:
    st.subheader("Camada de Decisão do Jogador")

    resultado = avaliar_jogo(sim)

    col1, col2, col3 = st.columns(3)

    col1.metric("Prob. de Perda", f"{resultado['Probabilidade de perda']*100:.1f}%")
    col2.metric("Valor Esperado", f"{resultado['Valor esperado']:.2f}")
    col3.metric("Ganho Médio", f"{resultado['Ganho médio']:.2f}")

    col1.metric("Pior perda (P5)", f"{resultado['Pior perda (P5)']:.2f}")
    col2.metric("Pior perda extrema (P1)", f"{resultado['Pior perda extrema (P1)']:.2f}")
    col3.metric("Perda Média", f"{resultado['Perda média']:.2f}")

    st.markdown("---")

    if "NÃO VALE" in resultado["Decisão"]:
        st.error(resultado["Decisão"])
    elif "VALE" in resultado["Decisão"]:
        st.success(resultado["Decisão"])
    else:
        st.warning(resultado["Decisão"])

    st.write(resultado["Explicação"])
