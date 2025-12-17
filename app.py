import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, lognorm, expon, pareto, kstest
import math

# =========================================================
# CONFIGURAÇÃO
# =========================================================

CAMINHO_ARQUIVO = "Resultados.csv"
COLUNA_VALOR = "valor"

st.set_page_config(
    page_title="Sistema de Probabilidade, Simulação e Decisão",
    layout="wide"
)

sns.set_style("darkgrid")

# =========================================================
# FUNÇÕES UTILITÁRIAS
# =========================================================

def formatar(x, casas=2):
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def log_likelihood(dist, params, data):
    pdf = dist.pdf(data, *params)
    pdf[pdf <= 0] = 1e-12
    return np.sum(np.log(pdf))


# =========================================================
# CACHE — DADOS E MODELOS
# =========================================================

@st.cache_data(show_spinner=False)
def carregar_dados():
    df = pd.read_csv(CAMINHO_ARQUIVO)
    return df[COLUNA_VALOR].dropna().astype(float)


@st.cache_data(show_spinner=False)
def ajustar_distribuicoes(dados):
    distros = {
        "Normal": norm,
        "Lognormal": lognorm,
        "Exponencial": expon,
        "Pareto": pareto
    }

    rows = []

    for nome, dist in distros.items():
        params = dist.fit(dados)
        ks, p = kstest(dados, dist.cdf, params)

        ll = log_likelihood(dist, params, dados)
        k = len(params)
        n = len(dados)

        aic = 2 * k - 2 * ll
        bic = k * math.log(n) - 2 * ll

        rows.append([nome, ks, p, aic, bic, params])

    df = pd.DataFrame(
        rows,
        columns=["Distribuição", "KS", "p-valor", "AIC", "BIC", "Parâmetros"]
    )

    df["ΔAIC"] = df["AIC"] - df["AIC"].min()
    df["ΔBIC"] = df["BIC"] - df["BIC"].min()

    return df.sort_values("KS")


@st.cache_data(show_spinner=False)
def ajustar_mistura(dados):
    p95 = np.percentile(dados, 95)
    comum = dados[dados < p95]
    raro = dados[dados >= p95]

    return (
        lognorm.fit(comum),
        pareto.fit(raro),
        len(comum) / len(dados)
    )


# =========================================================
# SIMULAÇÃO
# =========================================================

def sorteio_rodada():
    return (
        lognorm(*params_lognorm).rvs()
        if np.random.rand() < p_comum
        else pareto(*params_pareto).rvs()
    )


def simular_sessao(banca, aposta, rodadas):
    historico = []

    for _ in range(rodadas):
        banca += aposta * sorteio_rodada()
        historico.append(banca)
        if banca <= 0:
            return historico, True

    return historico, False


def stress_test(banca, aposta, rodadas, sessoes):
    finais = []
    quebras = 0

    for _ in range(sessoes):
        hist, quebrou = simular_sessao(banca, aposta, rodadas)
        finais.append(max(hist[-1], 0))
        if quebrou:
            quebras += 1

    return np.array(finais), quebras / sessoes


# =========================================================
# ESTRATÉGIA
# =========================================================

def kelly_fracionado(sim, fracao=0.25):
    ganhos = sim[sim > 0]
    perdas = -sim[sim < 0]

    if len(ganhos) == 0 or len(perdas) == 0:
        return 0

    p = len(ganhos) / len(sim)
    b = np.mean(ganhos) / np.mean(perdas)

    k = (p * (b + 1) - 1) / b
    return max(k * fracao, 0)


def ajustar_agressividade(primeiro):
    if primeiro > np.percentile(dados, 75):
        return 1.2
    if primeiro < np.percentile(dados, 25):
        return 0.7
    return 1.0


def regra_parada(banca, inicial):
    if banca <= inicial * 0.5:
        return True, "Stop Loss (-50%)"
    if banca >= inicial * 2:
        return True, "Stop Gain (+100%)"
    return False, ""


def score_risco(taxa_quebra):
    return int(min(100, taxa_quebra * 120))


# =========================================================
# DADOS E MODELOS
# =========================================================

dados = carregar_dados()
df_fit = ajustar_distribuicoes(dados)

params_lognorm, params_pareto, p_comum = ajustar_mistura(dados)

# =========================================================
# INTERFACE
# =========================================================

st.title("📊 Sistema de Probabilidade, Simulação e Decisão")

aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Resumo",
    "📈 Ajuste",
    "🏦 Banca & Estratégia",
    "🧠 Decisão Final"
])

# =========================================================
# ABA 1 — RESUMO
# =========================================================

with aba1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{len(dados):,}")
    c2.metric("Média", formatar(dados.mean()))
    c3.metric("Desvio", formatar(dados.std()))

    fig, ax = plt.subplots(figsize=(8,3))
    sns.histplot(dados, bins=80, kde=True, ax=ax)
    ax.set_title("Distribuição por Rodada")
    st.pyplot(fig)

# =========================================================
# ABA 2 — AJUSTE
# =========================================================

with aba2:
    df_show = df_fit.copy()
    df_show["KS"] = df_show["KS"].map(lambda x: f"{x:.3f}")
    df_show["p-valor"] = df_show["p-valor"].map(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
    df_show["ΔAIC"] = df_show["ΔAIC"].map(lambda x: f"{x:.1f}")
    df_show["ΔBIC"] = df_show["ΔBIC"].map(lambda x: f"{x:.1f}")
    df_show["Parâmetros"] = df_show["Parâmetros"].map(lambda p: [round(v, 3) for v in p])

    st.dataframe(df_show, use_container_width=True)
    st.success(f"Melhor ajuste global: {df_fit.iloc[0]['Distribuição']}")

# =========================================================
# ABA 3 — BANCA & ESTRATÉGIA
# =========================================================

with aba3:
    c1, c2, c3 = st.columns(3)

    banca = c1.number_input("Banca inicial", 100, 100000, 1000)
    aposta = c2.number_input("Aposta base", 1, 10000, 10)
    rodadas = c3.slider("Rodadas por sessão", 10, 500, 100)

    sim = np.array([sorteio_rodada() for _ in range(5000)])

    kelly_pct = kelly_fracionado(sim)
    aposta_kelly = banca * kelly_pct

    st.metric("Aposta recomendada (Kelly)", formatar(aposta_kelly))
    st.caption(f"{kelly_pct*100:.2f}% da banca")

    primeiro = sorteio_rodada()
    fator = ajustar_agressividade(primeiro)
    aposta_final = aposta_kelly * fator

    st.metric("Aposta ajustada (1ª rodada)", formatar(aposta_final))

# =========================================================
# ABA 4 — DECISÃO FINAL
# =========================================================

with aba4:
    finais, taxa_quebra = stress_test(
        banca,
        aposta_final,
        rodadas,
        sessoes=1500
    )

    risco = score_risco(taxa_quebra)

    c1, c2, c3 = st.columns(3)
    c1.metric("Chance de quebra", f"{taxa_quebra*100:.1f}%")
    c2.metric("Saldo médio final", formatar(np.mean(finais)))
    c3.metric("Score de risco", f"{risco}/100")

    parar, motivo = regra_parada(np.mean(finais), banca)

    if parar:
        st.error(f"⛔ PARAR: {motivo}")
    elif risco > 60:
        st.warning("⚠️ Risco elevado. Jogar com cautela.")
    else:
        st.success("✅ Estratégia viável e controlada.")

    fig, ax = plt.subplots(figsize=(8,3))
    sns.histplot(finais, bins=40, ax=ax)
    ax.set_title("Distribuição dos Saldos Finais")
    st.pyplot(fig)
