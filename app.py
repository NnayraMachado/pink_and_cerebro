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
    page_title="Sistema de Probabilidade e Decisão",
    layout="wide"
)

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def log_likelihood(dist, params, data):
    pdf_vals = dist.pdf(data, *params)
    pdf_vals[pdf_vals <= 0] = 1e-12
    return np.sum(np.log(pdf_vals))


def formatar_num(x, casas=3):
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# =========================================================
# CARREGAR DADOS
# =========================================================

df = pd.read_csv(CAMINHO_ARQUIVO)
dados = df[COLUNA_VALOR].dropna().astype(float)

# =========================================================
# AJUSTE DE DISTRIBUIÇÕES
# =========================================================

distros = {
    "Normal": norm,
    "Lognormal": lognorm,
    "Exponencial": expon,
    "Pareto": pareto
}

resultados = []

for nome, dist in distros.items():
    params = dist.fit(dados)

    D, p_ks = kstest(dados, dist.cdf, params)

    ll = log_likelihood(dist, params, dados)
    k = len(params)
    n = len(dados)

    AIC = 2 * k - 2 * ll
    BIC = k * math.log(n) - 2 * ll

    resultados.append([nome, D, p_ks, AIC, BIC, params])

df_fit = pd.DataFrame(
    resultados,
    columns=["Distribuição", "KS", "p-valor", "AIC", "BIC", "Parâmetros"]
)

# Normalização AIC / BIC
df_fit["ΔAIC"] = df_fit["AIC"] - df_fit["AIC"].min()
df_fit["ΔBIC"] = df_fit["BIC"] - df_fit["BIC"].min()

df_fit = df_fit.sort_values("KS")

# =========================================================
# DISTRIBUIÇÃO MISTA (ROBUSTA)
# =========================================================

p95 = np.percentile(dados, 95)
x_comum = dados[dados < p95]
x_raro = dados[dados >= p95]

params_lognorm = lognorm.fit(x_comum)
params_pareto = pareto.fit(x_raro)

p_comum = len(x_comum) / len(dados)

def sorteio_rodada():
    return (
        lognorm(*params_lognorm).rvs()
        if np.random.rand() < p_comum
        else pareto(*params_pareto).rvs()
    )

# =========================================================
# SIMULAÇÕES DE BANCA
# =========================================================

def simular_sessao(banca, aposta, n_rodadas):
    for _ in range(n_rodadas):
        banca += aposta * sorteio_rodada()
        if banca <= 0:
            return banca, True
    return banca, False


def stress_test(banca, aposta, n_rodadas, n_sessoes):
    finais = []
    quebras = 0

    for _ in range(n_sessoes):
        saldo, quebrou = simular_sessao(banca, aposta, n_rodadas)
        finais.append(max(saldo, 0))
        if quebrou:
            quebras += 1

    return np.array(finais), quebras / n_sessoes

# =========================================================
# INTERFACE
# =========================================================

st.title("📊 Sistema de Probabilidade, Simulação e Decisão")

aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Resumo",
    "📈 Ajuste Estatístico",
    "🏦 Banca & Risco",
    "🧠 Decisão Final"
])

# =========================================================
# ABA 1 — RESUMO
# =========================================================

with aba1:
    st.subheader("Resumo dos Dados")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total de Registros", f"{len(dados):,}")
    col2.metric("Média", formatar_num(dados.mean()))
    col3.metric("Desvio Padrão", formatar_num(dados.std()))

    fig, ax = plt.subplots(figsize=(9,4))
    sns.histplot(dados, bins=80, kde=True, ax=ax)
    ax.set_title("Distribuição dos Resultados por Rodada")
    st.pyplot(fig)

# =========================================================
# ABA 2 — AJUSTE ESTATÍSTICO
# =========================================================

with aba2:
    st.subheader("Teste de Aderência (formatado)")

    df_display = df_fit.copy()

    df_display["KS"] = df_display["KS"].apply(lambda x: f"{x:.3f}")
    df_display["p-valor"] = df_display["p-valor"].apply(
        lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}"
    )
    df_display["ΔAIC"] = df_display["ΔAIC"].apply(lambda x: f"{x:.1f}")
    df_display["ΔBIC"] = df_display["ΔBIC"].apply(lambda x: f"{x:.1f}")
    df_display["Parâmetros"] = df_display["Parâmetros"].apply(
        lambda p: [round(v, 3) for v in p]
    )

    st.dataframe(df_display, use_container_width=True)

    melhor = df_fit.iloc[0]
    st.success(f"Melhor ajuste global: **{melhor['Distribuição']}**")

    st.info(
        "⚠️ Atenção: para jogos reais, usamos **distribuição mista** "
        "(Lognormal para eventos comuns + Pareto para extremos)."
    )

# =========================================================
# ABA 3 — BANCA & RISCO
# =========================================================

with aba3:
    st.subheader("Simulação de Banca")

    col1, col2, col3 = st.columns(3)

    banca = col1.number_input("Banca inicial", 100, 100000, 1000)
    aposta = col2.number_input("Aposta por rodada", 1, 10000, 10)
    rodadas = col3.slider("Rodadas por sessão", 10, 500, 100)

    finais, taxa_quebra = stress_test(
        banca, aposta, rodadas, n_sessoes=2000
    )

    col1.metric("Chance de Quebra", f"{taxa_quebra*100:.1f}%")
    col2.metric("Saldo Médio Final", formatar_num(np.mean(finais)))
    col3.metric("Pior Caso (1%)", formatar_num(np.percentile(finais, 1)))

    fig, ax = plt.subplots(figsize=(9,4))
    sns.histplot(finais, bins=50, ax=ax)
    ax.set_title("Distribuição dos Saldos Finais")
    st.pyplot(fig)

# =========================================================
# ABA 4 — DECISÃO FINAL
# =========================================================

with aba4:
    st.subheader("Decisão Automatizada")

    if taxa_quebra > 0.5:
        st.error("❌ Estratégia extremamente arriscada. NÃO recomendado.")
    elif taxa_quebra > 0.25:
        st.warning("⚠️ Estratégia arriscada. Exige controle rígido.")
    else:
        st.success("✅ Estratégia viável com risco controlado.")

    st.markdown(
        f"""
        **Resumo Final:**
        - Chance de quebra: **{taxa_quebra*100:.1f}%**
        - Saldo médio esperado: **{formatar_num(np.mean(finais))}**
        - Cenário extremo (1%): **{formatar_num(np.percentile(finais, 1))}**
        """
    )
