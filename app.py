import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm, pareto

CAMINHO_ARQUIVO = "Resultados.csv"

# Título
st.title("Sistema de Probabilidade - Análise & Simulação")

# Carregar dados
df = pd.read_csv(CAMINHO_ARQUIVO)
dados = df["valor"].dropna()

# ---------------------------------------------------------
# ESTATÍSTICAS DESCRITIVAS
# ---------------------------------------------------------
st.subheader("Estatísticas Descritivas")
st.write(df.describe())

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# HISTOGRAMA
# ---------------------------------------------------------
st.subheader("Histograma por Intervalos")

bins = [0,10,20,30,40,50,100]
frequencias, _ = np.histogram(dados, bins=bins)
acima = (dados > bins[-1]).sum()
vazios = df["valor"].isna().sum()

faixas = [f"{bins[i-1]} – {bins[i]}" for i in range(1, len(bins))]
faixas.append(f"Acima de {bins[-1]}")
faixas.append("Vazios")

freq_final = list(frequencias) + [acima] + [vazios]

df_hist = pd.DataFrame({
    "Faixa": faixas,
    "Frequência": freq_final
})

st.write(df_hist)

# Gráfico Excel-like
fig, ax = plt.subplots(figsize=(10,4))
ax.bar(df_hist["Faixa"], df_hist["Frequência"], color="steelblue")
plt.xticks(rotation=45)
ax.set_title("Histograma – Intervalos Excel")
st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# HISTOGRAMA NORMAL
# ---------------------------------------------------------
st.subheader("Histograma Geral")

fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(dados, bins=50, kde=True, ax=ax)
ax.set_title("Histograma Geral com KDE")
st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# DISTRIBUIÇÃO MISTA E SIMULAÇÃO
# ---------------------------------------------------------
st.subheader("Simulação Monte Carlo Baseada na Distribuição Mista")

p95 = np.percentile(dados, 95)
x_comum = dados[dados < p95]
x_raro = dados[dados >= p95]

params_lognorm = lognorm.fit(x_comum)
params_pareto = pareto.fit(x_raro)

p_comum = len(x_comum) / len(dados)

def sorteio():
    return lognorm(*params_lognorm).rvs() if np.random.rand() < p_comum else pareto(*params_pareto).rvs()

n = st.slider("Número de sorteios", 100, 20000, 5000)

sim = np.array([sorteio() for _ in range(n)])

fig, ax = plt.subplots(figsize=(8,4))
sns.kdeplot(sim, ax=ax, label="Simulado")
sns.kdeplot(dados, ax=ax, label="Real")
ax.legend()
ax.set_title("Distribuição Real vs Simulada")
st.pyplot(fig)

st.success("Simulação concluída!")


