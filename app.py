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
dados = df["valor"]          # <- não removemos NaN ainda
dados_validos = dados.dropna()

# ---------------------------------------------------------
# ESTATÍSTICAS DESCRITIVAS
# ---------------------------------------------------------
st.subheader("Estatísticas Descritivas")
st.write(df.describe())

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# PERCENTIS 55 / 95 / 99 (incluindo nulos no total)
# ---------------------------------------------------------
st.subheader("Percentis 55, 95 e 99")

total = len(dados)
validos = len(dados_validos)
nulos = dados.isna().sum()

p55 = np.percentile(dados_validos, 55)
p95 = np.percentile(dados_validos, 95)
p99 = np.percentile(dados_validos, 99)

df_percentis = pd.DataFrame({
    "Medida": ["Total registros", "Válidos", "Nulos", "P55", "P95", "P99"],
    "Valor": [total, validos, nulos, p55, p95, p99]
})

st.write(df_percentis)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# HISTOGRAMA POR INTERVALOS (EXCEL-LIKE)
# ---------------------------------------------------------
st.subheader("Histograma por Intervalos")

bins = [0,10,20,30,40,50,100]
frequencias, _ = np.histogram(dados_validos, bins=bins)
acima = (dados_validos > bins[-1]).sum()

faixas = [f"{bins[i-1]} – {bins[i]}" for i in range(1, len(bins))]
faixas.append(f"Acima de {bins[-1]}")
faixas.append("Vazios")

freq_final = list(frequencias) + [acima] + [nulos]

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
# HISTOGRAMA NORMAL COM LINHAS DOS PERCENTIS
# ---------------------------------------------------------
st.subheader("Histograma Geral com Percentis")

fig, ax = plt.subplots(figsize=(10,4))
sns.histplot(dados_validos, bins=50, kde=True, ax=ax)

# Linhas verticais dos percentis
ax.axvline(p55, color='green', linestyle='--', linewidth=2, label=f"P55 = {p55:.2f}")
ax.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f"P95 = {p95:.2f}")
ax.axvline(p99, color='red', linestyle='--', linewidth=2, label=f"P99 = {p99:.2f}")

ax.legend()
ax.set_title("Histograma com Percentis 55, 95 e 99")
st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# DISTRIBUIÇÃO MISTA E SIMULAÇÃO
# ---------------------------------------------------------
st.subheader("Simulação Monte Carlo Baseada na Distribuição Mista")

x_comum = dados_validos[dados_validos < p95]
x_raro = dados_validos[dados_validos >= p95]

params_lognorm = lognorm.fit(x_comum)
params_pareto = pareto.fit(x_raro)

p_comum = len(x_comum) / len(dados_validos)

def sorteio():
    return lognorm(*params_lognorm).rvs() if np.random.rand() < p_comum else pareto(*params_pareto).rvs()

n = st.slider("Número de sorteios", 100, 20000, 5000)
sim = np.array([sorteio() for _ in range(n)])

fig, ax = plt.subplots(figsize=(8,4))
sns.kdeplot(sim, ax=ax, label="Simulado")
sns.kdeplot(dados_validos, ax=ax, label="Real")
ax.legend()
ax.set_title("Distribuição Real vs Simulada")
st.pyplot(fig)

st.success("Simulação concluída!")
