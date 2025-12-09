import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm, pareto
import os

CAMINHO_ARQUIVO = r"C:\Users\nirva\OneDrive\Área de Trabalho\NiR\PINK_CEREBRO\Resultados.csv"

st.title("🔮 Sistema de Probabilidade - Análise & Simulação")

st.write("Carregando dados...")
df = pd.read_csv(CAMINHO_ARQUIVO)

x = df["valor"].dropna().values

st.subheader("📊 Estatísticas Descritivas")
st.write(df.describe())

# Histograma
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(x, bins=50, kde=True, ax=ax)
st.pyplot(fig)

# Ajuste das distribuições mistas
p95 = np.percentile(x, 95)
x_comum = x[x < p95]
x_raro = x[x >= p95]

params_lognorm = lognorm.fit(x_comum)
params_pareto = pareto.fit(x_raro)

p_comum = len(x_comum) / len(x)

def sorteio():
    if np.random.rand() < p_comum:
        return lognorm(*params_lognorm).rvs()
    else:
        return pareto(*params_pareto).rvs()

st.subheader("🎲 Simulação Monte Carlo")
n = st.slider("Número de sorteios", 100, 20000, 5000)

sim = np.array([sorteio() for _ in range(n)])

fig, ax = plt.subplots(figsize=(8,4))
sns.kdeplot(sim, ax=ax, label="Simulado")
sns.kdeplot(x, ax=ax, label="Real")
ax.legend()
st.pyplot(fig)

st.success("Simulação concluída!")
