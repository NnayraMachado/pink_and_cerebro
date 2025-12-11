import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm, pareto

CAMINHO_ARQUIVO = "Resultados.csv"


from scipy.stats import norm

def criar_kde_percentil(dados, p, pontos=150):
    """Gera KDE entre xmin e o valor do percentil p."""
    dados = np.array([float(v) for v in dados if not pd.isna(v)])
    n = len(dados)

    limite = np.percentile(dados, p * 100)
    xmin = np.min(dados)
    xmax = limite

    # Bandwidth Silverman
    sd = np.std(dados, ddof=1)
    q1 = np.percentile(dados, 25)
    q3 = np.percentile(dados, 75)
    iqr = q3 - q1
    h = 0.9 * min(sd, iqr / 1.34) * (n ** -0.2)

    x_vals = np.linspace(xmin, xmax, pontos)
    kde_vals = []

    for x in x_vals:
        densidade = np.sum(norm.pdf(x, loc=dados, scale=h)) / (n * h)
        kde_vals.append(round(densidade, 6))

    return x_vals, np.array(kde_vals), limite


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
# HISTOGRAMA POR INTERVALOS
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

# Gráfico
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

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# K D E   P O R   P E R C E N T I S   (55 / 95 / 99)
# ---------------------------------------------------------

st.subheader("Curvas KDE para Percentis 55, 95 e 99")

# ---------------- KDE P55 ----------------
x55, kde55, limite55 = criar_kde_percentil(dados_validos, 0.55)

df_kde55 = pd.DataFrame({"X": x55, "KDE": kde55})
st.write("### KDE – Percentil 55%")
st.write(df_kde55.head(20))  # Mostra só 20 linhas no app

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(x55, kde55, color="green")
ax.set_title(f"KDE até o Percentil 55 (P55 = {limite55:.2f})")
ax.set_xlabel("X")
ax.set_ylabel("Densidade")
st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ---------------- KDE P95 ----------------
x95, kde95, limite95 = criar_kde_percentil(dados_validos, 0.95)

df_kde95 = pd.DataFrame({"X": x95, "KDE": kde95})
st.write("### KDE – Percentil 95%")
st.write(df_kde95.head(20))

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(x95, kde95, color="orange")
ax.set_title(f"KDE até o Percentil 95 (P95 = {limite95:.2f})")
ax.set_xlabel("X")
ax.set_ylabel("Densidade")
st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ---------------- KDE P99 ----------------
x99, kde99, limite99 = criar_kde_percentil(dados_validos, 0.99)

df_kde99 = pd.DataFrame({"X": x99, "KDE": kde99})
st.write("### KDE – Percentil 99%")
st.write(df_kde99.head(20))

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(x99, kde99, color="red")
ax.set_title(f"KDE até o Percentil 99 (P99 = {limite99:.2f})")
ax.set_xlabel("X")
ax.set_ylabel("Densidade")
st.pyplot(fig)

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


