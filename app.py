# ==========================================================
#  SISTEMA DE ANÁLISE E SIMULAÇÃO DE DISTRIBUIÇÃO MISTA
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import lognorm, pareto, kstest, anderson
from scipy import stats

# ----------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------

CAMINHO_ARQUIVO = "Resultados.csv"
COLUNA_VALOR = "valor"

# ----------------------------------------------------------
# 1) CARREGAR DADOS
# ----------------------------------------------------------

print("Carregando dados...")
df = pd.read_csv(CAMINHO_ARQUIVO)
df = df.dropna()

x = df[COLUNA_VALOR].values.astype(float)

print(f"Total de registros carregados: {len(x)}")

# ----------------------------------------------------------
# 2) ANÁLISE DESCRITIVA
# ----------------------------------------------------------

print("\n===== DESCRITIVO =====")
print(df[COLUNA_VALOR].describe())

percentis = df[COLUNA_VALOR].quantile([0.5, 0.75, 0.90, 0.95, 0.99])
print("\nPercentis:")
print(percentis)

# ----------------------------------------------------------
# 3) HISTOGRAMA E BOXPLOT
# ----------------------------------------------------------

os.makedirs("graficos", exist_ok=True)

plt.figure(figsize=(10, 5))
sns.histplot(df[COLUNA_VALOR], bins=60, kde=True)
plt.title("Histograma + KDE")
plt.savefig("graficos/histograma.png")
plt.close()

plt.figure(figsize=(8, 3))
sns.boxplot(x=df[COLUNA_VALOR])
plt.title("Boxplot")
plt.savefig("graficos/boxplot.png")
plt.close()

# ----------------------------------------------------------
# HISTOGRAMA EXCEL-LIKE (corrigido)
# ----------------------------------------------------------

dados = df[COLUNA_VALOR].dropna()
vazios = df[COLUNA_VALOR].isna().sum()

bins = [0, 10, 20, 30, 40, 50, 100]
frequencias, _ = np.histogram(dados, bins=bins)

faixas = [f"Maior que {bins[i-1]} e até {bins[i]}" for i in range(1, len(bins))]
faixas.append(f"Acima de {bins[-1]}")

acima = (dados > bins[-1]).sum()

df_histograma = pd.DataFrame({
    "Faixa": faixas + ["Células sem valor"],
    "Frequência": list(frequencias) + [acima] + [vazios]
})

print(df_histograma)

df_histograma.to_excel("histograma_saida.xlsx", index=False)

plt.figure(figsize=(10,5))
plt.bar(df_histograma["Faixa"], df_histograma["Frequência"])
plt.xticks(rotation=45)
plt.title("Histograma - Intervalos Excel")
plt.savefig("graficos/histograma_intervalos_excel.png")
plt.close()

# ----------------------------------------------------------
# 4) TESTES DE DISTRIBUIÇÃO
# ----------------------------------------------------------

def testar_distribuicao(dist, nome):
    params = dist.fit(x)
    ks = kstest(x, dist.name, params)
    ad = anderson(x, dist=dist.name if dist.name in ["expon", "norm", "lognorm"] else "norm")
    return {"nome": nome, "params": params, "KS": ks.statistic, "p": ks.pvalue, "AD": ad.statistic}

print("\n===== TESTE DE DISTRIBUIÇÕES =====")

for dist, nome in [
    (stats.lognorm, "Lognormal"),
    (stats.gamma, "Gamma"),
    (stats.expon, "Exponencial"),
    (stats.weibull_min, "Weibull"),
]:
    r = testar_distribuicao(dist, nome)
    print(f"\n{nome}: KS={r['KS']:.5f}, p={r['p']:.5f}, AD={r['AD']:.5f}")

# ----------------------------------------------------------
# 5) DISTRIBUIÇÃO MISTA
# ----------------------------------------------------------

p95 = np.percentile(x, 95)
x_comum = x[x < p95]
x_raro = x[x >= p95]

params_lognorm = lognorm.fit(x_comum)
params_pareto = pareto.fit(x_raro)

p_comum = len(x_comum)/len(x)

def sorteio():
    return lognorm(*params_lognorm).rvs() if np.random.rand() < p_comum else pareto(*params_pareto).rvs()

# ----------------------------------------------------------
# 6) SIMULAÇÃO
# ----------------------------------------------------------

sim = np.array([sorteio() for _ in range(10000)])

plt.figure(figsize=(10,5))
sns.kdeplot(x, label="Real")
sns.kdeplot(sim, label="Simulado")
plt.legend()
plt.savefig("graficos/comparacao_real_simulado.png")
plt.close()

pd.DataFrame([{
    "media_real": np.mean(x),
    "media_simulada": np.mean(sim),
    "p95_real": p95,
    "params_lognorm": params_lognorm,
    "params_pareto": params_pareto
}]).to_csv("resultados_distribuicao.csv", index=False)

print("\n=== FINALIZADO ===")
