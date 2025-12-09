# ==========================================================
#  SISTEMA DE ANÁLISE E SIMULAÇÃO DE DISTRIBUIÇÃO MISTA
# ==========================================================
#  Carrega dados reais, realiza análises, ajusta distribuições,
#  valida com KS/AD, cria distribuição mista e gera simulações
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import lognorm, pareto, kstest, anderson
from scipy.stats.mstats import winsorize
from sklearn.mixture import GaussianMixture

# ----------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------

CAMINHO_ARQUIVO = r"C:\Users\nirva\OneDrive\Área de Trabalho\NiR\PINK_CEREBRO\Resultados.csv"
COLUNA_VALOR = "valor"     # nome da coluna no CSV

# ----------------------------------------------------------
# 1) CARREGAR DADOS
# ----------------------------------------------------------

print("Carregando dados...")
df = pd.read_csv(CAMINHO_ARQUIVO, sep=",")
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

plt.figure(figsize=(10,5))
sns.histplot(x, bins=60, kde=True)
plt.title("Histograma + KDE")
plt.savefig("graficos/histograma.png")
plt.close()

plt.figure(figsize=(8,3))
sns.boxplot(x=x)
plt.title("Boxplot dos valores")
plt.savefig("graficos/boxplot.png")
plt.close()

# ----------------------------------------------------------
# 4) TESTE DE DISTRIBUIÇÕES COMUNS (KS e AD)
# ----------------------------------------------------------

from scipy import stats

def testar_distribuicao(dist, nome):
    params = dist.fit(x)
    ks = kstest(x, dist.name, params)
    ad = anderson(x, dist=dist.name if dist.name in ["expon", "norm", "lognorm"] else "norm")

    return {
        "nome": nome,
        "params": params,
        "KS": ks.statistic,
        "p-value": ks.pvalue,
        "AD": ad.statistic
    }

print("\n===== TESTE DE DISTRIBUIÇÕES =====")

candidatas = [
    (stats.lognorm, "Lognormal"),
    (stats.gamma, "Gamma"),
    (stats.expon, "Exponencial"),
    (stats.weibull_min, "Weibull")
]

resultados = [testar_distribuicao(dist, nome) for dist, nome in candidatas]

for r in resultados:
    print(f"\n{r['nome']}:")
    print(f"  KS = {r['KS']:.5f}")
    print(f"  p-value = {r['p-value']:.5f}")
    print(f"  AD = {r['AD']:.5f}")

# ----------------------------------------------------------
# 5) AJUSTE DE DISTRIBUIÇÃO MISTA (LOGNORMAL + PARETO)
# ----------------------------------------------------------

print("\n===== AJUSTE DE MISTURA =====")

p95 = np.percentile(x, 95)
x_comum = x[x < p95]
x_raro = x[x >= p95]

# Ajustar parte comum (lognormal)
params_lognorm = lognorm.fit(x_comum)

# Ajustar parte rara (pareto)
params_pareto = pareto.fit(x_raro)

p_comum = len(x_comum) / len(x)
p_raro = 1 - p_comum

print(f"Probabilidade comum: {p_comum:.3f}")
print(f"Probabilidade rara:  {p_raro:.3f}")

def sorteio():
    if np.random.rand() < p_comum:
        return lognorm(*params_lognorm).rvs()
    else:
        return pareto(*params_pareto).rvs()

# ----------------------------------------------------------
# 6) SIMULAÇÃO MONTE CARLO
# ----------------------------------------------------------

print("\nGerando 10.000 valores simulados...")
sim = np.array([sorteio() for _ in range(10000)])

plt.figure(figsize=(10,5))
sns.kdeplot(x, label="Real")
sns.kdeplot(sim, label="Simulado")
plt.title("Comparação Real vs Simulado")
plt.legend()
plt.savefig("graficos/comparacao_real_simulado.png")
plt.close()

print("\nSimulação concluída! Gráficos salvos na pasta /graficos.")

# ----------------------------------------------------------
# 7) SALVAR RESULTADOS IMPORTANTES
# ----------------------------------------------------------

resultado_dict = {
    "media_real": np.mean(x),
    "media_simulada": np.mean(sim),
    "p95_real": p95,
    "parametros_lognorm": params_lognorm,
    "parametros_pareto": params_pareto,
}

pd.DataFrame([resultado_dict]).to_csv("resultados_distribuicao.csv", index=False)

print("\nArquivo resultados_distribuicao.csv criado!")
print("\n=== FINALIZADO ===")
