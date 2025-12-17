# ==========================================================
# SISTEMA DE ANÁLISE ESTATÍSTICA DOS MULTIPLICADORES (AVIATOR)
# ==========================================================
# - Carrega dados reais
# - Análise descritiva
# - Ajuste de distribuições
# - Testes de aderência
# - Distribuição mista
# - Simulação de validação
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

os.makedirs("graficos", exist_ok=True)

# ----------------------------------------------------------
# 1) CARREGAR DADOS
# ----------------------------------------------------------

print("Carregando dados...")
df = pd.read_csv(CAMINHO_ARQUIVO)
df = df.dropna()

x = df[COLUNA_VALOR].astype(float).values

print(f"Total de registros válidos: {len(x)}")

# ----------------------------------------------------------
# 2) ANÁLISE DESCRITIVA
# ----------------------------------------------------------

print("\n===== DESCRITIVO =====")
print(df[COLUNA_VALOR].describe())

percentis = np.percentile(x, [50, 75, 90, 95, 99])
df_percentis = pd.DataFrame({
    "Percentil": ["P50", "P75", "P90", "P95", "P99"],
    "Valor": percentis
})

print("\nPercentis:")
print(df_percentis)

df_percentis.to_csv("percentis.csv", index=False)

# ----------------------------------------------------------
# 3) VISUALIZAÇÃO
# ----------------------------------------------------------

plt.figure(figsize=(10,5))
sns.histplot(x, bins=80, kde=True)
plt.title("Distribuição dos Multiplicadores Finais")
plt.savefig("graficos/histograma_kde.png")
plt.close()

plt.figure(figsize=(8,3))
sns.boxplot(x=x)
plt.title("Boxplot dos Multiplicadores")
plt.savefig("graficos/boxplot.png")
plt.close()

# ----------------------------------------------------------
# 4) TESTE DE DISTRIBUIÇÕES (KS + AD)
# ----------------------------------------------------------

def testar_distribuicao(dist, nome):
    params = dist.fit(x)
    ks = kstest(x, dist.cdf, params)

    try:
        ad = anderson(x, dist=dist.name if dist.name in ["norm", "expon", "lognorm"] else "norm")
        ad_stat = ad.statistic
    except:
        ad_stat = np.nan

    return {
        "Distribuição": nome,
        "KS": ks.statistic,
        "p-valor": ks.pvalue,
        "AD": ad_stat,
        "Parâmetros": params
    }

print("\n===== TESTE DE DISTRIBUIÇÕES =====")

candidatas = [
    (stats.lognorm, "Lognormal"),
    (stats.gamma, "Gamma"),
    (stats.expon, "Exponencial"),
    (stats.weibull_min, "Weibull")
]

resultados = [testar_distribuicao(dist, nome) for dist, nome in candidatas]
df_testes = pd.DataFrame(resultados).sort_values("KS")

print(df_testes[["Distribuição", "KS", "p-valor", "AD"]])
df_testes.to_csv("teste_distribuicoes.csv", index=False)

# ----------------------------------------------------------
# 5) DISTRIBUIÇÃO MISTA (LOGNORMAL + PARETO)
# ----------------------------------------------------------

print("\n===== AJUSTE DE DISTRIBUIÇÃO MISTA =====")

p95 = np.percentile(x, 95)

x_comum = x[x < p95]
x_raro = x[x >= p95]

params_lognorm = lognorm.fit(x_comum)
params_pareto = pareto.fit(x_raro)

p_comum = len(x_comum) / len(x)
p_raro = 1 - p_comum

print(f"P(comum) = {p_comum:.3f}")
print(f"P(raro)  = {p_raro:.3f}")

# ----------------------------------------------------------
# 6) SIMULAÇÃO PARA VALIDAÇÃO
# ----------------------------------------------------------

def sorteio_misto():
    if np.random.rand() < p_comum:
        return lognorm(*params_lognorm).rvs()
    else:
        return pareto(*params_pareto).rvs()

print("\nGerando simulação Monte Carlo (10.000 pontos)...")

sim = np.array([sorteio_misto() for _ in range(10_000)])

plt.figure(figsize=(10,5))
sns.kdeplot(x, label="Real")
sns.kdeplot(sim, label="Simulado")
plt.title("Distribuição Real vs Simulada")
plt.legend()
plt.savefig("graficos/comparacao_real_simulado.png")
plt.close()

# ----------------------------------------------------------
# 7) SALVAR RESULTADOS PARA O APP
# ----------------------------------------------------------

resultado_final = {
    "total_registros": len(x),
    "media_real": np.mean(x),
    "mediana_real": np.median(x),
    "p95": p95,
    "p99": np.percentile(x, 99),
    "p_comum": p_comum,
    "p_raro": p_raro,
    "lognorm_params": params_lognorm,
    "pareto_params": params_pareto
}

pd.DataFrame([resultado_final]).to_csv(
    "resultados_analise_estatistica.csv",
    index=False
)

print("\nArquivos gerados:")
print("- percentis.csv")
print("- teste_distribuicoes.csv")
print("- resultados_analise_estatistica.csv")
print("- gráficos em /graficos")

print("\n=== ANÁLISE FINALIZADA ===")
