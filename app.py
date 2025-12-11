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
from scipy import stats

# ----------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------

CAMINHO_ARQUIVO = r"S:\Projeto_Pink_And_Cerebro\Resultados.csv"
COLUNA_VALOR = "valor"

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
# 3) HISTOGRAMA E BOXPLOT (mesmos dados)
# ----------------------------------------------------------

os.makedirs("graficos", exist_ok=True)

# Histograma + KDE
plt.figure(figsize=(10, 5))
sns.histplot(df[COLUNA_VALOR], bins=60, kde=True)
plt.title("Histograma + KDE")
plt.xlabel(COLUNA_VALOR)
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("graficos/histograma.png")
plt.close()

# Boxplot
plt.figure(figsize=(8, 3))
sns.boxplot(x=df[COLUNA_VALOR])
plt.title("Boxplot dos valores")
plt.xlabel(COLUNA_VALOR)
plt.tight_layout()
plt.savefig("graficos/boxplot.png")
plt.close()

# ----------------------------------------------------------
# HISTOGRAMA NO PADRÃO DO EXCEL (corrigido)
# ----------------------------------------------------------

dados = df[COLUNA_VALOR]
vazios = dados.isna().sum()
dados_validos = dados.dropna()

bins = [0, 10, 20, 30, 40, 50, 100]

frequencias, _ = np.histogram(dados_validos, bins=bins)

faixas = [f"Maior que {bins[i-1]} e até {bins[i]}" for i in range(1, len(bins))]
faixas.append(f"Acima de {bins[-1]}")

acima = (dados_validos > bins[-1]).sum()

intervalos = bins + ["Vazios"]
frequencias_final = list(frequencias) + [acima] + [vazios]
faixas_final = faixas + ["Células sem valor"]

df_histograma = pd.DataFrame({
    "Intervalo": intervalos,
    "Frequência": frequencias_final,
    "Faixa": faixas_final
})

print(df_histograma)
df_histograma.to_excel("histograma_saida.xlsx", index=False)

# Gráfico Excel-like
plt.figure(figsize=(10, 5))
plt.hist(dados_validos, bins=bins + [dados_validos.max()], edgecolor='black')
plt.title("Histograma por Intervalos (Excel)")
plt.xlabel("Valores")
plt.ylabel("Frequência")
plt.xticks(bins + [dados_validos.max()])
plt.savefig("graficos/histograma_intervalos_excel.png")
plt.close()

# ----------------------------------------------------------
# 4) TESTE DE DISTRIBUIÇÕES (KS e AD)
# ----------------------------------------------------------

def testar_distribuicao(dist, nome):
    params = dist.fit(x)
    ks = kstest(x, dist.name, params)
    ad = anderson(x, dist=dist.name if dist.name in ["expon", "norm", "lognorm"] else "norm")
    return {"nome": nome, "params": params, "KS": ks.statistic, "p-value": ks.pvalue, "AD": ad.statistic}

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
# 5) AJUSTE DA DISTRIBUIÇÃO MISTA (LOGNORMAL + PARETO)
# ----------------------------------------------------------

p95 = np.percentile(x, 95)
x_comum = x[x < p95]
x_raro = x[x >= p95]

params_lognorm = lognorm.fit(x_comum)
params_pareto = pareto.fit(x_raro)

p_comum = len(x_comum) / len(x)
p_raro = 1 - p_comum

print("\n===== AJUSTE DE MISTURA =====")
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

plt.figure(figsize=(10, 5))
sns.kdeplot(x, label="Real")
sns.kdeplot(sim, label="Simulado")
plt.legend()
plt.title("Comparação Real vs Simulado")
plt.savefig("graficos/comparacao_real_simulado.png")
plt.close()

print("\nSimulação concluída! Gráficos salvos em /graficos.")

# ----------------------------------------------------------
# 7) SALVAR RESULTADOS IMPORTANTES
# ----------------------------------------------------------

res = {
    "media_real": np.mean(x),
    "media_simulada": np.mean(sim),
    "p95_real": p95,
    "parametros_lognorm": params_lognorm,
    "parametros_pareto": params_pareto,
}

pd.DataFrame([res]).to_csv("resultados_distribuicao.csv", index=False)

print("\nArquivo resultados_distribuicao.csv criado!")
print("\n=== FINALIZADO ===")
