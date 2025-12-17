import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÃO
# =========================================================

CAMINHO_ARQUIVO = "Resultados.csv"
COLUNA_VALOR = "valor"

st.set_page_config(
    page_title="Probabilidade Condicional",
    layout="wide"
)

# =========================================================
# UTILIDADES
# =========================================================

def formatar(x, casas=2):
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =========================================================
# CACHE DE DADOS
# =========================================================

@st.cache_data(show_spinner=False)
def carregar_dados():
    df = pd.read_csv(CAMINHO_ARQUIVO)
    return df[COLUNA_VALOR].dropna().astype(float)

dados = carregar_dados()

# =========================================================
# PROBABILIDADE CONDICIONAL 
# =========================================================

def distribuicao_condicional(dados, x_atual):
    """Distribuição dos multiplicadores finais dado que o jogo já chegou em x_atual"""
    return dados[dados >= x_atual]


def prob_atingir(dados, x_atual, alvo):
    cond = distribuicao_condicional(dados, x_atual)
    if len(cond) == 0 or alvo < x_atual:
        return 0.0
    return np.mean(cond >= alvo)


def valor_esperado_continuar(x_atual, alvo, prob):
    """
    EV simples e honesto:
    - se chegar no alvo → ganha (alvo - x_atual)
    - se crashar antes → perde x_atual
    """
    ganho = alvo - x_atual
    perda = x_atual
    return prob * ganho - (1 - prob) * perda


def melhor_cashout(dados, x_atual, alvos):
    resultados = []

    for alvo in alvos:
        p = prob_atingir(dados, x_atual, alvo)
        ev = valor_esperado_continuar(x_atual, alvo, p)
        resultados.append((alvo, p, ev))

    return sorted(resultados, key=lambda x: x[2], reverse=True)

# =========================================================
# INTERFACE
# =========================================================

st.title("✈️ Aviator — Análise Condicional em Tempo Real")

aba1, aba2 = st.tabs([
    "📊 Visão Geral",
    "🧠 Decisão Condicional"
])

# =========================================================
# ABA 1 — VISÃO GERAL (ENXUTA)
# =========================================================

with aba1:
    c1, c2, c3 = st.columns(3)

    c1.metric("Total de jogos", f"{len(dados):,}")
    c2.metric("Multiplicador médio", formatar(dados.mean()))
    c3.metric("Máximo histórico", formatar(dados.max()))

    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(dados, bins=80)
    ax.set_title("Distribuição dos Multiplicadores Finais")
    st.pyplot(fig)

# =========================================================
# ABA 2 — DECISÃO CONDICIONAL (AVIATOR REAL)
# =========================================================

with aba2:
    st.subheader("Estado Atual do Jogo")

    x_atual = st.number_input(
        "Multiplicador atual do jogo",
        min_value=1.01,
        value=1.50,
        step=0.01
    )

    cond = distribuicao_condicional(dados, x_atual)

    if len(cond) < 50:
        st.error("⚠️ Poucos dados históricos acima desse valor. Decisão instável.")
        st.stop()

    st.caption(
        f"Base estatística: {len(cond)} jogos históricos chegaram a pelo menos {x_atual:.2f}x"
    )

    # Alvos típicos do Aviator
    alvos = [
        round(x_atual + 0.2, 2),
        round(x_atual + 0.5, 2),
        round(x_atual + 1.0, 2),
        2.0, 3.0, 5.0, 10.0
    ]

    alvos = sorted(set([a for a in alvos if a > x_atual]))

    rows = []

    for alvo in alvos:
        p = prob_atingir(dados, x_atual, alvo)
        ev = valor_esperado_continuar(x_atual, alvo, p)

        rows.append({
            "Cashout alvo": f"{alvo:.2f}x",
            "Prob. de atingir": f"{p*100:.1f}%",
            "Valor esperado": formatar(ev)
        })

    df_decisao = pd.DataFrame(rows)
    st.dataframe(df_decisao, use_container_width=True)

    # Melhor decisão
    melhor = melhor_cashout(dados, x_atual, alvos)[0]

    st.markdown("---")

    if melhor[2] > 0:
        st.success(
            f"✅ Melhor decisão estatística: **cashout em {melhor[0]:.2f}x** "
            f"(EV = {formatar(melhor[2])})"
        )
    else:
        st.error(
            "❌ Nenhum cashout acima deste ponto apresenta valor esperado positivo.\n\n"
            "**Decisão racional: NÃO entrar ou sair imediatamente.**"
        )

    # Regras claras para o jogador
    st.markdown(
        """
        ### 📌 Interpretação prática
        - **Prob. de atingir**: chance real baseada em milhares de jogos
        - **Valor esperado**:
            - positivo → decisão racional
            - negativo → cassino tem vantagem
        - Se **todos os EV forem negativos**, o melhor movimento é **não jogar**
        """
    )
