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
    page_title="Probabilidade Condicional — Aviator",
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
# PROBABILIDADE CONDICIONAL (CORE AVIATOR)
# =========================================================

def distribuicao_condicional(dados, x_atual):
    return dados[dados >= x_atual]


def prob_atingir(dados, x_atual, alvo):
    cond = distribuicao_condicional(dados, x_atual)
    if len(cond) == 0 or alvo <= x_atual:
        return 0.0
    return np.mean(cond >= alvo)


def valor_esperado_continuar(x_atual, alvo, prob):
    ganho = alvo - x_atual
    perda = x_atual
    return prob * ganho - (1 - prob) * perda


def tabela_decisao(dados, x_atual, alvos):
    linhas = []

    for alvo in alvos:
        p = prob_atingir(dados, x_atual, alvo)
        ev = valor_esperado_continuar(x_atual, alvo, p)
        linhas.append({
            "Cashout alvo": f"{alvo:.2f}x",
            "Prob. de atingir": p,
            "Valor esperado": ev
        })

    df = pd.DataFrame(linhas)
    return df.sort_values("Valor esperado", ascending=False)

# =========================================================
# INTERFACE
# =========================================================

st.title("✈️ Análise Condicional — Aviator")

aba1, aba2 = st.tabs([
    "📊 Dados & Análises",
    "🧠 Decisão Condicional"
])

# =========================================================
# ABA 1 — DADOS & ANÁLISES
# =========================================================

with aba1:
    st.subheader("Base Estatística Utilizada")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de jogos", f"{len(dados):,}")
    c2.metric("Multiplicador médio", formatar(dados.mean()))
    c3.metric("Máximo histórico", formatar(dados.max()))

    st.markdown("### Distribuição dos multiplicadores finais")

    fig, ax = plt.subplots(figsize=(9,4))
    ax.hist(dados, bins=80)
    ax.set_xlabel("Multiplicador final")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)

    st.markdown(
        """
        **Como esses dados são usados:**

        - Cada valor representa o **multiplicador final de um jogo**
        - A análise condicional usa **apenas jogos que passaram pelo valor atual**
        - Isso garante que a decisão seja baseada no **estado real do jogo**
        """
    )

# =========================================================
# ABA 2 — DECISÃO CONDICIONAL
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
        st.error("⚠️ Base estatística insuficiente acima desse valor.")
        st.stop()

    st.caption(
        f"{len(cond)} jogos históricos chegaram a pelo menos {x_atual:.2f}x"
    )

    # Alvos possíveis
    alvos = sorted(set([
        round(x_atual + 0.2, 2),
        round(x_atual + 0.5, 2),
        round(x_atual + 1.0, 2),
        2.0, 3.0, 5.0, 10.0
    ]))

    alvos = [a for a in alvos if a > x_atual]

    df_decisao = tabela_decisao(dados, x_atual, alvos)

    # Exibição formatada
    df_show = df_decisao.copy()
    df_show["Prob. de atingir"] = df_show["Prob. de atingir"].apply(lambda x: f"{x*100:.1f}%")
    df_show["Valor esperado"] = df_show["Valor esperado"].apply(formatar)

    st.markdown("### Probabilidades condicionais e valor esperado")
    st.dataframe(df_show, use_container_width=True)

    # DECISÃO FINAL DE ENTRADA
    melhor = df_decisao.iloc[0]
    ev_max = melhor["Valor esperado"]

    st.markdown("---")
    st.subheader("📌 Decisão Final")

    if ev_max > 0:
        st.success(
            f"✅ **APOSTAR AGORA**\n\n"
            f"Melhor cashout estatístico: **{melhor['Cashout alvo']}**\n\n"
            f"Valor esperado positivo: **{formatar(ev_max)}**"
        )
    else:
        st.error(
            "❌ **NÃO APOSTAR AGORA**\n\n"
            "Nenhum cenário acima deste ponto apresenta valor esperado positivo.\n\n"
            "A decisão racional é **não entrar** ou **sair imediatamente**."
        )

    st.markdown(
        """
        **Como interpretar a decisão:**
        - A entrada só é recomendada se **existe EV positivo**
        - O cashout indicado é o **ponto estatisticamente ótimo**
        - Se o EV for negativo, o jogo favorece o cassino
        """
    )
