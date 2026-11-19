"""
Dashboard DRPIED (HODMD) + proteção diferencial ANSI 87.

Executar:
    streamlit run dashboard_drpied.py

Página principal : controles do sinal e da perda de pacotes, reparo por HODMD.
Página ANSI 87   : habilitada por checkbox; característica diferencial percentual.

Os controles do sinal ficam na barra lateral e valem para as duas páginas.
Qualquer alteração reexecuta a simulação na hora (atualização quente). Com
`.streamlit/config.toml` (runOnSave = true), salvar o código também recarrega o app.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from drpied_hodmd import SPC, executar_drpied, gerar_sinais
from drpied_hodmd_v2 import fasor_dft

# Executado com "python dashboard_drpied.py" (ex.: botão Run do VS Code):
# relança pelo Streamlit, que é quem abre o servidor e o navegador.
if __name__ == "__main__" and not st.runtime.exists():
    import sys
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(stcli.main())

st.set_page_config(page_title="DRPIED - HODMD", page_icon="⚡", layout="wide")

FASES = ["a", "b", "c"]
S1 = [f"I{ph}1" for ph in FASES]
S2 = [f"I{ph}2" for ph in FASES]
CORES = {"referencia": "#2ca02c", "sem_drpied": "#7f7f7f", "com_drpied": "#d62728"}
CENARIOS = {"referencia": "Referência (sem perda)",
            "sem_drpied": "Sem DRPIED (zeros)",
            "com_drpied": "Com DRPIED"}

# Valores iniciais dos widgets. Copiar o estado a cada execução evita que o
# Streamlit descarte os valores da página 87 quando ela não está visível.
PADROES = {"f_nom": 60, "ciclos": 20, "i_rms": 1.0, "smp_ini": 800, "n_perd": 9,
           "hab87": False, "limitar9": True, "fallback": "zeros", "ruido": 0.0,
           "pickup": 0.2, "ires0": 1.0, "slope": 25.0}
for k, v in PADROES.items():
    st.session_state[k] = st.session_state.get(k, v)


# ----------------------------------------------------------------------------
# Simulação
# ----------------------------------------------------------------------------
@st.cache_data(max_entries=128, show_spinner=False)
def simular(f_nom, ciclos, i_rms, smp_ini, n_perd, theta_max, fallback, ruido):
    base = gerar_sinais(ciclos, f=f_nom, i_rms=i_rms, f_nom=f_nom)
    rng = np.random.default_rng(0)
    df = base[["smpCnt", "t"]].copy()
    for ph, c1, c2 in zip(["Ia", "Ib", "Ic"], S1, S2):
        df[c1] = base[ph] + ruido * i_rms * rng.standard_normal(len(df))
        df[c2] = base[ph] + ruido * i_rms * rng.standard_normal(len(df))
    out, drp = executar_drpied(df, smp_ini, n_perd, fases=S1, f_nom=f_nom,
                               theta_max=theta_max, fallback=fallback)
    return out, drp.eventos


def caracteristica(ires, pickup, ires0, slope):
    """Idiff de operação: pickup até Ires0, depois cresce com a inclinação."""
    return np.where(ires <= ires0, pickup, pickup + slope * (ires - ires0))


def calcular_87(out, i_base, pickup, ires0, slope):
    res = {}
    for cen in CENARIOS:
        for ph, c1, c2 in zip(FASES, S1, S2):
            if cen == "referencia":
                i1 = out[c1].to_numpy()
            elif cen == "sem_drpied":
                i1 = np.nan_to_num(out[f"{c1}_rx"].to_numpy())
            else:
                i1 = np.nan_to_num(out[f"{c1}_rep"].to_numpy())
            I1 = fasor_dft(i1) / i_base
            I2 = fasor_dft(out[c2].to_numpy()) / i_base
            idiff = np.abs(I1 - I2)
            ires = np.abs(I1 + I2) / 2
            lim = caracteristica(ires, pickup, ires0, slope)
            res[(cen, ph)] = (idiff, ires, lim, np.nan_to_num(idiff) > np.nan_to_num(lim, nan=np.inf))
    return res


def layout(fig, titulo, altura=330, **kw):
    kw = {"hovermode": "x unified", **kw}
    fig.update_layout(title=titulo, height=altura, margin=dict(l=10, r=10, t=45, b=10),
                      legend=dict(orientation="h", y=-0.22), **kw)
    return fig


# ----------------------------------------------------------------------------
# Barra lateral (compartilhada pelas duas páginas)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Sinal")
    st.radio("Frequência nominal [Hz]", [50, 60], key="f_nom", horizontal=True,
             help="Sempre 80 amostras por ciclo (4000 ou 4800 Hz).")
    f_nom = st.session_state.f_nom
    max_ciclos = f_nom  # 1 s de dados: o smpCnt não dá a volta (0 ... f·80-1)
    st.session_state.ciclos = min(st.session_state.ciclos, max_ciclos)
    st.number_input("Quantidade de ciclos", min_value=4, max_value=max_ciclos, step=1, key="ciclos")
    st.number_input("Corrente RMS [A]", min_value=0.001, step=0.1, format="%.3f", key="i_rms")

    st.header("Perda de pacotes (SV1)")
    total = int(st.session_state.ciclos * SPC)
    st.session_state.smp_ini = min(st.session_state.smp_ini, total - 2)
    st.slider("smpCnt do primeiro pacote perdido", 0, total - 2, key="smp_ini")
    max_n = total - st.session_state.smp_ini - 1
    st.session_state.n_perd = min(st.session_state.n_perd, max_n)
    st.slider("Pacotes perdidos (θ)", 1, max(max_n, 2), key="n_perd")

    with st.expander("Avançado"):
        st.checkbox("Limitar reparo a θ ≤ 9 (paper)", key="limitar9",
                    help="Acima disso o paper não repara. Desmarcado: extrapola com o HODMD.")
        st.selectbox("Fallback quando não repara", ["zeros", "linear"], key="fallback")
        st.number_input("Ruído gaussiano [pu do RMS]", 0.0, 0.2, step=0.001,
                        format="%.3f", key="ruido")

    st.divider()
    st.checkbox("Habilitar página da proteção ANSI 87", key="hab87")

out, eventos = simular(f_nom, int(st.session_state.ciclos), float(st.session_state.i_rms),
                       int(st.session_state.smp_ini), int(st.session_state.n_perd),
                       9 if st.session_state.limitar9 else None,
                       st.session_state.fallback, float(st.session_state.ruido))
t_ms = out["t"].to_numpy() * 1e3
perd = ~out["recebido"].to_numpy()
idx_perd = np.flatnonzero(perd)


# ----------------------------------------------------------------------------
# Página principal
# ----------------------------------------------------------------------------
def pagina_principal():
    st.title("DRPIED – reparo de Sampled Values por HODMD")
    st.caption(f"{f_nom} Hz · 80 amostras/ciclo · fs = {f_nom * SPC} Hz · "
               f"{st.session_state.ciclos} ciclos · Δt HODMD = 10·Ts = {10e3 / (f_nom * SPC):.3f} ms")

    ev = eventos[0] if eventos else None
    erro = out.loc[perd, "Ia1_err%"].max() if perd.any() else 0.0
    c = st.columns(6)
    c[0].metric("θ (pacotes perdidos)", int(perd.sum()))
    c[1].metric("Modo", ev["modo"] if ev else "-")
    c[2].metric("f dominante", f"{abs(ev['f_dom']):.3f} Hz" if ev and np.isfinite(ev["f_dom"]) else "-")
    c[3].metric("δ dominante", f"{ev['delta_dom']:.2e}" if ev and np.isfinite(ev["delta_dom"]) else "-")
    c[4].metric("Condition check", ("OK" if ev["condicao_ok"] else "Falhou") if ev else "-")
    c[5].metric("Erro máx. (fase a)", f"{erro:.4f} %" if np.isfinite(erro) else "não reparado")

    if ev and ev["modo"] == "bloqueio":
        st.warning(f"θ = {ev['theta']} > 9: o reparo foi bloqueado e as amostras foram preenchidas "
                   f"com '{st.session_state.fallback}'. Desmarque o limite em *Avançado* para extrapolar.")
    if (out["status"] == "perdida").any():
        st.info("A perda vai até o fim dos dados: nenhum pacote chegou depois dela, então não houve reparo.")

    # Forma de onda completa
    fig = go.Figure()
    fig.add_scatter(x=t_ms, y=out["Ia1"], mode="lines", name="Sinal verdadeiro",
                    line=dict(color=CORES["referencia"], width=1.2))
    fig.add_scatter(x=t_ms[perd], y=out["Ia1"][perd], mode="markers", name="SVs perdidos",
                    marker=dict(symbol="x", color="red", size=7))
    st.plotly_chart(layout(fig, "Ia – SV1", xaxis_title="t [ms]", yaxis_title="I [A]"),
                    width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        if idx_perd.size:
            z = slice(max(idx_perd[0] - 120, 0), min(idx_perd[-1] + 120, len(out)))
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_scatter(x=t_ms[z], y=out["Ia1"].to_numpy()[z], mode="lines", name="Verdadeiro",
                            line=dict(color=CORES["referencia"]))
            fig.add_scatter(x=t_ms[z], y=out["Ia1_rx"].to_numpy()[z], mode="markers", name="Recebido",
                            marker=dict(color="#1f77b4", size=4))
            fig.add_scatter(x=t_ms[idx_perd], y=out["Ia1_rep"].to_numpy()[idx_perd], mode="markers",
                            name="Reparado", marker=dict(color="red", size=9, symbol="circle-open"))
            fig.add_scatter(x=t_ms[z], y=out["Ia1_err%"].to_numpy()[z], mode="lines", name="Erro [%]",
                            line=dict(color="magenta", width=1), secondary_y=True)
            fig.update_yaxes(title_text="I [A]", secondary_y=False)
            fig.update_yaxes(title_text="erro [%]", secondary_y=True)
            st.plotly_chart(layout(fig, "Zoom na perda", xaxis_title="t [ms]"), width="stretch")
    with col2:
        fig = go.Figure()
        for cen, col in [("referencia", "Ia1"), ("sem_drpied", "Ia1_rx"), ("com_drpied", "Ia1_rep")]:
            x = np.nan_to_num(out[col].to_numpy())
            fig.add_scatter(x=t_ms, y=np.abs(fasor_dft(x)), mode="lines", name=CENARIOS[cen],
                            line=dict(color=CORES[cen], dash={"referencia": "solid",
                                      "sem_drpied": "dash", "com_drpied": "dot"}[cen], width=2))
        st.plotly_chart(layout(fig, "Módulo do fasor (DFT de 1 ciclo)", xaxis_title="t [ms]",
                               yaxis_title="|I| [A RMS]"), width="stretch")

    if ev and ev["hodmd"] is not None:
        with st.expander("Modos HODMD (formato das Tabelas II/III do paper)"):
            st.dataframe(ev["hodmd"].tabela(0), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------
# Página ANSI 87
# ----------------------------------------------------------------------------
def pagina_87():
    st.title("Proteção diferencial percentual – ANSI 87")
    st.caption("Idiff = |I1 − I2| · Ires = |I1 + I2| / 2 · valores em pu da corrente RMS informada. "
               "Opera se Idiff > característica.")

    c = st.columns(3)
    c[0].number_input("Corrente diferencial mínima – Idiff> [pu]", 0.01, 5.0, step=0.05,
                      format="%.2f", key="pickup")
    c[1].number_input("Corrente de restrição do ponto de quebra – Ires0 [pu]", 0.0, 10.0, step=0.1,
                      format="%.2f", key="ires0")
    c[2].number_input("Slope [%]", 0.0, 100.0, step=1.0, format="%.1f", key="slope")
    pickup, ires0, slope = st.session_state.pickup, st.session_state.ires0, st.session_state.slope / 100

    res = calcular_87(out, st.session_state.i_rms, pickup, ires0, slope)
    fase = st.radio("Fase exibida", FASES, horizontal=True, format_func=str.upper)

    m = st.columns(3)
    for col, cen in zip(m, CENARIOS):
        idiff, _, lim, trip = res[(cen, fase)]
        col.metric(CENARIOS[cen], "TRIP" if trip.any() else "Sem trip",
                   f"Idiff máx = {np.nanmax(idiff):.3f} pu",
                   delta_color="inverse" if trip.any() else "off")

    estilo = {"referencia": "solid", "sem_drpied": "dash", "com_drpied": "dot"}
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for cen in CENARIOS:
            fig.add_scatter(x=t_ms, y=res[(cen, fase)][0], mode="lines", name=CENARIOS[cen],
                            line=dict(color=CORES[cen], dash=estilo[cen], width=2))
        fig.add_scatter(x=t_ms, y=res[("referencia", fase)][2], mode="lines", name="Limiar (referência)",
                        line=dict(color="black", width=1))
        st.plotly_chart(layout(fig, "Corrente diferencial", xaxis_title="t [ms]", yaxis_title="Idiff [pu]"),
                        width="stretch")
    with col2:
        fig = go.Figure()
        for cen in CENARIOS:
            fig.add_scatter(x=t_ms, y=res[(cen, fase)][1], mode="lines", name=CENARIOS[cen],
                            line=dict(color=CORES[cen], dash=estilo[cen], width=2))
        st.plotly_chart(layout(fig, "Corrente de restrição", xaxis_title="t [ms]", yaxis_title="Ires [pu]"),
                        width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        ires_max = np.nanmax([np.nanmax(res[(c_, fase)][1]) for c_ in CENARIOS])
        xr = np.linspace(0, max(1.5, ires0 * 1.5, ires_max * 1.3), 300)
        yr = caracteristica(xr, pickup, ires0, slope)
        ymax = max(0.6, pickup * 2.5, np.nanmax(res[("sem_drpied", fase)][0]) * 1.2)
        fig = go.Figure()
        fig.add_scatter(x=np.r_[xr, xr[::-1]], y=np.r_[yr, np.full_like(xr, ymax)], fill="toself",
                        fillcolor="rgba(214,39,40,0.08)", line=dict(width=0), name="Região de operação",
                        hoverinfo="skip")
        fig.add_scatter(x=xr, y=yr, mode="lines", name="Característica", line=dict(color="black"))
        for cen in CENARIOS:
            idiff, ires, _, _ = res[(cen, fase)]
            fig.add_scatter(x=ires, y=idiff, mode="lines+markers", name=CENARIOS[cen],
                            marker=dict(size=3), line=dict(color=CORES[cen], dash=estilo[cen]))
        st.plotly_chart(layout(fig, "Plano Ires × Idiff", xaxis_title="Ires [pu]", yaxis_title="Idiff [pu]",
                               xaxis_range=[0, xr.max()], yaxis_range=[0, ymax], hovermode="closest"),
                        width="stretch")
    with col2:
        fig = go.Figure()
        for i, cen in enumerate(CENARIOS):
            fig.add_scatter(x=t_ms, y=res[(cen, fase)][3].astype(int) + 1.5 * i, mode="lines",
                            name=CENARIOS[cen], line=dict(color=CORES[cen], shape="hv", width=2))
        fig.update_yaxes(tickvals=[0, 1.5, 3], ticktext=["Ref.", "Zeros", "DRPIED"])
        st.plotly_chart(layout(fig, "Sinais de trip", xaxis_title="t [ms]"), width="stretch")

    linhas = []
    for (cen, ph), (idiff, _, lim, trip) in res.items():
        linhas.append({"Cenário": CENARIOS[cen], "Fase": ph.upper(),
                       "Idiff máx [pu]": np.nanmax(idiff),
                       "Idiff / limiar máx": np.nanmax(idiff / lim),
                       "Trip": "SIM" if trip.any() else "não",
                       "1º trip [ms]": t_ms[np.argmax(trip)] if trip.any() else None,
                       "Duração do trip [ms]": trip.sum() * 1e3 / (f_nom * SPC)})
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------
paginas = [st.Page(pagina_principal, title="DRPIED / HODMD", icon="📈", default=True)]
if st.session_state.hab87:
    paginas.append(st.Page(pagina_87, title="Proteção ANSI 87", icon="🛡️", url_path="ansi87"))
st.navigation(paginas).run()
