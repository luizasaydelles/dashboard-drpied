"""
DRPIED v3 - mesma simulação da v2 (HODMD + diferencial percentual), com os
gráficos da v1 (forma de onda, zoom na perda, fasor) e da v2 (função 87)
reunidos em uma única figura.

Layout:
    linha 1 : forma de onda completa de SV1 com as perdas         (v1)
    linha 2 : zoom na perda + erro  |  módulo do fasor             (v1)
    linha 3 : Idiff(t)              |  Ires(t)                     (v2)
    linha 4 : plano Ires × Idiff    |  sinais de trip              (v2)
"""

import argparse

import matplotlib.pyplot as plt

from drpied_hodmd import F_NOM, executar_drpied, plotar, resumo
from drpied_hodmd_v2 import S1, avaliar_87, gerar_dois_streams, plotar_87, resumo_87


def plotar_tudo(out, res, fase="a", pickup=0.2, slope=0.25):
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(4, 2)

    ax_v1 = [fig.add_subplot(gs[0, :]),
             fig.add_subplot(gs[1, 0]),
             fig.add_subplot(gs[1, 1])]
    ax_v2 = [[fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
             [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]]

    plotar(out, fase=f"I{fase}1", ax=ax_v1)
    plotar_87(out, res, fase, pickup, slope, ax=ax_v2)

    n_perd = int((~out["recebido"]).sum())
    smp_ini = out.loc[~out["recebido"], "smpCnt"].iloc[0] if n_perd else "-"
    fig.suptitle(f"DRPIED - perda de {n_perd} pacote(s) a partir de smpCnt {smp_ini} (SV1)",
                 fontsize=13)
    fig.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser(description="DRPIED v3 - gráficos da v1 e da v2 juntos")
    p.add_argument("--smpcnt", type=int, default=800, help="smpCnt do primeiro pacote perdido (SV1)")
    p.add_argument("--n", type=int, default=9, help="quantidade de pacotes perdidos")
    p.add_argument("--ciclos", type=int, default=20)
    p.add_argument("--smpcnt0", type=int, default=0)
    p.add_argument("--fase", type=float, default=0.0, help="ângulo de fase comum [graus]")
    p.add_argument("--freq", type=float, default=F_NOM)
    p.add_argument("--ruido", type=float, default=0.0, help="ruído gaussiano RMS [pu]")
    p.add_argument("--theta-max", type=int, default=9, help="máx. perdas reparadas (<=0: sem limite)")
    p.add_argument("--fallback", choices=["zeros", "linear"], default="zeros")
    p.add_argument("--pickup", type=float, default=0.2, help="corrente mínima de operação [pu]")
    p.add_argument("--slope", type=float, default=0.25, help="inclinação da característica (0-1)")
    p.add_argument("--plot-fase", choices=["a", "b", "c"], default="a", help="fase mostrada nos gráficos")
    p.add_argument("--sem-plot", action="store_true")
    a = p.parse_args()

    df = gerar_dois_streams(a.ciclos, a.freq, fase_graus=a.fase,
                            smpcnt_inicial=a.smpcnt0, ruido_rms=a.ruido)
    out, drp = executar_drpied(df, a.smpcnt, a.n, fases=S1,
                               theta_max=a.theta_max if a.theta_max > 0 else None,
                               fallback=a.fallback)
    resumo(out, drp, fases=S1)

    res = avaliar_87(out, a.pickup, a.slope)
    resumo_87(out, res, a.pickup, a.slope)

    if not a.sem_plot:
        plotar_tudo(out, res, a.plot_fase, a.pickup, a.slope)


if __name__ == "__main__":
    main()
