"""
DRPIED v2 - reparo de SV por HODMD + função diferencial percentual (87).

Dois streams SV (Fig. 8 do paper):
    - SV1 (CT1): sofre a perda de pacotes;
    - SV2 (CT2): íntegro.

Função diferencial percentual, por fase, sobre fasores de 1 ciclo:
    Idiff = |I1 - I2|
    Ires  = |I1 + I2| / 2
    Opera se Idiff > max(Ipickup, slope · Ires)   (característica de inclinação única)

São comparados três cenários:
    - referência : SV1 sem perda;
    - sem DRPIED : amostras perdidas substituídas por zero;
    - com DRPIED : amostras perdidas reparadas pelo HODMD.

O núcleo (geração de sinais, HODMD e DRPIED) é importado de drpied_hodmd.py.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drpied_hodmd import F_NOM, SPC, TS, executar_drpied, gerar_sinais, resumo

FASES = ["a", "b", "c"]
S1 = [f"I{ph}1" for ph in FASES]
S2 = [f"I{ph}2" for ph in FASES]
CENARIOS = {"referencia": "Referência (sem perda)",
            "sem_drpied": "Sem DRPIED (zeros)",
            "com_drpied": "Com DRPIED"}


# ----------------------------------------------------------------------------
# Sinais
# ----------------------------------------------------------------------------
def gerar_dois_streams(n_ciclos=20, f=F_NOM, i_rms=1.0, fase_graus=0.0,
                       smpcnt_inicial=0, ruido_rms=0.0, seed=0):
    """SV1 e SV2 com mesmo RMS e mesma fase (carga passante: Idiff = 0)."""
    base = gerar_sinais(n_ciclos, f, i_rms, fase_graus, smpcnt_inicial, 0.0)
    rng = np.random.default_rng(seed)
    df = base[["smpCnt", "t"]].copy()
    for ph, c1, c2 in zip(["Ia", "Ib", "Ic"], S1, S2):
        df[c1] = base[ph] + ruido_rms * rng.standard_normal(len(df))
        df[c2] = base[ph] + ruido_rms * rng.standard_normal(len(df))
    return df


# ----------------------------------------------------------------------------
# Função diferencial percentual
# ----------------------------------------------------------------------------
def fasor_dft(x, n=SPC):
    """Fasor complexo (valor RMS) por DFT de ciclo completo (equivalente à RDFT)."""
    w = np.sqrt(2) / n * np.exp(-2j * np.pi * np.arange(n) / n)
    y = np.convolve(x, w[::-1])[:len(x)]
    y[:n - 1] = np.nan
    return y


def diferencial_percentual(i1, i2, pickup=0.2, slope=0.25):
    """
    i1, i2 : amostras instantâneas (N,) das correntes de um par de fases.
    Retorna Idiff, Ires, limiar e sinal de operação (por amostra).
    """
    I1, I2 = fasor_dft(i1), fasor_dft(i2)
    idiff = np.abs(I1 - I2)
    ires = np.abs(I1 + I2) / 2
    limiar = np.maximum(pickup, slope * ires)
    trip = np.nan_to_num(idiff) > np.nan_to_num(limiar, nan=np.inf)
    return idiff, ires, limiar, trip


def avaliar_87(out, pickup=0.2, slope=0.25):
    """Calcula Idiff/Ires/trip para os três cenários e as três fases."""
    res = {}
    for cen in CENARIOS:
        for ph, c1, c2 in zip(FASES, S1, S2):
            if cen == "referencia":
                i1 = out[c1].to_numpy()
            elif cen == "sem_drpied":
                i1 = np.nan_to_num(out[f"{c1}_rx"].to_numpy())
            else:
                i1 = np.nan_to_num(out[f"{c1}_rep"].to_numpy())
            res[(cen, ph)] = diferencial_percentual(i1, out[c2].to_numpy(), pickup, slope)
    return res


def resumo_87(out, res, pickup, slope):
    t = out["t"].to_numpy()
    print(f"\n=== Função diferencial percentual (Ipickup = {pickup} pu, slope = {slope * 100:.0f} %) ===")
    linhas = []
    for (cen, ph), (idiff, ires, lim, trip) in res.items():
        margem = np.nanmax(idiff / lim)
        linhas.append({
            "cenário": CENARIOS[cen], "fase": ph,
            "Idiff máx [pu]": np.nanmax(idiff),
            "Idiff/limiar máx": margem,
            "trip": "SIM" if trip.any() else "não",
            "t 1º trip [ms]": t[np.argmax(trip)] * 1e3 if trip.any() else np.nan,
            "duração trip [ms]": trip.sum() * TS * 1e3,
        })
    print(pd.DataFrame(linhas).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def plotar_87(out, res, fase="a", pickup=0.2, slope=0.25, ax=None):
    """Plota a função 87. Se `ax` (matriz 2x2 de eixos) for dado, desenha nele
    e não chama plt.show()."""
    t = out["t"].to_numpy() * 1e3
    estilos = {"referencia": dict(color="tab:green", lw=1.5),
               "sem_drpied": dict(color="tab:gray", ls="--", lw=1.5),
               "com_drpied": dict(color="tab:red", ls=":", lw=2)}

    mostrar = ax is None
    if mostrar:
        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax = np.asarray(ax)
    for cen, est in estilos.items():
        idiff, ires, lim, trip = res[(cen, fase)]
        ax[0, 0].plot(t, idiff, label=CENARIOS[cen], **est)
        ax[0, 1].plot(t, ires, label=CENARIOS[cen], **est)
        ax[1, 0].plot(ires, idiff, label=CENARIOS[cen], **est)
        ax[1, 1].step(t, trip.astype(int) + 1.2 * list(estilos).index(cen),
                      where="post", label=CENARIOS[cen], **est)

    lim_ref = res[("referencia", fase)][2]
    ax[0, 0].plot(t, lim_ref, color="k", lw=1, label="Limiar de operação")
    ax[0, 0].set(title=f"Idiff = |I1 - I2| (fase {fase})", xlabel="t [ms]", ylabel="pu")
    ax[0, 1].set(title=f"Ires = |I1 + I2|/2 (fase {fase})", xlabel="t [ms]", ylabel="pu")

    xr = np.linspace(0, max(1.5, np.nanmax(res[("referencia", fase)][1]) * 1.3), 200)
    ax[1, 0].fill_between(xr, np.maximum(pickup, slope * xr), xr.max(), color="tab:red", alpha=.08)
    ax[1, 0].plot(xr, np.maximum(pickup, slope * xr), "k", lw=1, label="Característica")
    ax[1, 0].set(title="Plano Ires × Idiff", xlabel="Ires [pu]", ylabel="Idiff [pu]",
                 xlim=(0, xr.max()), ylim=(0, max(0.6, pickup * 2)))

    ax[1, 1].set(title="Sinal de trip (deslocado por cenário)", xlabel="t [ms]")
    ax[1, 1].set_yticks([])
    for a in ax.flat:
        a.grid(alpha=.3)
        a.legend(fontsize=8, loc="upper right")
    if mostrar:
        fig.tight_layout()
        plt.show()


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="DRPIED v2 - HODMD + diferencial percentual (60 Hz, 80 a/c)")
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
    p.add_argument("--csv", help="salvar DataFrame de saída em CSV")
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

    if a.csv:
        for (cen, ph), (idiff, ires, _, trip) in res.items():
            out[f"Idiff_{ph}_{cen}"] = idiff
            out[f"Ires_{ph}_{cen}"] = ires
            out[f"trip_{ph}_{cen}"] = trip
        out.to_csv(a.csv, index=False)
    if not a.sem_plot:
        plotar_87(out, res, "a", a.pickup, a.slope)


if __name__ == "__main__":
    main()
