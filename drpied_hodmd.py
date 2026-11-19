"""
DRPIED - Data-Repairing for Protection IED (implementação offline).

Baseado em:
    M. K. Katoulaei, C. M. Adrah, H. K. Høidalen, "Implementation and Testing a
    Robust Data-Repairing Method for Protection IEDs on Process Bus",
    IEEE Trans. Power Delivery, vol. 39, no. 6, pp. 3153-3163, Dec. 2024.

Adaptações em relação ao paper:
    - 60 Hz, 80 amostras/ciclo  ->  fs = 4800 Hz, smpCnt de 0 a 4799.
    - Execução offline sobre um DataFrame (padrão: 20 ciclos), mas o
      processamento é feito amostra a amostra, como num subscriber SV real.
    - Perda de pacotes definida por (smpCnt inicial, quantidade perdida).

Fluxo (Fig. 7 do paper):
    1. Captura/buffer circular de 80 amostras por stream.
    2. Detecção de perda: theta = smpCnt_new - smpCnt_pre - 1  (eq. 15).
    3. Seleção de K = 8 amostras espaçadas de 10 amostras (Δt = 10·Ts).
    4. HODMD (eqs. 4-14) -> modos, amplitudes, δm, ωm.
    5. Condition check: f dominante ≈ f nominal (ou DC) e δ dominante < 1.
    6. Cálculo das amostras faltantes pela expansão DMD (eq. 3).
    7. (Avaliação) Fasor por DFT de ciclo completo, equivalente à RDFT do 87T.
"""

import argparse
import time
from collections import OrderedDict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

F_NOM = 60.0                      # frequência nominal [Hz]
SPC = 80                          # amostras por ciclo
FS = F_NOM * SPC                  # 4800 Hz
TS = 1.0 / FS                     # 208.33 µs
SMPCNT_MAX = int(round(FS))       # smpCnt varia de 0 a SMPCNT_MAX-1
FASES = ["Ia", "Ib", "Ic"]


# ----------------------------------------------------------------------------
# Geração dos sinais e da perda de pacotes
# ----------------------------------------------------------------------------
def gerar_sinais(n_ciclos=20, f=F_NOM, i_rms=1.0, fase_graus=0.0,
                 smpcnt_inicial=0, ruido_rms=0.0, seed=None, f_nom=F_NOM):
    """DataFrame com smpCnt, t e as três correntes (mesmo valor RMS e mesma fase).
    `f_nom` define a taxa de amostragem (f_nom · 80) e o ciclo do smpCnt."""
    n = int(n_ciclos * SPC)
    k = np.arange(n)
    t = k / (f_nom * SPC)
    sinal = np.sqrt(2) * i_rms * np.cos(2 * np.pi * f * t + np.deg2rad(fase_graus))
    rng = np.random.default_rng(seed)

    smp_max = int(round(f_nom * SPC))
    df = pd.DataFrame({"smpCnt": (smpcnt_inicial + k) % smp_max, "t": t})
    for ph in FASES:
        df[ph] = sinal + ruido_rms * rng.standard_normal(n)
    return df


def mascara_perda(df, smpcnt_perda, n_perdidos):
    """Vetor booleano: True = pacote recebido, False = pacote perdido."""
    idx = np.flatnonzero(df["smpCnt"].to_numpy() == smpcnt_perda)
    if idx.size == 0:
        raise ValueError(f"smpCnt {smpcnt_perda} não existe no DataFrame.")
    recebido = np.ones(len(df), dtype=bool)
    recebido[idx[0]:idx[0] + n_perdidos] = False
    return recebido


# ----------------------------------------------------------------------------
# HODMD (Seção III do paper)
# ----------------------------------------------------------------------------
@dataclass
class ResultadoHODMD:
    modos: np.ndarray      # (J, M) complexo: a_m·q_m no espaço físico
    delta: np.ndarray      # (M,) taxa de crescimento δm [1/s]
    omega: np.ndarray      # (M,) frequência angular ωm [rad/s]
    lambdas: np.ndarray    # (M,) autovalores discretos λm

    @property
    def amplitude(self):
        return np.linalg.norm(self.modos, axis=0)

    @property
    def dominante(self):
        return int(np.argmax(self.amplitude))

    def reconstruir(self, tau):
        """Eq. (3): i(t) = Σ a_m q_m exp((δm + jωm)(t - t1)), com tau = t - t1."""
        expo = np.exp(np.outer(self.delta + 1j * self.omega, np.atleast_1d(tau)))
        return np.real(self.modos @ expo)

    def tabela(self, fase=0):
        """Tabela no formato das Tabelas II/III do paper (modos por amplitude)."""
        ordem = np.argsort(self.amplitude)[::-1]
        return pd.DataFrame({
            "m": np.arange(1, ordem.size + 1),
            "amqm(real)": self.modos[fase, ordem].real,
            "amqm(imag)": self.modos[fase, ordem].imag,
            "delta_m": self.delta[ordem],
            "omega_m": self.omega[ordem],
            "f_m [Hz]": self.omega[ordem] / (2 * np.pi),
        })


def hodmd(V, dt, d=None, tol_svd=1e-4):
    """
    HODMD de uma matriz de snapshots V (J x K) amostrada com passo dt.

    d       : ordem da DMD (paper: d = K/2).
    tol_svd : truncamento relativo dos valores singulares (filtra ruído).
    """
    J, K = V.shape
    d = K // 2 if d is None else d
    n_col = K - d + 1

    # Matriz de snapshots atrasados (eq. 6): (d·J) x (K-d+1)
    Vd = np.vstack([V[:, i:i + n_col] for i in range(d)])
    X, Y = Vd[:, :-1], Vd[:, 1:]

    # SVD de X e truncamento (eq. 8)
    U, s, Th = np.linalg.svd(X, full_matrices=False)
    if s[0] <= 0:
        return None
    r = max(1, int(np.sum(s > tol_svd * s[0])))
    U, s, Th = U[:, :r], s[:r], Th[:r, :]

    # Koopman reduzido (eq. 9), autodecomposição (eq. 10) e modos (eq. 11)
    Y_T_Sinv = (Y @ Th.conj().T) / s
    R_hat = U.conj().T @ Y_T_Sinv
    lam, W = np.linalg.eig(R_hat)
    Q = Y_T_Sinv @ W

    # Amplitudes (eq. 13). O paper usa a = Q⁻¹·i1; aqui é usado o ajuste por
    # mínimos quadrados sobre todos os snapshots, que é mais robusto a ruído.
    A = np.vstack([Q * lam ** k for k in range(n_col)])
    a, *_ = np.linalg.lstsq(A, Vd.T.reshape(-1).astype(complex), rcond=None)

    # δm + jωm = log(λm)/Δt (eq. 14)
    with np.errstate(divide="ignore"):
        s_cont = np.log(lam.astype(complex)) / dt

    return ResultadoHODMD(modos=Q[:J, :] * a, delta=s_cont.real,
                          omega=s_cont.imag, lambdas=lam)


# ----------------------------------------------------------------------------
# DRPIED: subscriber com buffer, detecção de perda e reparo
# ----------------------------------------------------------------------------
class DRPIED:
    def __init__(self, n_fases=3, f_nom=F_NOM, amostras_ciclo=SPC, K=8, passo=10,
                 tam_buffer=80, theta_max=9, tol_svd=1e-4, tol_freq_hz=0.5,
                 delta_max=1.0, fallback="zeros", usar_reparadas=False):
        self.n_fases = n_fases
        self.f_nom = f_nom
        self.ts = 1.0 / (f_nom * amostras_ciclo)
        self.smp_max = int(round(f_nom * amostras_ciclo))
        self.K = K
        self.passo = passo
        self.tam_buffer = tam_buffer
        self.theta_max = theta_max          # None = sem limite
        self.tol_svd = tol_svd
        self.tol_freq_hz = tol_freq_hz
        self.delta_max = delta_max
        self.fallback = fallback            # "zeros" (paper) ou "linear"
        self.usar_reparadas = usar_reparadas

        self.buffer = OrderedDict()         # smpCnt -> (valores, recebida)
        self.smp_pre = None
        self.eventos = []

    # --- buffer -------------------------------------------------------------
    def _guardar(self, smp, valores, recebida):
        self.buffer[smp] = (valores, recebida)
        self.buffer.move_to_end(smp)
        while len(self.buffer) > self.tam_buffer:
            self.buffer.popitem(last=False)

    def _amostra(self, smp, extra=None):
        if extra is not None and smp == extra[0]:
            return extra[1]
        item = self.buffer.get(smp)
        if item is None or (not item[1] and not self.usar_reparadas):
            return None
        return item[0]

    def _selecionar(self, ancora, extra=None):
        """K amostras espaçadas de `passo`, terminando em `ancora`."""
        grade = [(ancora - self.passo * j) % self.smp_max for j in range(self.K)][::-1]
        cols = [self._amostra(g, extra) for g in grade]
        if any(c is None for c in cols):
            return None
        return grade, np.column_stack(cols)

    # --- condition check (Seção V-C.1) ----------------------------------------
    def _condicao(self, res):
        m = res.dominante
        f = abs(res.omega[m]) / (2 * np.pi)
        f_ok = abs(f - self.f_nom) <= self.tol_freq_hz or f <= self.tol_freq_hz
        return bool(f_ok and res.delta[m] < self.delta_max)

    # --- processamento ------------------------------------------------------
    def receber(self, smp, valores):
        """Processa um pacote SV. Retorna lista de (smpCnt, valores, status)."""
        valores = np.asarray(valores, dtype=float)
        saida = []
        if self.smp_pre is not None:
            theta = (smp - self.smp_pre - 1) % self.smp_max      # eq. (15)
            if theta >= self.smp_max // 2:                       # pacote atrasado/duplicado
                return saida
            if theta > 0:
                saida.extend(self._reparar(smp, valores, theta))
        self._guardar(smp, valores, True)
        self.smp_pre = smp
        saida.append((smp, valores, "recebida"))
        return saida

    def _reparar(self, smp_new, val_new, theta):
        faltantes = [(self.smp_pre + 1 + i) % self.smp_max for i in range(theta)]
        ev = {"smpCnt_ini": faltantes[0], "theta": theta, "modo": "bloqueio",
              "f_dom": np.nan, "delta_dom": np.nan, "condicao_ok": False,
              "tempo_us": np.nan, "hodmd": None}

        sel = None
        if self.theta_max is None or theta <= self.theta_max:
            # Interpolação: grade termina no pacote recém-chegado. Com passo 10,
            # isso só é possível para theta <= 9 (limite reportado no paper).
            sel = self._selecionar(smp_new, extra=(smp_new, val_new))
            ev["modo"] = "interpolacao"
            if sel is None:
                sel = self._selecionar(self.smp_pre)
                ev["modo"] = "extrapolacao"
            if sel is None:
                ev["modo"] = "sem_historico"

        estimados = None
        if sel is not None:
            grade, V = sel
            t0 = time.perf_counter()
            res = hodmd(V, self.passo * self.ts, d=self.K // 2, tol_svd=self.tol_svd)
            if res is not None:
                ev["hodmd"] = res
                ev["f_dom"] = res.omega[res.dominante] / (2 * np.pi)
                ev["delta_dom"] = res.delta[res.dominante]
                ev["condicao_ok"] = self._condicao(res)
                if ev["condicao_ok"]:
                    tau = np.array([(m - grade[0]) % self.smp_max for m in faltantes]) * self.ts
                    estimados = res.reconstruir(tau)
            ev["tempo_us"] = (time.perf_counter() - t0) * 1e6

        if estimados is None:
            status = f"fallback_{self.fallback}"
            if self.fallback == "linear":
                v_pre = self.buffer[self.smp_pre][0]
                frac = np.arange(1, theta + 1) / (theta + 1)
                estimados = v_pre[:, None] + np.outer(val_new - v_pre, frac)
            else:
                estimados = np.zeros((self.n_fases, theta))
        else:
            status = "reparada"

        self.eventos.append(ev)
        saida = []
        for i, smp in enumerate(faltantes):
            vals = estimados[:, i]
            self._guardar(smp, vals, False)
            saida.append((smp, vals, status))
        return saida


# ----------------------------------------------------------------------------
# Execução offline e avaliação
# ----------------------------------------------------------------------------
def executar_drpied(df, smpcnt_perda, n_perdidos, fases=FASES, **kw_drpied):
    """Simula a chegada dos pacotes SV do DataFrame com perda e aplica o DRPIED."""
    recebido = mascara_perda(df, smpcnt_perda, n_perdidos)
    X = df[fases].to_numpy(dtype=float)
    pos = {int(s): i for i, s in enumerate(df["smpCnt"])}

    rep = np.where(recebido[:, None], X, np.nan)
    status = np.where(recebido, "recebida", "perdida").astype(object)

    drp = DRPIED(n_fases=len(fases), **kw_drpied)
    for i in np.flatnonzero(recebido):
        for smp, vals, st in drp.receber(int(df["smpCnt"].iat[i]), X[i]):
            j = pos[smp]
            rep[j] = vals
            status[j] = st

    out = df.copy()
    out["recebido"] = recebido
    out["status"] = status
    rms = np.sqrt(np.mean(X ** 2, axis=0))
    for k, ph in enumerate(fases):
        out[f"{ph}_rx"] = np.where(recebido, X[:, k], np.nan)
        out[f"{ph}_rep"] = rep[:, k]
        # erro normalizado pelo RMS do sinal verdadeiro (como no paper)
        out[f"{ph}_err%"] = np.where(recebido, 0.0, np.abs(rep[:, k] - X[:, k]) / rms[k] * 100)
    return out, drp


def fasor_dft(x, n=SPC):
    """Módulo RMS do fasor por DFT de ciclo completo (equivalente à RDFT)."""
    w = np.sqrt(2) / n * np.exp(-2j * np.pi * np.arange(n) / n)
    y = np.convolve(x, w[::-1])[:len(x)]
    y[:n - 1] = np.nan
    return np.abs(y)


def resumo(out, drp, fases=FASES):
    print(f"fs = {FS:.0f} Hz | Ts = {TS * 1e6:.2f} us | Δt HODMD = {drp.passo * TS * 1e3:.4f} ms")
    for ev in drp.eventos:
        print("\n--- Evento de perda ---")
        print(f"smpCnt inicial = {ev['smpCnt_ini']} | theta = {ev['theta']} | modo = {ev['modo']}")
        print(f"f dominante = {ev['f_dom']:.4f} Hz | δ dominante = {ev['delta_dom']:.3e} "
              f"| condition check = {ev['condicao_ok']} | tempo HODMD = {ev['tempo_us']:.1f} us")
        if ev["hodmd"] is not None:
            print("\nModos HODMD (fase " + fases[0] + "):")
            print(ev["hodmd"].tabela(0).to_string(index=False, float_format=lambda v: f"{v: .6e}"))

    mask = ~out["recebido"]
    print("\nErro máximo nas amostras reparadas [% do RMS]:")
    for ph in fases:
        print(f"  {ph}: {out.loc[mask, f'{ph}_err%'].max():.4f} %")
    nao = (out["status"] == "perdida").sum()
    if nao:
        print(f"Atenção: {nao} amostra(s) perdida(s) no fim do DataFrame não puderam ser "
              "reparadas (nenhum pacote posterior chegou).")


def plotar(out, fase="Ia", ax=None):
    """Plota forma de onda, zoom na perda e fasor. Se `ax` (3 eixos) for dado,
    desenha neles e não chama plt.show()."""
    t = out["t"].to_numpy() * 1e3
    perdidas = ~out["recebido"].to_numpy()
    idx = np.flatnonzero(perdidas)

    mostrar = ax is None
    if mostrar:
        fig, ax = plt.subplots(3, 1, figsize=(11, 10))

    ax[0].plot(t, out[fase], color="tab:green", lw=1, label="Sinal verdadeiro")
    ax[0].plot(t[perdidas], out.loc[perdidas, fase], "rx", label="SVs perdidos")
    ax[0].set(title=f"{fase} - {len(out) // SPC} ciclos", xlabel="t [ms]", ylabel="I [pu]")
    ax[0].legend(loc="upper right"); ax[0].grid(alpha=.3)

    if idx.size:
        z = slice(max(idx[0] - 120, 0), min(idx[-1] + 120, len(out)))
        ax[1].plot(t[z], out[fase].to_numpy()[z], color="tab:green", lw=1.2, label="Verdadeiro")
        ax[1].plot(t[z], out[f"{fase}_rx"].to_numpy()[z], "o", color="tab:blue", ms=3, label="Recebido")
        ax[1].plot(t[idx], out[f"{fase}_rep"].to_numpy()[idx], "o", mfc="none", color="tab:red",
                   ms=7, label="Reparado (DRPIED)")
        ax1b = ax[1].twinx()
        ax1b.plot(t[z], out[f"{fase}_err%"].to_numpy()[z], color="magenta", lw=1)
        ax1b.set_ylabel("erro [%]", color="magenta")
        ax[1].set(title="Zoom na perda", xlabel="t [ms]", ylabel="I [pu]")
        ax[1].legend(loc="upper left"); ax[1].grid(alpha=.3)

    x_true = out[fase].to_numpy()
    x_zero = np.nan_to_num(out[f"{fase}_rx"].to_numpy())
    x_rep = np.nan_to_num(out[f"{fase}_rep"].to_numpy())
    ax[2].plot(t, fasor_dft(x_true), color="tab:green", label="Verdadeiro")
    ax[2].plot(t, fasor_dft(x_zero), "--", color="tab:gray", label="Perdas preenchidas com zero")
    ax[2].plot(t, fasor_dft(x_rep), ":", color="tab:red", lw=2, label="Reparado (DRPIED)")
    ax[2].set(title="Módulo do fasor (DFT de 1 ciclo)", xlabel="t [ms]", ylabel="|I| [pu RMS]")
    ax[2].legend(loc="lower right"); ax[2].grid(alpha=.3)

    if mostrar:
        fig.tight_layout()
        plt.show()


def main():
    p = argparse.ArgumentParser(description="DRPIED/HODMD offline - 60 Hz, 80 amostras/ciclo")
    p.add_argument("--smpcnt", type=int, default=800, help="smpCnt do primeiro pacote perdido")
    p.add_argument("--n", type=int, default=9, help="quantidade de pacotes perdidos")
    p.add_argument("--ciclos", type=int, default=20)
    p.add_argument("--smpcnt0", type=int, default=0, help="smpCnt da primeira amostra do DataFrame")
    p.add_argument("--fase", type=float, default=0.0, help="ângulo de fase comum [graus]")
    p.add_argument("--freq", type=float, default=F_NOM, help="frequência real do sinal [Hz]")
    p.add_argument("--ruido", type=float, default=0.0, help="ruído gaussiano RMS [pu]")
    p.add_argument("--theta-max", type=int, default=9, help="máx. perdas reparadas (<=0: sem limite)")
    p.add_argument("--fallback", choices=["zeros", "linear"], default="zeros")
    p.add_argument("--csv", help="salvar o DataFrame de saída em CSV")
    p.add_argument("--sem-plot", action="store_true")
    a = p.parse_args()

    df = gerar_sinais(n_ciclos=a.ciclos, f=a.freq, fase_graus=a.fase,
                      smpcnt_inicial=a.smpcnt0, ruido_rms=a.ruido, seed=0)
    out, drp = executar_drpied(df, a.smpcnt, a.n,
                               theta_max=a.theta_max if a.theta_max > 0 else None,
                               fallback=a.fallback)
    resumo(out, drp)
    if a.csv:
        out.to_csv(a.csv, index=False)
    if not a.sem_plot:
        plotar(out)


if __name__ == "__main__":
    main()
