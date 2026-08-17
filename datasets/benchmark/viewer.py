"""Interactive viewer for the ADM1 benchmark time series (numpy + matplotlib).

Browse the train / test series in two views (toggle with 'v'):

* **overview** — the 5 sensors, the substrate feed (switch times marked), and a row
  with the FOS/TAC acidification indicator (VFA/TAC, with the Nordmann thresholds) plus
  four key hidden states, each with the UKF reference overlaid (test set only).
* **states** — a paged grid of *all 41* ADM1 states (truth + UKF ±2σ); page through
  them with the up/down keys. This is the "look at every state" view.

    python viewer.py                          # interactive
    python viewer.py --dataset test           # start on the test set
    python viewer.py --view states            # start in the all-states browser
    python viewer.py --dataset test --view states --state-page 2 --save p2.png

Keys: <-/-> switch series · 'v' toggle overview/states · up/down page states ·
      't' toggle train/test.
"""

from __future__ import annotations

import argparse
import math

import loader
import numpy as np

CHANNELS = loader.CHANNELS  # ["Q_gas", "Q_ch4", "Q_co2", "pH", "TS"]
CHANNEL_UNITS = ["m³/d", "m³/d", "m³/d", "-", "% TS"]
# Key hidden states in the overview's third row. S_ac is not shown directly — the
# FOS/TAC panel (which it dominates) takes its place, as the interpretable health view.
STATE_PANELS = [(5, "S_pro"), (8, "S_ch4"), (10, "S_nh4"), (27, "X_ac")]

# --- FOS/TAC (= VFA / TAC), the acidification indicator ---------------------
# Faithful numpy port of pyadm1's vfa_torch / tac_torch (verified to match): these
# are fixed acid/COD physical constants, unaffected by the per-series kinetic
# perturbation, so the viewer stays numpy-only (no torch / pyadm1 needed).
_FT_M_HAC = 60.0
_FT_COD = {"ac": 64.0, "pro": 112.0, "bu": 160.0, "va": 208.0}
_FT_KA = {
    "IN": 1.7422628075231628e-09,
    "co2": 5.275470941922822e-07,
    "ac": 1.7378008287493764e-05,
    "pro": 1.3182567385564074e-05,
    "bu": 1.5135612484362071e-05,
    "va": 1.3803842646028839e-05,
}
_FT_I = {
    "S_va": 3,
    "S_bu": 4,
    "S_pro": 5,
    "S_ac": 6,
    "S_co2": 9,
    "S_nh4": 10,
    "S_cation": 29,
    "S_anion": 30,
    "S_va_ion": 31,
    "S_bu_ion": 32,
    "S_pro_ion": 33,
    "S_ac_ion": 34,
    "S_hco3": 35,
    "S_nh3": 36,
}
# Nordmann acidification thresholds (FOS/TAC): healthy < 0.3, warning ~0.4, critical > 0.6.
FOSTAC_THRESHOLDS = [(0.3, "gesund"), (0.4, "Warnung"), (0.6, "kritisch")]
# Physical floor on the TAC denominator [kg CaCO3/m³]. A working digester always has
# some buffer, so a near-zero alkalinity is an unphysical *estimate* (the UKF's TAC
# occasionally dives toward 0 in acidified series) — dividing VFA by it makes FOS/TAC
# explode to millions, a divide-by-~0 artefact rather than a real error. The true TAC
# never drops below ~8 here, so this floor never clips a physical value; it only tames
# an unphysical estimate. (pyadm1's raw clamp is 1e-6, which does not guard against it.)
FOSTAC_TAC_FLOOR = 1.0


def fostac(states: np.ndarray) -> np.ndarray:
    """FOS/TAC (VFA / TAC) for a ``(T, 41)`` (or ``(41,)``) ADM1 state array.

    VFA = total volatile fatty acids [kg HAc-eq/m³]; TAC = total alkalinity
    [kg CaCO3/m³] (titration endpoint pH 5). Numpy port of pyadm1's vfa/tac, with the
    denominator floored at :data:`FOSTAC_TAC_FLOOR` so an unphysical near-zero TAC
    estimate cannot blow the ratio up (see that constant).
    """
    x = np.asarray(states, float)
    g = lambda n: x[..., _FT_I[n]]
    vfa = _FT_M_HAC * (
        g("S_ac") / _FT_COD["ac"]
        + g("S_pro") / _FT_COD["pro"]
        + g("S_bu") / _FT_COD["bu"]
        + g("S_va") / _FT_COD["va"]
    )
    h = 1.0e-5  # H+ at the pH-5 titration endpoint
    a = {k: _FT_KA[k] / (h + _FT_KA[k]) for k in _FT_KA}
    total_n = g("S_nh4") + g("S_nh3")
    tac_mol = (
        (g("S_nh3") - a["IN"] * total_n)
        + (g("S_hco3") - a["co2"] * g("S_co2"))
        + (g("S_ac_ion") / _FT_COD["ac"] - a["ac"] * g("S_ac") / _FT_COD["ac"])
        + (g("S_pro_ion") / _FT_COD["pro"] - a["pro"] * g("S_pro") / _FT_COD["pro"])
        + (g("S_bu_ion") / _FT_COD["bu"] - a["bu"] * g("S_bu") / _FT_COD["bu"])
        + (g("S_va_ion") / _FT_COD["va"] - a["va"] * g("S_va") / _FT_COD["va"])
        + g("S_anion")
        - g("S_cation")
    )
    return vfa / np.clip(50.0 * tac_mol, FOSTAC_TAC_FLOOR, None)


# all-states browser layout
STATE_COLS = 4
STATE_ROWS = 4
STATES_PER_PAGE = STATE_COLS * STATE_ROWS


def _load():
    """Return the two datasets as lists of per-series dicts, substrate + state names."""
    meta = loader.load_meta()
    subs = sorted(meta["substrates"].items(), key=lambda kv: kv[1]["index"])
    sub_names = [name for name, _ in subs]

    imap = meta.get("state_index_map", {})
    n_state = meta.get("state_size", 41)
    state_names = [str(imap.get(str(i), f"state_{i}")) for i in range(n_state)]

    tr = loader.load_train()
    train = [
        {
            "time": tr["time"][i],
            "measurements": tr["measurements"][i],
            "feed_noisy": tr["feed_noisy"][i],
            "states": tr["states"][i],
            "switch_days": tr["switch_days"][i],
            "regime": str(tr["regime"][i]) if "regime" in tr else "",
            **({"fostac": tr["fostac"][i]} if "fostac" in tr else {}),
        }
        for i in range(len(tr["time"]))
    ]
    test = loader.load_test()
    return {"train": train, "test": test}, sub_names, state_names


def _series(datasets, name, i):
    lst = datasets[name]
    return lst[i % len(lst)], i % len(lst), len(lst)


def _switches(ax, sw):
    for d in np.atleast_1d(sw):
        ax.axvline(float(d), color="0.7", ls=":", lw=0.8)


def _has_ukf(s):
    return "ukf_x_hat" in s and not s.get("ukf_pending", False)


def _draw_overview(fig, s, sub_names):
    from matplotlib.gridspec import GridSpec

    t = s["time"]
    sw = np.atleast_1d(s["switch_days"])
    gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 1.1])

    for c in range(5):  # row 1: sensors
        ax = fig.add_subplot(gs[0, c])
        ax.plot(t, s["measurements"][:, c], color=f"C{c}", lw=0.9)
        _switches(ax, sw)
        ax.set_title(f"{CHANNELS[c]} [{CHANNEL_UNITS[c]}]", fontsize=9)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.set_ylabel("gemessen", fontsize=8)

    axf = fig.add_subplot(gs[1, :])  # row 2: feed
    for j, sname in enumerate(sub_names):
        axf.plot(t, s["feed_noisy"][:, j], lw=1.1, label=sname)
    _switches(axf, sw)
    axf.set_ylabel("Feed [m³/d]", fontsize=8)
    axf.set_title(
        "Substrat-Input (verrauscht); gepunktet = Substratwechsel", fontsize=9
    )
    axf.legend(fontsize=7, ncol=len(sub_names), loc="upper right")
    axf.tick_params(labelsize=7)

    has_ukf = _has_ukf(s)  # row 3: FOS/TAC health indicator + key hidden states

    axft = fig.add_subplot(gs[2, 0])  # panel 0: FOS/TAC (replaces the S_ac panel)
    ft_true = fostac(s["states"])
    axft.plot(t, ft_true, "k-", lw=1.6, label="Truth")
    if has_ukf:
        axft.plot(t, fostac(s["ukf_x_hat"]), "C3-", lw=1.0, label="UKF")
    # The laboratory titration is stored hourly; show it at the interval an
    # operator would realistically sample, so the panel conveys how sparse the
    # measurement actually is next to the continuous truth.
    if "fostac" in s:
        from fostac import SAMPLE_EVERY_DAYS, sample_indices

        lab = np.asarray(s["fostac"], dtype=float)
        idx = sample_indices(len(lab), every_days=SAMPLE_EVERY_DAYS)
        idx = idx[lab[idx, 0] > 0.0]  # drop below-detection titrations
        axft.plot(
            t[idx],
            lab[idx, 0] / np.maximum(lab[idx, 1], 1e-9),
            "C0o",
            ms=4,
            mfc="none",
            label=f"Labor ({SAMPLE_EVERY_DAYS:.0f} d)",
        )
    for val, lbl in FOSTAC_THRESHOLDS:
        axft.axhline(val, color="0.6", ls="--", lw=0.7)
        axft.text(
            t[-1], val, f" {lbl}", fontsize=6, color="0.4", va="center", ha="right"
        )
    _switches(axft, sw)
    axft.set_title("FOS/TAC = VFA/TAC", fontsize=9)
    axft.set_xlabel("Zeit [d]", fontsize=8)
    # Scale to the truth's range; a poor UKF estimate then clips at the top (visibly
    # off the chart) instead of an unphysical spike dominating the axis.
    axft.set_ylim(0, max(0.8, 1.2 * float(np.nanmax(ft_true))))
    axft.tick_params(labelsize=7)
    axft.legend(fontsize=7)

    for k, (si, sl) in enumerate(STATE_PANELS):
        ax = fig.add_subplot(gs[2, k + 1])
        ax.plot(t, s["states"][:, si], "k-", lw=1.6, label="Truth")
        if has_ukf:
            std = s["ukf_std"][:, si]
            ax.plot(t, s["ukf_x_hat"][:, si], "C3-", lw=1.0, label="UKF")
            ax.fill_between(
                t,
                s["ukf_x_hat"][:, si] - 2 * std,
                s["ukf_x_hat"][:, si] + 2 * std,
                color="C3",
                alpha=0.15,
            )
        _switches(ax, sw)
        ax.set_title(f"{sl} (idx {si})", fontsize=9)
        ax.set_xlabel("Zeit [d]", fontsize=8)
        ax.tick_params(labelsize=7)


def _draw_states(fig, s, state_names, page):
    """Paged grid of all states: truth + UKF ±2σ. Returns (page, n_pages)."""
    from matplotlib.gridspec import GridSpec

    t = s["time"]
    sw = np.atleast_1d(s["switch_days"])
    n_state = s["states"].shape[1]
    n_pages = max(1, math.ceil(n_state / STATES_PER_PAGE))
    page %= n_pages
    start = page * STATES_PER_PAGE
    idxs = range(start, min(start + STATES_PER_PAGE, n_state))
    has_ukf = _has_ukf(s)

    gs = GridSpec(STATE_ROWS, STATE_COLS, figure=fig)
    for k, si in enumerate(idxs):
        ax = fig.add_subplot(gs[k // STATE_COLS, k % STATE_COLS])
        ax.plot(t, s["states"][:, si], "k-", lw=1.4, label="Truth")
        if has_ukf:
            std = s["ukf_std"][:, si]
            ax.plot(t, s["ukf_x_hat"][:, si], "C3-", lw=0.9, label="UKF")
            ax.fill_between(
                t,
                s["ukf_x_hat"][:, si] - 2 * std,
                s["ukf_x_hat"][:, si] + 2 * std,
                color="C3",
                alpha=0.15,
            )
        _switches(ax, sw)
        nm = state_names[si] if si < len(state_names) else f"state_{si}"
        ax.set_title(f"{nm} (idx {si})", fontsize=8)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=7)
    return page, n_pages


def draw(fig, datasets, sub_names, state_names, st):
    fig.clear()
    s, idx, n = _series(datasets, st["name"], st["idx"])
    st["idx"] = idx
    reg = f" · {s['regime']}" if s.get("regime") else ""
    n_sw = len(np.atleast_1d(s["switch_days"]))

    if st["view"] == "states":
        page, n_pages = _draw_states(fig, s, state_names, st["page"])
        st["page"] = page
        head = f"alle Zustände · Seite {page + 1}/{n_pages}"
    else:
        _draw_overview(fig, s, sub_names)
        head = "Übersicht (Sensoren · Feed · Schlüsselzustände)"

    fig.suptitle(
        f"{st['name'].upper()} Reihe {idx + 1}/{n}{reg} · {head}   "
        f"({len(s['time'])} Schritte, {n_sw} Wechsel)   "
        f"[v: Ansicht · ↑/↓: Seite · ←/→: Reihe · t: train/test]",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["train", "test"], default="train")
    ap.add_argument("--series", type=int, default=0)
    ap.add_argument("--view", choices=["overview", "states"], default="overview")
    ap.add_argument(
        "--state-page", type=int, default=0, help="[states view] page to show"
    )
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets, sub_names, state_names = _load()
    st = {
        "name": args.dataset,
        "idx": args.series,
        "view": args.view,
        "page": args.state_page,
    }
    fig = plt.figure(figsize=(16, 9))

    def redraw():
        draw(fig, datasets, sub_names, state_names, st)
        fig.canvas.draw_idle()

    if args.save:
        draw(fig, datasets, sub_names, state_names, st)
        fig.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"Wrote {args.save}")
        return 0

    def on_key(event):
        if event.key in ("right", "n"):
            st["idx"] += 1
        elif event.key in ("left", "p"):
            st["idx"] -= 1
        elif event.key == "down":
            st["page"] += 1
        elif event.key == "up":
            st["page"] -= 1
        elif event.key == "v":
            st["view"] = "states" if st["view"] == "overview" else "overview"
        elif event.key == "t":
            st["name"] = "test" if st["name"] == "train" else "train"
            st["idx"] = 0
        else:
            return
        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    print(
        "Keys: <-/-> Reihe · 'v' Ansicht (Übersicht/alle Zustände) · up/down Seite · 't' train/test"
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
