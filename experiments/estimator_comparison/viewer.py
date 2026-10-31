"""Interactive comparison of the four estimators on the 60-day test set.

    python -m experiments.estimator_comparison.viewer

Four views, cycled with ``v``:

``overview``
    Per-state accuracy over the whole test set, all estimators at once. Answers which
    states any of them resolves at all. A dot plot, not bars: the accuracies span
    several orders of magnitude, so the axis has to be logarithmic, and a bar length
    measured from a baseline a log axis does not have would misrepresent the ratios.
``table``
    The same numbers as ``overview``, written out, best estimator per state marked.
    Nothing here has to be read off a colour.
``error``
    For the selected state, how the error develops over the 60 days: median across the
    20 series with the interquartile band. Shows whether an estimator is biased,
    drifts, or only stumbles at the load switches.
``series``
    For the selected series and state, every estimate against the truth, with the
    residuals below.

Keys: ``left``/``right`` series, ``up``/``down`` state, ``v`` view, ``s`` sort by an
estimator, ``q`` quit.

Units
-----
Every view names the unit of the selected state, from
:data:`~.comparison.STATE_UNITS`. Only ``series`` plots the state itself, so only there
does the unit belong on the value axis; ``overview``, ``table`` and ``error`` all show
dimensionless quantities, and there the unit is attached to the state's own label
instead of to the number. Worth keeping straight: three of the units in this model
are not interchangeable -- ``kg COD/m³`` for the organics, ``kmol/m³`` for the strong
ions, and ``bar`` for the gas phase -- so a ranking across states compares
normalised errors, never magnitudes.
"""

from __future__ import annotations

import argparse

import numpy as np

from .comparison import (
    BURNIN_STEPS,
    COLORS,
    MARKERS,
    REPO,
    SOURCES,
    STYLES,
    Comparison,
    load,
)

#: Where the overview's accuracy axis stops. An estimate worse than this is already
#: useless, so the exact figure only has to be legible, not placed on the scale.
OFF_SCALE_PCT = 300.0


def _decorate(ax) -> None:
    """The recessive grid every panel shares."""
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def _burnin_band(ax, cmp_: Comparison) -> None:
    """Shade the window both PINN evaluations exclude from their scores."""
    ax.axvspan(0, cmp_.time[BURNIN_STEPS], color="0.92", zorder=0)


def _sort_note(sort_by: str | None) -> str:
    """How the states are ordered, for the header."""
    return f"sortiert nach {sort_by}" if sort_by else "Zustandsreihenfolge"


def _state_order(cmp_: Comparison, sort_by: str | None) -> np.ndarray:
    """State indices, either in model order or ranked by one estimator."""
    if sort_by in cmp_.estimators:
        return np.argsort(cmp_.overall(sort_by))
    return np.arange(cmp_.n_states)


def draw_overview(fig, cmp_: Comparison, state: int, sort_by: str | None) -> str:
    """Per-state accuracy across the whole test set, as a dot plot."""
    ax = fig.add_subplot(111)
    order = _state_order(cmp_, sort_by)
    y = np.arange(cmp_.n_states)

    # One estimator misses S_CATION by five orders of magnitude. Letting it set the
    # axis would squeeze the other forty states into a third of the width, so the
    # scale stops at OFF_SCALE_PCT and anything beyond is drawn at the edge with its
    # value written out. Off the scale, not out of the picture.
    values = {est: cmp_.overall(est)[order] * 100 for est in cmp_.estimators}
    n_est = len(cmp_.estimators)
    for k, est in enumerate(cmp_.estimators):
        rows = y + (k - (n_est - 1) / 2) * 0.20
        vals = values[est]
        inside = vals <= OFF_SCALE_PCT

        # A faint profile through one estimator's dots, so its run down the states can
        # be followed by eye. The states are categories, so the slope between two of
        # them means nothing -- which is why the line stays behind the dots and light.
        # Off-scale values are pinned to the edge, where their marker sits; anything
        # a log axis cannot place breaks the line instead of bending it.
        profile = np.where(
            np.isfinite(vals) & (vals > 0), np.minimum(vals, OFF_SCALE_PCT), np.nan
        )
        ax.plot(
            profile,
            rows,
            color=COLORS.get(est),
            lw=1.1,
            alpha=0.35,
            zorder=1,
            solid_joinstyle="round",
        )
        ax.plot(
            vals[inside],
            rows[inside],
            linestyle="none",
            marker=MARKERS.get(est, "o"),
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            color=COLORS.get(est),
            label=est,
        )
        for value, row in zip(vals[~inside], rows[~inside]):
            ax.plot(
                OFF_SCALE_PCT,
                row,
                linestyle="none",
                marker=">",
                markersize=6,
                color=COLORS.get(est),
                clip_on=False,
            )
            ax.annotate(
                f" {value:,.0f}",
                (OFF_SCALE_PCT, row),
                fontsize=6.5,
                va="center",
                color="0.25",
                annotation_clip=False,
            )

    pos = int(np.where(order == state)[0][0])
    ax.axhspan(pos - 0.5, pos + 0.5, color="gold", alpha=0.2, zorder=0)

    ax.set_yticks(y)
    # NRMSE is dimensionless, so the unit describes the state, not the number on the
    # axis. It rides on the tick label rather than the x-label for exactly that reason.
    ax.set_yticklabels(
        [f"{cmp_.state_names[i]}  [{cmp_.unit(i)}]" for i in order], fontsize=6.5
    )
    ax.set_ylim(cmp_.n_states - 0.5, -0.5)
    ax.set_xscale("log")
    ax.set_xlim(right=OFF_SCALE_PCT)
    ax.set_xlabel("NRMSE [%], Median über die 20 Testreihen")
    _decorate(ax)
    ax.legend(loc="lower left", fontsize=9)

    return f"Genauigkeit je Zustand über den gesamten Testsatz ({_sort_note(sort_by)})"


def _fmt(pct: float) -> str:
    """A percentage that stays informative across five orders of magnitude.

    Two decimals turn the gas-pressure states, which the UKF hits to within a
    ten-thousandth of a percent, into a row of zeros.
    """
    return f"{pct:.2f}" if pct >= 0.01 else f"{pct:.1e}"


def draw_table(fig, cmp_: Comparison, state: int, sort_by: str | None) -> str:
    """The per-state accuracies as numbers, best estimator per state marked."""
    ax = fig.add_subplot(111)
    ax.axis("off")
    order = _state_order(cmp_, sort_by)

    values = np.stack([cmp_.overall(e) for e in cmp_.estimators])
    best = np.argmin(values, axis=0)

    rows = [
        [cmp_.state_names[i], cmp_.unit(i)]
        + [_fmt(100 * values[k, i]) for k in range(len(cmp_.estimators))]
        for i in order
    ]
    n_col = len(cmp_.estimators)
    table = ax.table(
        cellText=rows,
        colLabels=["Zustand", "Einheit", *cmp_.estimators],
        cellLoc="right",
        loc="upper center",
        colWidths=[0.16, 0.13] + [0.13] * n_col,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 0.9)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_edgecolor("0.85")
        if r == 0:
            cell.set_text_props(weight="bold")
        else:
            # Bold marks the winner of the row. Weight carries it on its own, so the
            # number stays a number instead of growing a symbol beside it.
            if c > 1 and best[order[r - 1]] == c - 2:
                cell.set_text_props(weight="bold")
            if order[r - 1] == state:
                cell.set_facecolor("gold")
                cell.set_alpha(0.3)
        if c <= 1:
            cell.set_text_props(ha="left")
        if c == 1 and r > 0:
            cell.set_text_props(color="0.35")

    return f"NRMSE [%] je Zustand, Median über 20 Reihen · fett = bester Schätzer ({_sort_note(sort_by)})"


def draw_error(fig, cmp_: Comparison, state: int, switch_days) -> str:
    """How one state's error develops over the 60 days, pooled over the series."""
    ax = fig.add_subplot(111)
    for day in np.unique(np.concatenate([np.atleast_1d(s) for s in switch_days])):
        ax.axvline(float(day), color="0.93", lw=0.8, zorder=0)

    for est in cmp_.estimators:
        med, lo, hi = cmp_.error_band(est, state)
        ax.plot(
            cmp_.time,
            med,
            lw=1.4,
            color=COLORS.get(est),
            linestyle=STYLES.get(est, "-"),
            label=est,
        )
        ax.fill_between(cmp_.time, lo, hi, color=COLORS.get(est), alpha=0.12, lw=0)

    ax.axhline(0.0, color="0.35", lw=1.0)
    _burnin_band(ax, cmp_)
    ax.set_xlim(0, cmp_.time[-1])
    ax.set_xlabel("Zeit [d]")
    # The ratio itself has no unit; naming the state's makes the denominator readable.
    ax.set_ylabel(
        f"Fehler / RMS von {cmp_.state_names[state]}  (beide in {cmp_.unit(state)})"
    )
    _decorate(ax)
    ax.legend(loc="upper right", fontsize=9, ncol=len(cmp_.estimators))

    return "Abweichung vom echten Zustand über 60 Tage · Median und Quartilsband über 20 Reihen"


def draw_series(fig, cmp_: Comparison, series: int, state: int, switch_days) -> str:
    """One series, one state: every estimate against the truth, plus the residuals."""
    top = fig.add_subplot(211)
    bottom = fig.add_subplot(212, sharex=top)

    truth = cmp_.truth[series, :, state]
    top.plot(cmp_.time, truth, color="0.15", lw=1.8, label="wahr", zorder=5)
    scale = np.sqrt(np.nanmean(truth[BURNIN_STEPS:] ** 2))

    for est in cmp_.estimators:
        est_series = cmp_.estimates[est][series, :, state]
        style = {
            "color": COLORS.get(est),
            "linestyle": STYLES.get(est, "-"),
            "alpha": 0.9,
        }
        top.plot(cmp_.time, est_series, lw=1.2, label=est, **style)
        bottom.plot(cmp_.time, (est_series - truth) / (scale + 1e-12), lw=1.1, **style)

    bottom.axhline(0.0, color="0.35", lw=1.0)
    for ax in (top, bottom):
        for day in np.atleast_1d(switch_days[series]):
            ax.axvline(float(day), color="0.88", lw=0.8, zorder=0)
        _burnin_band(ax, cmp_)
        _decorate(ax)
        ax.set_xlim(0, cmp_.time[-1])

    top.set_ylabel(cmp_.state_label(state))
    top.legend(loc="best", fontsize=8, ncol=len(cmp_.estimators) + 1)
    top.tick_params(labelbottom=False)
    bottom.set_ylabel("Fehler / RMS")
    bottom.set_xlabel("Zeit [d]")

    scores = "   ".join(
        f"{e} {100 * cmp_.nrmse[e][series, state]:.1f}%" for e in cmp_.estimators
    )
    return f"Schätzungen gegen den echten Zustand · NRMSE: {scores}"


def draw(fig, cmp_: Comparison, switch_days, st: dict) -> None:
    """Render the current view and the header that explains the controls."""
    fig.clear()
    st["series"] %= cmp_.n_series
    st["state"] %= cmp_.n_states

    if st["view"] == "overview":
        head = draw_overview(fig, cmp_, st["state"], st["sort_by"])
    elif st["view"] == "table":
        head = draw_table(fig, cmp_, st["state"], st["sort_by"])
    elif st["view"] == "error":
        head = draw_error(fig, cmp_, st["state"], switch_days)
    else:
        head = draw_series(fig, cmp_, st["series"], st["state"], switch_days)

    where = (
        f"Reihe {st['series'] + 1}/{cmp_.n_series} ({cmp_.series_names[st['series']]})"
        f" · Zustand {st['state'] + 1}/{cmp_.n_states} ({cmp_.state_label(st['state'])})"
    )
    warn = f"   [fehlt: {', '.join(cmp_.missing)}]" if cmp_.missing else ""
    fig.suptitle(
        f"{head}\n{where}{warn}   [v: Ansicht · ←/→: Reihe · ↑/↓: Zustand · s: Sortierung]",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])


VIEWS = ["overview", "table", "error", "series"]


def main() -> int:
    """Parse the arguments, load the runs, and either save a figure or open the UI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", choices=VIEWS, default="overview")
    ap.add_argument("--series", type=int, default=0)
    ap.add_argument("--state", type=int, default=0, help="state index, 0..40")
    ap.add_argument("--sort-by", default=None, help="rank the states by this estimator")
    ap.add_argument(
        "--save", default=None, help="write a PNG instead of opening the window"
    )
    args = ap.parse_args()

    cmp_ = load()
    if not cmp_.estimators:
        print("No estimator run found. Expected:")
        for label, (path, _) in SOURCES.items():
            print(f"  {label:12} {path}")
        return 1
    if cmp_.missing:
        print(f"Not loaded, file missing: {', '.join(cmp_.missing)}")
    print(
        f"Loaded {', '.join(cmp_.estimators)} on {cmp_.n_series} series x {cmp_.n_states} states"
    )

    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    switch_days = np.load(REPO / "datasets/benchmark/test.npz", allow_pickle=True)[
        "switch_days"
    ]
    st = {
        "view": args.view,
        "series": args.series,
        "state": args.state,
        "sort_by": args.sort_by,
    }
    fig = plt.figure(figsize=(15, 9))

    if args.save:
        draw(fig, cmp_, switch_days, st)
        fig.savefig(args.save, dpi=110, bbox_inches="tight")
        print(f"Wrote {args.save}")
        return 0

    def on_key(event):
        if event.key == "right":
            st["series"] += 1
        elif event.key == "left":
            st["series"] -= 1
        elif event.key == "down":
            st["state"] += 1
        elif event.key == "up":
            st["state"] -= 1
        elif event.key == "v":
            st["view"] = VIEWS[(VIEWS.index(st["view"]) + 1) % len(VIEWS)]
        elif event.key == "s":
            options = [None, *cmp_.estimators]
            st["sort_by"] = options[(options.index(st["sort_by"]) + 1) % len(options)]
        else:
            return
        draw(fig, cmp_, switch_days, st)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(fig, cmp_, switch_days, st)
    print("Keys: v Ansicht · ←/→ Reihe · ↑/↓ Zustand · s Sortierung")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
