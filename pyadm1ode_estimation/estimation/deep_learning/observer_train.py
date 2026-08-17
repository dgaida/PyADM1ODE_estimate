"""Offline pre-training for the amortised observer — two objectives.

The observer can be pre-trained two ways, and this module offers both behind a
consistent API:

* **Supervised** (:func:`pretrain_observer`) — on a **simulator** dataset, where
  the true 41-state is known. A scaled state MSE (each state normalised by its
  RMS magnitude) directly teaches the full state, including the many dimensions
  no sensor observes. Only the model can provide this signal.
* **Self-supervised** (:func:`pretrain_observer_selfsup`) — on **measurement
  data with no ground truth** (real plant history, or simulated windows for an
  ablation). It uses the *same* objective as the online fine-tuning — a
  measurement fit ``((h(x̂) − y)/σ)²`` plus a rate-scaled physics residual — so
  the observer can be primed directly on the real plant it will run on.

The recommended sim→real recipe (:func:`pretrain_observer_sim2real`) does both in
order: supervised on the simulator to learn the full-state structure, then
self-supervised on real history to close the sim-to-real gap.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from pyadm1.core.adm1_torch import Adm1TorchParams, adm1da_rhs_torch

from .observation_torch import TorchObservationModel
from .observer import Adm1Observer
from .observer_data import MeasurementDataset, ObserverDataset

_N_LIQUID = 37


@dataclass
class PretrainResult:
    """Outputs of :func:`pretrain_observer`."""

    history: dict[str, list[float]]
    train_idx: np.ndarray
    val_idx: np.ndarray
    scale: np.ndarray  # per-state RMS used for normalisation, shape (41,)
    best_epoch: int = -1  # epoch whose weights were restored (-1 = none)
    best_val: float = float("nan")  # its validation loss
    stopped_early: bool = False  # True if patience ran out before ``epochs``


def _scaled_state_loss(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    scale: torch.Tensor,
    burnin: int = 0,
) -> torch.Tensor:
    """Per-state-normalised MSE, optionally ignoring the first ``burnin`` steps.

    A causal observer cannot know the initial state, so the error in the first
    steps measures the unknowable rather than the model. Excluding it stops that
    from dominating the gradient.
    """
    if burnin:
        x_hat, x_true = x_hat[..., burnin:, :], x_true[..., burnin:, :]
    return (((x_hat - x_true) / scale) ** 2).mean()


def _noise_scales(
    dataset: ObserverDataset, noise_std: Sequence[float] | None
) -> torch.Tensor | None:
    """Per-feature noise magnitude in *normalised* feature units.

    ``noise_std`` is given in raw sensor units (what a datasheet quotes). The
    features are ``(raw - mean) / std``, an affine map, so adding noise ``eps``
    to a reading is adding ``eps / feat_std`` to its feature. Only the leading
    measurement block is perturbed; the feed columns are a known control input.
    """
    if noise_std is None:
        return None
    if dataset.channel_names is None:
        raise ValueError(
            "noise augmentation needs dataset.channel_names to locate the sensor "
            "columns; rebuild the dataset with the measurement block included."
        )
    sigma = np.asarray(noise_std, dtype=float)
    n_ch = len(dataset.channel_names)
    if sigma.shape != (n_ch,):
        raise ValueError(f"noise_std must have shape ({n_ch},), got {sigma.shape}.")
    scales = np.zeros(dataset.features.shape[-1], dtype=float)
    scales[:n_ch] = sigma / np.asarray(dataset.feat_std, dtype=float)[:n_ch]
    return torch.tensor(scales, dtype=torch.get_default_dtype())


def _selfsup_losses(
    x: torch.Tensor,
    meas_t: torch.Tensor,
    valid: torch.Tensor,
    obs_model: TorchObservationModel,
    sigma: torch.Tensor,
    params: Adm1TorchParams,
    scale: torch.Tensor,
    dt: float,
    res_clip: float | None = None,
    with_meas: bool = True,
    with_phys: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The self-supervised (measurement + rate-scaled physics) loss terms.

    Shared by online fine-tuning and self-supervised pre-training. ``x`` is
    ``(..., T, 41)`` (a single window or a batch of windows); the time axis is
    the second-to-last, so the same code handles ``(T, 41)`` and ``(B, T, 41)``.
    ``scale`` is the ``(37,)`` liquid magnitude used as the physics-rate floor.

    ``res_clip`` optionally caps each standardised residual to ``[-c, c]`` (a
    Huber-like robustification). The ill-conditioned biogas map can otherwise
    produce a single huge ``Q_gas`` residual whose gradient step overshoots into
    the region where the quasi-steady gas solve diverges; capping the residual
    bounds that gradient so a far-off (e.g. off-distribution) start can still
    take stable steps.

    ``with_meas`` / ``with_phys`` skip a term entirely (returning a constant 0).
    This is not the same as a zero weight: the stiff acid-base RHS backward is
    **not gradient-finite** far from the training manifold, and ``0 · NaN`` still
    poisons the combined gradient — so a disabled term must never enter the graph.
    """
    if with_meas:
        # measurement fit (masked over missing sensors)
        y = obs_model.predict(x)
        res_m = torch.where(valid, (y - meas_t) / sigma, torch.zeros_like(y))
        if res_clip is not None:
            res_m = torch.clamp(res_m, -res_clip, res_clip)
        l_meas = (res_m**2).sum() / valid.sum().clamp(min=1)
    else:
        l_meas = x.new_zeros(())

    if with_phys:
        # discrete, rate-scaled physics on the liquid states
        dxdt = (x[..., 1:, :] - x[..., :-1, :]) / dt
        f = adm1da_rhs_torch(x[..., :-1, :], params)
        denom = torch.clamp(f[..., :_N_LIQUID].detach().abs(), min=scale)
        res_p = (dxdt[..., :_N_LIQUID] - f[..., :_N_LIQUID]) / denom
        if res_clip is not None:
            res_p = torch.clamp(res_p, -res_clip, res_clip)
        l_phys = (res_p**2).mean()
    else:
        l_phys = x.new_zeros(())
    return l_meas, l_phys


def pretrain_observer(
    observer: Adm1Observer,
    dataset: ObserverDataset,
    *,
    val_dataset: ObserverDataset | None = None,
    epochs: int = 300,
    lr: float = 1.0e-3,
    batch_size: int = 32,
    val_frac: float = 0.2,
    burnin: int = 0,
    noise_std: Sequence[float] | None = None,
    weight_decay: float = 0.0,
    restore_best: bool = True,
    patience: int | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> PretrainResult:
    """Supervised pre-training of the observer on a simulator dataset.

    Args:
        dataset: training sequences. If ``val_dataset`` is given this is used in
            full for training; otherwise it is split internally by ``val_frac``.
        val_dataset: an **externally split** validation set. Pass this to share
            one split across estimators — the deep-learning adapter's
            :meth:`~.data_adapter.PinnData.observer_dataset` emits train and val
            from the same stratified split the filters use, which is what makes a
            filter and a network comparable. Splitting internally instead gives
            each model its own random split.
        burnin: leading steps excluded from the loss (the observer cannot know
            the initial state; see :func:`_scaled_state_loss`).
        noise_std: per-channel sensor noise in **raw units**, resampled onto the
            measurement features every batch. Cheap augmentation that targets the
            actual nuisance, and with ~80 series the binding constraint is
            overfitting, not capacity.
        weight_decay: AdamW-style L2 on the weights.
        restore_best: return the best-validated weights rather than the last
            ones. Training past the validation minimum only makes the model
            worse; the pilot's best epoch was 62 of 200.
        patience: stop after this many epochs without a validation improvement
            (``None`` = run all ``epochs``).

    Returns:
        :class:`PretrainResult` with the loss history, the split, the per-state
        normalisation scale, and which epoch was restored.
    """
    dtype = observer._dtype
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if val_dataset is not None:
        feats = torch.tensor(dataset.features, dtype=dtype)
        states = torch.tensor(dataset.states, dtype=dtype)
        val_feats = torch.tensor(val_dataset.features, dtype=dtype)
        val_states = torch.tensor(val_dataset.states, dtype=dtype)
        train_idx = np.arange(feats.shape[0])
        val_idx = np.arange(val_feats.shape[0])
        train_states = states
    else:
        feats = torch.tensor(dataset.features, dtype=dtype)
        states = torch.tensor(dataset.states, dtype=dtype)
        perm = rng.permutation(feats.shape[0])
        n_val = max(1, round(val_frac * feats.shape[0]))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        val_feats = feats[torch.as_tensor(val_idx)]
        val_states = states[torch.as_tensor(val_idx)]
        train_states = states[torch.as_tensor(train_idx)]

    # Normalisation from the TRAINING sequences only — computing it over the
    # whole set first would leak validation statistics into the objective.
    scale = torch.sqrt((train_states**2).mean(dim=(0, 1))) + 1e-8  # (n_state,)
    noise = _noise_scales(dataset, noise_std)
    if noise is not None:
        noise = noise.to(dtype=dtype)

    opt = torch.optim.Adam(observer.parameters(), lr=lr, weight_decay=weight_decay)
    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val, best_epoch, best_state = float("inf"), -1, None
    stopped_early = False

    for ep in range(epochs):
        observer.train()
        order = rng.permutation(
            train_idx if val_dataset is None else np.arange(len(feats))
        )
        batch_losses = []
        for i in range(0, len(order), batch_size):
            b = torch.as_tensor(order[i : i + batch_size])
            xb, yb = feats[b], states[b]
            if noise is not None:
                xb = xb + noise * torch.randn_like(xb)
            opt.zero_grad()
            loss = _scaled_state_loss(observer(xb), yb, scale, burnin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(observer.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss.detach()))

        observer.eval()
        with torch.no_grad():
            val_loss = float(
                _scaled_state_loss(observer(val_feats), val_states, scale, burnin)
            )
        history["train"].append(float(np.mean(batch_losses)))
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val, best_epoch = val_loss, ep
            if restore_best:
                best_state = copy.deepcopy(observer.state_dict())
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(
                f"[{ep:4d}] train={history['train'][-1]:.4e}  val={val_loss:.4e}",
                flush=True,
            )
        if patience is not None and ep - best_epoch >= patience:
            stopped_early = True
            if verbose:
                print(
                    f"[{ep:4d}] no val improvement for {patience} epochs — stopping.",
                    flush=True,
                )
            break

    if best_state is not None:
        observer.load_state_dict(best_state)

    return PretrainResult(
        history=history,
        train_idx=train_idx,
        val_idx=val_idx,
        scale=scale.cpu().numpy(),
        best_epoch=best_epoch,
        best_val=best_val,
        stopped_early=stopped_early,
    )


def finetune_observer(
    observer: Adm1Observer,
    features: np.ndarray,
    obs_model: TorchObservationModel,
    measurements: np.ndarray,
    dt_days: float,
    *,
    epochs: int = 50,
    lr: float = 1.0e-4,
    lambda_phys: float = 1.0,
    lambda_meas: float = 1.0,
    lambda_anchor: float = 0.0,
    params: Adm1TorchParams | None = None,
    grad_clip: float | None = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, list[float]]:
    """Phase-2 online fine-tuning — **self-supervised** (no ground-truth state).

    In operation only the measurements are known, so the pre-trained observer is
    adapted on the recent window by two self-supervised losses, warm-started from
    its pre-trained weights (small ``lr``):

    * **measurement fit** ``((h(x̂) − y_meas) / σ)²`` — the estimate must
      reproduce the observed sensors, and
    * **physics** ``((Δx̂/Δt − f(x̂)) / |f|)²`` — a *discrete* (finite-difference)
      rate-scaled ODE residual on the 37 liquid states.

    An optional ``lambda_anchor`` keeps the trajectory near the frozen
    pre-trained prediction (a trust region against drifting on noisy data).

    Args:
        features: ``(T, n_feat)`` or ``(1, T, n_feat)`` normalised observer inputs.
        obs_model: differentiable observation map (with the channels' noise stds).
        measurements: ``(T, n_channels)`` *raw* measured values (``NaN`` = missing),
            in ``obs_model.channel_names`` order.
        dt_days: measurement step [days] for the discrete physics residual.
        params: physics parameters; defaults to the observer's (pass an updated
            snapshot if the live feed / operating point differs).

    Returns:
        Loss history dict (``loss``, ``meas``, ``phys``).
    """
    torch.manual_seed(seed)
    dtype = observer._dtype
    p = params if params is not None else observer.params

    feats = np.asarray(features, dtype=float)
    if feats.ndim == 2:
        feats = feats[None, ...]
    feats_t = torch.tensor(feats, dtype=dtype)
    meas_t = torch.tensor(np.asarray(measurements, dtype=float), dtype=dtype)
    valid = ~torch.isnan(meas_t)
    sigma = obs_model.noise_std_tensor(dtype=dtype)
    scale = observer._base  # (37,) liquid magnitudes, used as the rate floor
    dt = float(dt_days)

    x_frozen = None
    if lambda_anchor > 0:
        observer.eval()
        with torch.no_grad():
            x_frozen = observer(feats_t)[0].detach()

    opt = torch.optim.Adam(observer.parameters(), lr=lr)
    history: dict[str, list[float]] = {"loss": [], "meas": [], "phys": []}
    # Keep the best-so-far weights and restore them at the end, so self-supervised
    # fine-tuning is monotone-safe: it never returns a model worse than the start
    # (the quasi-steady solve on a drifted state can overflow / diverge).
    import copy

    best_loss = float("inf")
    best_state = copy.deepcopy(observer.state_dict())
    observer.train()
    for ep in range(epochs):
        opt.zero_grad()
        x = observer(feats_t)[0]  # (T, 41)

        l_meas, l_phys = _selfsup_losses(
            x, meas_t, valid, obs_model, sigma, p, scale, dt
        )
        loss = lambda_meas * l_meas + lambda_phys * l_phys
        if x_frozen is not None:
            loss = (
                loss
                + lambda_anchor
                * (((x[:, :_N_LIQUID] - x_frozen[:, :_N_LIQUID]) / scale) ** 2).mean()
            )

        if not torch.isfinite(loss):
            if verbose:
                print(
                    f"[{ep:4d}] non-finite loss — stopping, restoring best weights.",
                    flush=True,
                )
            break

        # These weights produced the finite loss above → remember them *before*
        # stepping, so a restored checkpoint always yields a finite forward.
        lo = float(loss.detach())
        if lo < best_loss:
            best_loss = lo
            best_state = copy.deepcopy(observer.state_dict())

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(observer.parameters(), grad_clip)
        opt.step()

        history["loss"].append(lo)
        history["meas"].append(float(l_meas.detach()))
        history["phys"].append(float(l_phys.detach()))
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(
                f"[{ep:4d}] loss={lo:.4e} meas={history['meas'][-1]:.4e} "
                f"phys={history['phys'][-1]:.4e}",
                flush=True,
            )

    observer.load_state_dict(best_state)  # return the best-scoring weights
    return history


@dataclass
class SelfSupPretrainResult:
    """Outputs of :func:`pretrain_observer_selfsup`."""

    history: dict[str, list[float]]  # per-epoch train/val loss + meas/phys terms
    train_idx: np.ndarray
    val_idx: np.ndarray


def pretrain_observer_selfsup(
    observer: Adm1Observer,
    dataset: MeasurementDataset,
    obs_model: TorchObservationModel,
    *,
    epochs: int = 200,
    lr: float = 1.0e-4,
    batch_size: int = 8,
    val_frac: float = 0.2,
    lambda_meas: float = 1.0,
    lambda_phys: float = 1.0,
    res_clip: float | None = None,
    grad_clip: float | None = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> SelfSupPretrainResult:
    """Self-supervised pre-training on measurement-only windows (no ground truth).

    Trains the observer on a :class:`MeasurementDataset` with the same objective
    as the online fine-tuning — measurement fit + rate-scaled physics — so it can
    be primed directly on **real** plant history. Like the fine-tuning it is
    **monotone-safe**: the best-validated weights are restored at the end and a
    non-finite forward/loss stops training, so it never returns a worse (or
    NaN) model than it started from.

    Numerical note: the biogas measurement map is ill-conditioned (the knife-edge
    ``Q_gas`` / pH cancellation documented throughout this project), so the
    self-supervised gradient can only *nudge* the fit and a **cold** start can
    diverge in a single step (the quasi-steady gas Newton blows up on an
    off-state). In practice run it as the second stage on top of a supervised
    (or otherwise reasonable) init — see :func:`pretrain_observer_sim2real`. From
    a cold start it safely returns the initial weights rather than a NaN model.

    Args:
        dataset: measurement windows (build via :meth:`MeasurementDataset.from_real`
            or :meth:`MeasurementDataset.from_observer_dataset`).
        obs_model: differentiable observation map (channels + noise stds).
        val_frac: fraction of sequences held out for validation / best-weight
            selection (ignored when the dataset has a single sequence).

    Returns:
        :class:`SelfSupPretrainResult` (loss history + the sequence split). The
        recorded ``val`` history holds only finite values (training stops before
        recording a diverged step).
    """
    import copy

    torch.manual_seed(seed)
    dtype = observer._dtype
    feats = torch.tensor(dataset.features, dtype=dtype)  # (N, T, n_feat)
    meas = torch.tensor(
        np.asarray(dataset.measurements, float), dtype=dtype
    )  # (N, T, n_ch)
    valid = ~torch.isnan(meas)
    sigma = obs_model.noise_std_tensor(dtype=dtype)
    scale = observer._base  # (37,) rate floor
    p = dataset.params if dataset.params is not None else observer.params
    dt = float(dataset.dt_days)

    n = feats.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if n <= 1 or val_frac <= 0:
        n_val = 0  # no held-out split: select best weights on the training set
    else:
        n_val = max(1, min(round(val_frac * n), n - 1))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if n_val > 0 else perm
    eval_t = torch.as_tensor(val_idx if n_val > 0 else train_idx)

    def _seq_loss(idx: torch.Tensor) -> torch.Tensor:
        x = observer(feats[idx])  # (B, T, 41)
        l_meas, l_phys = _selfsup_losses(
            x,
            meas[idx],
            valid[idx],
            obs_model,
            sigma,
            p,
            scale,
            dt,
            res_clip=res_clip,
            with_meas=lambda_meas > 0.0,
            with_phys=lambda_phys > 0.0,
        )
        return lambda_meas * l_meas + lambda_phys * l_phys

    opt = torch.optim.Adam(observer.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val = float("inf")
    best_state = copy.deepcopy(observer.state_dict())

    for ep in range(epochs):
        # Evaluate current weights *first* so the recorded history reflects a
        # finite model and the best-weight snapshot precedes any diverging step.
        observer.eval()
        with torch.no_grad():
            val_loss = float(_seq_loss(eval_t))
        if not np.isfinite(val_loss):
            if verbose:
                print(
                    f"[{ep:4d}] non-finite eval — stopping, restoring best weights.",
                    flush=True,
                )
            break
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(observer.state_dict())

        observer.train()
        order = rng.permutation(train_idx)
        batch_losses: list[float] = []
        diverged = False
        for i in range(0, len(order), batch_size):
            b = torch.as_tensor(order[i : i + batch_size])
            opt.zero_grad()
            loss = _seq_loss(b)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            # A single ill-conditioned batch (stiff physics / gas map) can produce
            # a non-finite gradient; applying it would poison every weight. Skip
            # that step instead of corrupting the model.
            if any(
                prm.grad is not None and not torch.isfinite(prm.grad).all()
                for prm in observer.parameters()
            ):
                opt.zero_grad()
                continue
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(observer.parameters(), grad_clip)
            opt.step()
            batch_losses.append(float(loss.detach()))
        history["train"].append(
            float(np.mean(batch_losses)) if batch_losses else float("nan")
        )
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(
                f"[{ep:4d}] train={history['train'][-1]:.4e}  val={val_loss:.4e}",
                flush=True,
            )
        if diverged:
            if verbose:
                print(
                    f"[{ep:4d}] non-finite batch — stopping, restoring best weights.",
                    flush=True,
                )
            break

    observer.load_state_dict(best_state)  # return the best-validated weights
    return SelfSupPretrainResult(history=history, train_idx=train_idx, val_idx=val_idx)


def pretrain_observer_sim2real(
    observer: Adm1Observer,
    sim_dataset: ObserverDataset,
    real_dataset: MeasurementDataset,
    obs_model: TorchObservationModel,
    *,
    sup_epochs: int = 300,
    selfsup_epochs: int = 200,
    sup_kwargs: dict | None = None,
    selfsup_kwargs: dict | None = None,
    verbose: bool = False,
) -> tuple[PretrainResult, SelfSupPretrainResult]:
    """Sim→real pre-training: supervised on the simulator, then self-supervised.

    Stage 1 learns the full-state structure from the known model (the only source
    of truth for the unobservable states); stage 2 adapts to the real plant with
    the measurement + physics objective. Share the normalisation stats between the
    two datasets (pass ``sim_dataset.feat_mean / feat_std`` to
    :meth:`MeasurementDataset.from_real`) so both stages see the same input scale.

    Returns the (supervised, self-supervised) result objects.
    """
    r_sup = pretrain_observer(
        observer, sim_dataset, epochs=sup_epochs, verbose=verbose, **(sup_kwargs or {})
    )
    r_self = pretrain_observer_selfsup(
        observer,
        real_dataset,
        obs_model,
        epochs=selfsup_epochs,
        verbose=verbose,
        **(selfsup_kwargs or {}),
    )
    return r_sup, r_self


def observer_predict(observer: Adm1Observer, features: np.ndarray) -> np.ndarray:
    """Run the observer on ``(N, T, n_feat)`` inputs → ``(N, T, 41)`` states."""
    observer.eval()
    with torch.no_grad():
        x = observer(torch.tensor(features, dtype=observer._dtype))
    return x.cpu().numpy()


def per_state_nrmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-state RMSE ÷ truth RMS, averaged over scenarios and time. Shape (41,)."""
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)))
    denom = np.sqrt(np.mean(true**2, axis=(0, 1))) + 1e-12
    return rmse / denom
