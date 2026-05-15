# OHO Stability Experiments

Sequential ablation sweeps to identify what's required for stable OHO training. Each experiment answers one question; later experiments use the previous winner as their base.

**Metric:** `eval_accumulated/level2/loss/0/0/0` (lower = better)
**Seeds per cell:** 1 (feasibility study)
**Common fixed settings:** target = `{meta1_beta}`, damping = 1e-3, latent_dim = 2, all of VAE_BETA_OHO_V2's inner-SGD settings.

---

## E01 — `e01_outer_optimizer.py`

**Question:** What combination of (outer optimizer, mlr, RTRL beta, val_beta) gives the lowest test loss when meta-learning β alone? And does EG fix the instability that additive optimizers have?

**Base task:** `7860a1fe947842b5b76861455a20515e` (stub from VAE_BETA_OHO_V2 config `d9980a8c24584a6caafe4674962d3081`)

**Cells:** 90 total
- additive SGD: 3 mlr × 3 RTRL × 2 val_beta = 18
- additive Adam: 3 mlr × 3 RTRL × 2 val_beta × 2 b1 = 36
- EG(SGD): 3 mlr × 3 RTRL × 2 val_beta = 18
- EG(Adam): 3 mlr × 3 RTRL × 2 val_beta = 18

**Status:** pending
**Winner:** (TBD)

---

## E02 — `e02_damping.py`

**Question:** Does RTRL damping help, given E01's winner?

**Cells:** 2 (damping ∈ {0, 1e-3})

**Status:** not yet written
**Winner:** (TBD)

---

## E03 — `e03_target.py`

**Question:** How big can the OHO target subset grow without breaking stability?

**Cells:** 3 (target ∈ {`{β}`, `{β, lr}`, `{β, lr, wd}`})

**Status:** not yet written
**Winner:** (TBD)

---

## E04 — `e04_latent_dim.py`

**Question:** Does latent dim affect OHO stability?

**Cells:** 2 (latent_dim ∈ {2, 10})

**Status:** not yet written
**Winner:** (TBD)
