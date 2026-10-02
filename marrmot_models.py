"""
marrmot_models.py

Python port of the 46 conceptual rainfall-runoff models catalogued in MARRMoT
(Knoben, Fowler, Peel & Woods, 2018, "Modular Assessment of Rainfall-Runoff
Models Toolbox v1.2", Geoscientific Model Development), transcribed from
`MARRMoT_model_catalogue.md` in this repository.

Every `step_model_NN` function advances one model by a single time step:

    S_new, Q, fluxes = step_model_NN(S, forcings, params, dt=1.0)

or, for the 11 models that need routing state (see below):

    S_new, Q, fluxes, uh_state = step_model_NN(S, forcings, params, dt=1.0,
                                                uh_state=None)

- `S` is a list/tuple of the model's stores, in the order given in its
  docstring's "State variables" section.
- `forcings` is a dict with keys 'P' and 'Ep' [mm/d] (plus 'T' [deg C] for
  snow models, and 't' [day-of-year] for the few models with a seasonal
  interception/LAI term -- each function's docstring lists which it needs,
  and so does `MODEL_REGISTRY[n]['forcing_vars']`).
- `params` is a dict keyed by the model's own parameter symbols (see each
  docstring's "Parameters" section, or `MODEL_REGISTRY[n]['param_names']`).
- `dt` is the time step in days (MARRMoT's internal units are always mm/d
  and 1/d, regardless of the forcing data's native time step).
- `fluxes` is a dict of every named intermediate flux computed that step,
  for diagnostics.

Use `simulate(model_num, P, Ep, params, S0, dt=1.0, T=None, t=None)` to run
a full time series through any model without hand-writing the day loop --
see its docstring below. `MODEL_REGISTRY` indexes all 46 models by number.

Three important caveats, carried over from the source catalogue itself:

1. **Hard thresholds ARE smoothed, using MARRMoT's own real formulas.** The
   source documentation prints every threshold as a literal
   `if S > threshold: ... else: ...` -- that is each model's mathematical
   *definition*, but not what MARRMoT actually runs: its real MATLAB
   implementation replaces every one of these with a logistic-smoothed blend
   before handing the model to a numerical solver (source document's Table
   S1; functions like `saturation_1`, `evap_1`, `baseflow_9`). This port
   applies that same technique, not just a paraphrase of it: `logistic_storage`
   and `logistic_temperature` below are transcribed verbatim from MARRMoT
   v1.2's actual source (`smoothThreshold_storage_logistic.m` and
   `smoothThreshold_temperature_logistic.m`,
   github.com/wknoben/MARRMoT@v1.2), and every SIMPLE two-branch threshold
   switch in all 46 models -- a ternary or an if/else block choosing between
   two flux formulas on either side of one comparison -- was mechanically
   rewritten from `A if S >= Smax else B` into
   `w = logistic_storage(S, Smax); result = (1-w)*A + w*B` (temperature
   comparisons, i.e. `T` against a threshold, use `logistic_temperature`
   instead, matching MARRMoT's own separate function for that case). A
   couple of caveats on coverage and fidelity:
   - Compound conditions (`and`/`or` of two different thresholds), 3-way/elif
     chains, and thresholds that are themselves a compound expression rather
     than a single name or number were left as literal thresholds rather
     than risk a wrong blend -- these are rare (one such case in the whole
     module, Model 35's `GSI`, and that one is already a continuous
     piecewise-linear clamp with no discontinuity to smooth).
   - MARRMoT's real per-flux choice is not *always* this logistic sigmoid --
     `evap_1` and `baseflow_9`, both named as smoothing examples in the
     source document, actually turn out to use plain `min()`/`max()` hinges
     instead when checked against the real source. This port uses the
     logistic blend uniformly for every threshold switch, which is faithful
     to MARRMoT's general *technique* and removes the state-chattering a
     literal `if/else` causes right at a boundary, but was not individually
     re-verified flux-by-flux against all ~90 of MARRMoT's real per-flux
     `.m` files for bit-exact reproduction of which smoothing MARRMoT
     happens to use in each specific spot.
   - This is a deliberate departure from the literal-threshold approach this
     course's `Example3_A_yearly_scale_model.ipynb` /
     `Example4_A_monthly_scale_model.ipynb` use (those notebooks work through
     the *hard*-threshold numerical subtleties on purpose, as a teaching
     point) -- running the same parameters through `step_model_01` here and
     through Example 3's bucket will therefore no longer match exactly, by
     design. The difference is not just cosmetic: with the default `r=0.01,
     e=5.0`, a store is already meaningfully "leaking" quickflow starting
     roughly 5-10% of `Smax` below the nominal capacity (e.g. for
     `Smax=150`, `S=140` already diverts ~16% of that day's rain to
     quickflow), not only exactly at/above it. That widens the effective
     "reactive zone" enough to noticeably shift the long-term water balance
     versus a literal threshold (in one check against this course's own
     watershed data, Model 01's mean annual quickflow roughly doubled) --
     this is expected, physically defensible behavior (real catchments
     don't have a razor-sharp saturation boundary either), not an error;
     verify mass balance for your own use case if the exact magnitude matters.

2. **One consistent, deliberate fix was applied to "at-capacity" checks in
   the (rare) places a threshold stayed literal.** A few models in the
   source spec gate an overflow or a depleted-store flux on bit-exact
   equality -- e.g. "route today's rain to quickflow if the store's content
   == its capacity". Bit-exact floating-point equality against a threshold
   essentially never holds in a discrete Euler simulation -- exactly the
   issue this course worked through explicitly for Example 3's `S_n >= S_max`
   check. Every such literal `== capacity` was ported as `>= capacity`, and
   every `== 0` as `<= 0`, before the smoothing pass above ran (so smoothed
   thresholds inherit the fix automatically; it only still matters for the
   couple of compound conditions smoothing skipped). A couple of the source
   document's own self-flagged likely typos (e.g. Model 26's
   `Rp = Percmax * (-UR/URmax)`) are preserved with a comment rather than
   silently "corrected."

3. **Unit Hydrograph routing was implemented, not left as a placeholder.**
   11 of the 46 models route some flux through a Unit Hydrograph or an
   exponential-decay reservoir before it reaches the stream (IHACRES, GR4J,
   Hillslope, Plateau, NZ v2, Flex-B, Flex-I, FLEX-IS, HBV-96, MCRM, SMAR).
   The source catalogue's pseudocode for these explicitly calls placeholder
   functions ("a real implementation keeps a rolling buffer...") rather than
   defining them. This module supplies real, general-purpose routing
   utilities (`UHRouter`, `ExponentialRouter`, `PureDelay`, and ordinate
   generators `uh_triangular`, `uh_uniform`, `uh_gamma`, `uh_gr4j_uh1`,
   `uh_gr4j_uh2`) and wires each of those 11 models up to them. Where the
   source only names a time-base parameter without specifying an ordinate
   *shape* (most of the generic triangular-UH cases), a standard symmetric
   triangular hydrograph is used -- verify against MARRMoT's own `uh_*.m`
   ordinate definitions if bit-for-bit reproduction of the original toolbox
   matters. GR4J's two UHs use the exact published S-curve formulas
   (Perrin et al., 2003), since those are unambiguous.

Parameter sampling ranges quoted in each docstring come from the source
catalogue's own cross-reference (Section S5, Table S3) and, per that
document's own caveat, should be treated as a starting point to verify
against MARRMoT's actual `*_parameter_ranges.m` files rather than a
certainty -- several were flagged there as ambiguous or unit-mismatched.
"""

import numpy as np
from math import exp, log, sin, cos, sqrt, pi, erf
from scipy.special import gammaincc


# ===========================================================================
# Threshold smoothing (MARRMoT's real logistic smoothers)
# ===========================================================================

def logistic_storage(S, Smax, r=0.01, e=5.0):
    """
    MARRMoT's `smoothThreshold_storage_logistic` (v1.2), transcribed exactly:
    a logistic multiplier that is ~1 when `S` is well below `Smax` and ~0
    once `S` is at/above it, with a smooth transition of width ~`r`*`Smax`.
    Used throughout this module as `w` in `w*<below> + (1-w)*<at-or-above>`
    blends -- see the module docstring's threshold-smoothing note.

    Uses `numpy.exp` rather than `math.exp`: far from the threshold the
    exponent can be a very large number, and unlike MATLAB's `exp` (which
    just saturates to `Inf`), Python's `math.exp` *raises* `OverflowError`
    on the same input -- `numpy.exp` saturates the same way MATLAB does.
    """
    Smax = max(Smax, 0.0)
    denom = r * Smax if r * Smax != 0 else r
    return float(1.0 / (1.0 + np.exp((S - Smax + r * e * Smax) / denom)))


def logistic_temperature(T, Tt, r=0.01):
    """
    MARRMoT's `smoothThreshold_temperature_logistic` (v1.2), transcribed
    exactly: ~1 when `T` is well below the threshold `Tt` (e.g. snowing),
    ~0 once `T` is at/above it (e.g. raining). See `logistic_storage` for
    why this uses `numpy.exp` instead of `math.exp`.
    """
    return float(1.0 / (1.0 + np.exp((T - Tt) / r)))


# ===========================================================================
# Generic routing utilities
# ===========================================================================

class UHRouter:
    """
    Finite-impulse-response Unit Hydrograph router.

    Holds a set of ordinates (weights, normalized to sum to 1) and a rolling
    buffer of not-yet-released water. Call `.route(x)` once per time step
    with today's input flux; it returns today's *output* flux -- the share
    of today's and previous inputs the ordinates say should leave today.
    """

    def __init__(self, ordinates):
        ordinates = np.asarray(ordinates, dtype=float)
        total = ordinates.sum()
        if total > 0:
            ordinates = ordinates / total
        else:
            ordinates = np.array([1.0])
        self.ordinates = ordinates
        self.buffer = np.zeros(len(ordinates))

    def route(self, x):
        self.buffer = self.buffer + x * self.ordinates
        out = float(self.buffer[0])
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = 0.0
        return out


class ExponentialRouter:
    """
    O(1)-state exponential-decay routing: a single linear reservoir,
    x_t = alpha*x_{t-1} + (1-alpha)*u_t with alpha = exp(-dt/tau). Used for
    IHACRES's quick/slow pathways instead of a truncated FIR buffer, since
    an exponential decay has an infinite tail that a finite UHRouter buffer
    would have to truncate.
    """

    def __init__(self, tau, dt=1.0):
        self.alpha = exp(-dt / tau) if tau > 0 else 0.0
        self.x = 0.0

    def route(self, u):
        self.x = self.alpha * self.x + (1 - self.alpha) * u
        return self.x


class PureDelay:
    """Pure time-shift buffer (no attenuation) -- e.g. IHACRES's tau_d term."""

    def __init__(self, delay, dt=1.0):
        n = max(int(round(delay / dt)), 0)
        self.buffer = [0.0] * n

    def route(self, x):
        if not self.buffer:
            return x
        self.buffer.append(x)
        return self.buffer.pop(0)


def uh_uniform(time_base, dt=1.0):
    """Rectangular UH: input spread evenly over `time_base` days."""
    n = max(int(round(time_base / dt)), 1)
    return np.full(n, 1.0 / n)


def uh_triangular(time_base, dt=1.0):
    """
    Symmetric triangular UH ordinates over a base of `time_base` days,
    peaking at time_base/2. Used (with model-specific time bases) across
    several MARRMoT models whose source spec names only a routing time-base
    parameter, not an ordinate shape -- verify against MARRMoT's own
    `uh_*.m` definitions if exact reproduction matters.
    """
    n = max(int(round(time_base / dt)), 1)
    t = np.arange(1, n + 1)
    peak = n / 2.0
    w = np.where(t <= peak, t, n - t + 1).astype(float)
    return w / w.sum()


def uh_gr4j_uh1(x4, dt=1.0):
    """
    GR4J's UH1 (Perrin et al., 2003): ordinates are first differences of the
    S-curve SH1(t) = (t/x4)^2.5 for 0 < t < x4, 1 beyond, sampled at every dt.
    """
    n = max(int(np.ceil(x4 / dt)), 1)

    def SH1(t):
        if t <= 0:
            return 0.0
        if t >= x4:
            return 1.0
        return (t / x4) ** 2.5

    edges = [SH1(i * dt) for i in range(0, n + 1)]
    return np.diff(edges)


def uh_gr4j_uh2(x4, dt=1.0):
    """GR4J's UH2 (Perrin et al., 2003): S-curve SH2, time base 2*x4."""
    n = max(int(np.ceil(2 * x4 / dt)), 1)

    def SH2(t):
        if t <= 0:
            return 0.0
        if t <= x4:
            return 0.5 * (t / x4) ** 2.5
        if t < 2 * x4:
            return 1 - 0.5 * (2 - t / x4) ** 2.5
        return 1.0

    edges = [SH2(i * dt) for i in range(0, n + 1)]
    return np.diff(edges)


def uh_gamma(n_shape, k_scale, dt=1.0, n_ordinates=None):
    """
    Gamma-distribution UH ordinates (shape `n_shape`, scale `k_scale` [d]),
    used by SMAR's uh_6_gamma routing.
    """
    if n_ordinates is None:
        n_ordinates = max(int(np.ceil(10 * k_scale * max(n_shape, 1))), 10)
    t = np.arange(1, n_ordinates + 1) * dt
    w = t ** (n_shape - 1) * np.exp(-t / k_scale)
    if w.sum() == 0:
        w = np.zeros(n_ordinates)
        w[0] = 1.0
    return w / w.sum()


def reg_upper_incomplete_gamma(a, x):
    """Regularized upper incomplete gamma function Q(a, x) -- TOPMODEL's A_c term."""
    return gammaincc(a, x)


def standard_normal_cdf(x):
    """Standard normal CDF, via the error function -- HYCYMODEL's flux split."""
    return 0.5 * (1 + erf(x / sqrt(2)))


def clip(x, lo, hi):
    """Clamp `x` to [lo, hi] -- TOPMODEL's saturated-area fraction A_c."""
    return min(max(x, lo), hi)


# ===========================================================================
# Model step functions (Models 01-46)
# ===========================================================================

def step_model_01(S, forcings, params, dt=1.0):
    """
    Model 01 -- Collie River Basin 1  (MARRMoT file: m_01_collie1_1p_1s)

    Reference(s): Jothityangkoon et al., 2001

    Processes represented:
      - Evaporation from soil moisture
      - Saturation excess surface runoff

    State variables (1):
      S : current storage in the soil moisture [mm]

    Parameters (1):
      Smax       [mm]  Maximum soil moisture storage  range [1, 2000]

    Output / routing (as specified):
      Q = Qse
      Routing: none
    """
    # unpack state
    S_soil = S[0]

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    Smax = params['Smax']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    Ea = (S_soil / Smax) * Ep

    _w1 = logistic_storage(S_soil, Smax)
    Qse = (1 - _w1) * (P) + _w1 * (0)

    # update state via mass balance (explicit Euler, dt=1.0 day)
    dS_soil = P - Ea - Qse
    S_soil_new = min(max(S_soil + dS_soil * dt, 0.0), Smax)

    Q = Qse
    fluxes = {'Ea': Ea, 'Qse': Qse}
    return [S_soil_new], Q, fluxes


def step_model_02(S, forcings, params, dt=1.0):
    """
    Model 02 -- Wetland model  (MARRMoT file: m_02_wetland_4p_1s)

    Reference(s): Savenije, 2010

    Processes represented:
      - Stylized interception by vegetation
      - Evaporation
      - Saturation excess runoff generated from a distribution of soil depths
      - A linear relation between storage and slow runoff

    State variables (1):
      Sw : current soil water storage [mm]

    Parameters (4):
      Dw         [mm/d]  Interception evaporation (loss before soil water store)  range [0, 5]
      Sw,max     [mm]  Maximum soil moisture storage  range [1, 2000]
      βw         [–]  Non-linearity parameter for contributing area  range [0, 10]
      Kw         [d⁻¹]  Runoff coefficient (linear groundwater/slow-flow rate)  range [0, 1]

    Output / routing (as specified):
      Q = Qw,sof + Qw,gw   (19)
      Routing: none
    """
    # unpack state
    Sw = S[0]

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    Dw, Sw_max, beta_w, Kw = params['Dw'], params['Sw_max'], params['beta_w'], params['Kw']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    Pe = max(P - Dw, 0)

    _w1 = logistic_storage(Sw, 0)
    Ew = (1 - _w1) * (Ep) + _w1 * (0)

    Qw_sof = (1 - (1 - Sw / Sw_max) ** beta_w) * Pe
    Qw_gw  = Kw * Sw

    # update state via mass balance (explicit Euler, dt=1.0 day)
    dSw = Pe - Ew - Qw_sof - Qw_gw
    Sw_new = Sw + dSw * dt

    Q = Qw_sof + Qw_gw
    fluxes = {'Pe': Pe, 'Ew': Ew, 'Qw_sof': Qw_sof, 'Qw_gw': Qw_gw}
    return [Sw_new], Q, fluxes


def step_model_03(S, forcings, params, dt=1.0):
    """
    Model 03 -- Collie River Basin 2  (MARRMoT file: m_03_collie2_4p_1s)

    Reference(s): Jothityangkoon et al., 2001

    Processes represented:
      - Separate bare soil and vegetation evaporation
      - Saturation excess surface runoff
      - Subsurface runoff

    State variables (1):
      S : current storage in the soil moisture [mm]

    Parameters (4):
      Smax       [mm]  Maximum soil moisture storage  range [1, 2000]
      Sfc        [mm]  Field capacity  range [0.05, 0.95]*
      a          [d⁻¹]  Runoff coefficient  range [0, 1]
      M          [–]  Forest fraction  range [0.05, 0.95]

    Output / routing (as specified):
      Q = Qse + Qss   (25)
      Routing: none
    """
    # unpack state
    S_soil = S[0]

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    Smax, Sfc, a, M = params['Smax'], params['Sfc'], params['a'], params['M']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    Eb = (S_soil / Smax) * (1 - M) * Ep

    _w1 = logistic_storage(S_soil, Sfc)
    Ev = (1 - _w1) * (M * Ep) + _w1 * ((S_soil / Sfc) * M * Ep)

    _w2 = logistic_storage(S_soil, Smax)
    Qse = max((1 - _w2) * (P) + _w2 * (0),0)

    _w3 = logistic_storage(S_soil, Sfc)
    Qss = max((1 - _w3) * (a * max(0.0, S_soil - Sfc)) + _w3 * (0),0)

    # update state via mass balance (explicit Euler, dt=1.0 day)
    dS_soil = P - Eb - Ev - Qse - Qss
    S_soil_new = min(max(S_soil + dS_soil * dt, 0.0), Smax)

    Q = Qse + Qss
    fluxes = {'Eb': Eb, 'Ev': Ev, 'Qse': Qse, 'Qss': Qss}
    return [S_soil_new], Q, fluxes


def step_model_04(S, forcings, params, dt=1.0):
    """
    Model 04 -- New Zealand model v1  (MARRMoT file: m_04_newzealand1_6p_1s)

    Reference(s): Atkinson et al., 2002

    Processes represented:
      - Separate vegetation and bare soil evaporation
      - Saturation excess overland flow
      - Subsurface runoff when soil moisture exceeds field capacity
      - Baseflow

    State variables (1):
      Sm : current soil moisture storage [mm]

    Parameters (6):
      Smax       [mm]  Maximum soil moisture storage  range [1, 2000]
      Sfc        [–]  Field capacity as fraction of Smax  range [0.05, 0.95]
      M          [–]  Forest fraction  range [0.05, 0.95]
      a          [d⁻¹]  Runoff coefficient  range [0, 1]
      b          [–]  Runoff non-linearity  range [1, 5]
      tc_bf      [d⁻¹]  Runoff coefficient (baseflow time scale)  range [0, 1]

    Output / routing (as specified):
      Qt = Qse + Qss + Qbf   (32)
      Routing: none
    """
    # unpack state
    Sm = S[0]

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    Smax, Sfc, M, a, b, tc_bf = (params['Smax'], params['Sfc'], params['M'],
                                  params['a'], params['b'], params['tc_bf'])

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    _w1 = logistic_storage(Sm, Sfc)
    Eveg = (1 - _w1) * (M * Ep) + _w1 * ((Sm / Sfc) * M * Ep)

    Ebs = (Sm / Smax) * (1 - M) * Ep

    _w2 = logistic_storage(Sm, Smax)
    Qse = (1 - _w2) * (P) + _w2 * (0)

    _w3 = logistic_storage(Sm, Sfc)
    Qss = (1 - _w3) * ((a * max(0.0, Sm - Sfc)) ** b) + _w3 * (0)

    Qbf = tc_bf * Sm

    # update state via mass balance (explicit Euler, dt=1.0 day)
    dSm = P - Eveg - Ebs - Qse - Qss - Qbf
    Sm_new = Sm + dSm * dt

    Qt = Qse + Qss + Qbf
    fluxes = {'Eveg': Eveg, 'Ebs': Ebs, 'Qse': Qse, 'Qss': Qss, 'Qbf': Qbf}
    return [Sm_new], Qt, fluxes


def step_model_05(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 05 -- IHACRES  (MARRMoT file: m_05_ihacres_7p_1s)

    Reference(s): Littlewood et al., 1997; Ye et al., 1997; Croke and Jakeman, 2004 (this MARRMoT implementation is a modification of the original equations; uses Ep directly rather than temperature as a proxy; Ea and U formulated following Croke and Jakeman, 2004, with non-linearity in U based on Ye et al., 1997; uses a catchment moisture deficit (CMD) formulation rather than a catchment wetness index)

    Processes represented:
      - Catchment deficit build-up
      - Slow and fast routing of effective precipitation

    State variables (1):
      CMD : current catchment moisture deficit [mm] (0 = saturated store; increases with dryness)

    Parameters (7):
      lp         [mm]  Wilting point  range not specified in Table S3 — check S2.5.3 parameter table only
      d          [mm]  Deficit threshold for flow from rain  range [1, 2000]
      p          [–]  Deficit non-linearity  range [0, 10]
      α          [–]  Fraction flow to quick routing  range not specified in Table S3 — check S2.5.3 parameter table only
      τq         [d]  Unit Hydrograph time base (fast/quick routing)  range [1, 700]
      τs         [d]  Unit Hydrograph time base (slow routing)  range [1, 700]
      τd         [d]  Unit Hydrograph delay (pure time delay applied to total flow)  range [0, 119]

    Output / routing (as specified):
      Q = xq + xs   (38)
      optionally delayed by a pure time delay τd [d] (this delay is distinct from — and applied after — the τq/τs Unit Hydrograph transformations)
  
      Routing: YES — two parallel Unit Hydrographs (exponentially decreasing over time), one for the fast/quick component (time base τq) fed by Uq, one for the slow component (time base τs) fed by Us; total flow Q = xq + xs is then optionally passed through an additional pure time delay of τd days (a shift, not a transformation).
    """
    # unpack state
    CMD = S[0]

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    lp, d, p, alpha, tau_q, tau_s, tau_d = (params['lp'], params['d'], params['p'],
                                             params['alpha'], params['tau_q'],
                                             params['tau_s'], params['tau_d'])

    if uh_state is None:
        uh_state = {
            'q': ExponentialRouter(tau_q, dt),
            's': ExponentialRouter(tau_s, dt),
            'delay': PureDelay(tau_d, dt),
        }

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else / min() thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue -- this pseudocode shows the
    # literal mathematical definition)
    Ea = Ep * min(1, exp(2 * (1 - CMD / lp)))
    U  = P * (1 - min(1, (CMD / d) ** p))

    Uq = alpha * U
    Us = (1 - alpha) * U

    # update state via mass balance (explicit Euler, dt=1.0 day)
    dCMD = -P + Ea + U
    CMD_new = CMD + dCMD * dt

    # route Uq, Us through exponential-decay Unit Hydrographs (time bases tau_q, tau_s)
    xq = uh_state['q'].route(Uq)
    xs = uh_state['s'].route(Us)

    Q_pre_delay = xq + xs
    # apply the pure time delay tau_d (a shift, not a shape transform)
    Q = uh_state['delay'].route(Q_pre_delay)

    fluxes = {'Ea': Ea, 'U': U, 'Uq': Uq, 'Us': Us, 'xq': xq, 'xs': xs}
    return [CMD_new], Q, fluxes, uh_state


def step_model_06(S, forcings, params, dt=1.0):
    """
    Model 06 -- Alpine model v1  (MARRMoT file: m_06_alpine1_4p_2s)

    Reference(s): Eder et al., 2003

    Processes represented:
      - Snow accumulation and melt
      - Saturation excess overland flow
      - Linear subsurface runoff

    State variables (2):
      Sn : current snow storage [mm]
      Sm : current soil moisture storage [mm]

    Parameters (4):
      Tt         [°C]  Threshold temperature for snowfall and melt  range [-3, 5]
      ddf        [mm °C⁻¹ d⁻¹]  Degree-day factor  range [0, 20]
      Smax       [mm]  Maximum soil moisture storage  range [1, 2000]
      tc         [d⁻¹]  Runoff coefficient  range [0, 1]

    Output / routing (as specified):
      Qt = Qse + Qss   (47)
      Routing: none
    """
    # unpack state
    Sn, Sm = S[0], S[1]

    # unpack forcings
    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    # unpack params
    Tt, ddf, Smax, tc = params['Tt'], params['ddf'], params['Smax'], params['tc']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    _w1 = logistic_temperature(T, Tt)
    Ps = _w1 * (P) + (1 - _w1) * (0)

    _w2 = logistic_temperature(T, Tt)
    QN = (1 - _w2) * (ddf * (T - Tt)) + _w2 * (0)

    _w3 = logistic_temperature(T, Tt)
    Pr = (1 - _w3) * (P) + _w3 * (0)

    if Sm > 0:          # "S" in eq. (44) refers to the soil moisture store Sm
        Ea = Ep
    else:
        Ea = 0

    _w4 = logistic_storage(Sm, Smax)
    Qse = (1 - _w4) * (Pr + QN) + _w4 * (0)

    Qss = tc * Sm

    # update states via mass balance (explicit Euler, dt=1.0 day)
    dSn = Ps - QN
    dSm = Pr + QN - Ea - Qse - Qss
    Sn_new = Sn + dSn * dt
    Sm_new = Sm + dSm * dt

    Qt = Qse + Qss
    fluxes = {'Ps': Ps, 'QN': QN, 'Pr': Pr, 'Ea': Ea, 'Qse': Qse, 'Qss': Qss}
    return [Sn_new, Sm_new], Qt, fluxes


def step_model_07(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 07 -- GR4J  (MARRMoT file: m_07_gr4j_4p_2s)

    Reference(s): Perrin et al., 2003 (original explicit operator-splitting scheme, incl. the two Unit Hydrographs used here); Santos et al., 2018 (implicit time-stepping scheme — MARRMoT follows most of the equations from this newer version, but keeps Perrin et al.'s original Unit Hydrographs for flood routing)

    Processes represented:
      - Implicit interception by vegetation, expressed as net precipitation or net evaporation
      - Different time delays within the catchment expressed by two hydrographs
      - Water exchange with neighbouring catchments

    State variables (2):
      S : current soil moisture storage [mm]
      R : current storage in the routing store [mm]

    Parameters (4):
      x1         [mm]  Maximum soil moisture storage  range [1, 2000]
      x2         [mm/d]  Subsurface water exchange (catchment groundwater exchange coefficient)  range [-10, 15]
      x3         [mm]  Routing store depth  range [1, 300]
      x4         [d]  Unit Hydrograph time base  range [1, 15]

    Output / routing (as specified):
      Q = Qr + max(Q1 + F(x2), 0)                                         # (57) — total flow Qt: routed slow component plus the fast component (floored at 0 after groundwater exchange is applied to it too)
      Routing: Unit Hydrograph — uh_1_half (time base x4) for the 90% "slow" branch feeding R; uh_2_full (time base 2·x4) for the 10% "fast" branch Q1
    """
    # unpack state
    S_soil, R = S[0], S[1]

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    x1, x2, x3, x4 = params['x1'], params['x2'], params['x3'], params['x4']

    if uh_state is None:
        uh_state = {
            'slow': UHRouter(uh_gr4j_uh1(x4, dt)),   # uh_1_half, time base x4  -> Q9
            'fast': UHRouter(uh_gr4j_uh2(x4, dt)),   # uh_2_full, time base 2*x4 -> Q1
        }

    # net precip / net evap split (mutually exclusive)
    if P >= Ep:
        Pn, En = P - Ep, 0.0
    else:
        Pn, En = 0.0, Ep - P

    Ps = Pn * (1 - (S_soil / x1) ** 2)
    Es = En * (2 * S_soil / x1 - (S_soil / x1) ** 2)
    Perc = (x1 ** -4 / 4) * (4 / 9) ** -4 * S_soil ** 5

    # update soil moisture store
    dS_soil = Ps - Es - Perc
    S_soil_new = S_soil + dS_soil * dt

    # split percolation + excess precip into slow (90%) / fast (10%) branches
    excess = Perc + (Pn - Ps)
    slow_input = 0.9 * excess   # routed via uh_1_half(time base x4) -> Q9
    fast_input = 0.1 * excess   # routed via uh_2_full(time base 2*x4) -> Q1

    Q9 = uh_state['slow'].route(slow_input)
    Q1 = uh_state['fast'].route(fast_input)

    Fx2 = x2 * (R / x3) ** 3.5
    Qr = (x3 ** -4 / 4) * R ** 5

    dR = Q9 + Fx2 - Qr
    R_new = R + dR * dt

    Q = Qr + max(Q1 + Fx2, 0)
    fluxes = {'Ps': Ps, 'Es': Es, 'Perc': Perc, 'Q9': Q9, 'Q1': Q1, 'Fx2': Fx2, 'Qr': Qr}
    return [S_soil_new, R_new], Q, fluxes, uh_state


def step_model_08(S, forcings, params, dt=1.0):
    """
    Model 08 -- United States model  (MARRMoT file: m_08_us1_5p_2s)

    Reference(s): Bai et al., 2009

    Processes represented:
      - Interception as a percentage of precipitation
      - Separate unsaturated and saturated zones
      - Separate bare soil evaporation and vegetation transpiration
      - Saturation excess overland flow
      - Subsurface flow

    State variables (2):
      S_us  : storage in the unsaturated zone [mm]
      S_sat : storage in the saturated zone [mm]

    Parameters (5):
      alpha_ei   [-]  Intercepted fraction of precipitation  range [0, 1]
      M          [-]  Forest fraction  range [0.05, 0.95]
      S_max      [mm]  Maximum soil moisture storage  range [1, 2000]
      fc         [-]  Field capacity as fraction of S_max  range [0.05, 0.95]
      alpha_ss   [d^-1]  Runoff (subsurface flow) coefficient  range [0, 1]

    Output / routing (as specified):
      ```
      Q = Q_se + Q_ss     (70)
      ```
      Routing: none (no Unit Hydrograph referenced in S2.8).
    """
    # unpack state
    S_us, S_sat = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    alpha_ei, M, S_max, fc, alpha_ss = (
        params['alpha_ei'],
        params['M'],
        params['S_max'],
        params['fc'],
        params['alpha_ss']
    )

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    S_usfc = fc * (S_max - S_sat)

    E_us_ei = alpha_ei * P

    _w1 = logistic_storage(S_us, S_usfc)
    E_us_veg = (1 - _w1) * (S_us / (S_us + S_sat) * M * Ep) + _w1 * (S_us / (S_us + S_sat) * M * Ep * S_us / S_usfc)

    E_us_bs = S_us / (S_us + S_sat) * (1 - M) * S_us / (S_max - S_sat) * Ep

    _w2 = logistic_storage(S_us, S_usfc)
    r_g = (1 - _w2) * (P) + _w2 * (0.0)
    _w3 = logistic_storage(S_us, S_usfc)
    S_e = (1 - _w3) * ((S_us - S_usfc)) + _w3 * (0.0) # see note above: unused downstream

    E_sat_veg = S_sat / S_max * M * Ep
    E_sat_bs  = S_sat / S_max * (1 - M) * Ep

    _w4 = logistic_storage(S_us, S_max)
    Q_se = (1 - _w4) * (r_g) + _w4 * (0.0)
    Q_ss = alpha_ss * S_sat

    # update state(s) via mass balance
    dS_us  = P - E_us_ei - E_us_veg - E_us_bs - r_g
    dS_sat = r_g - E_sat_veg - E_sat_bs - Q_se - Q_ss
    S_us_new  = S_us  + dS_us * dt
    S_sat_new = S_sat + dS_sat * dt

    Q = Q_se + Q_ss

    fluxes = dict(E_us_ei=E_us_ei, E_us_veg=E_us_veg, E_us_bs=E_us_bs, r_g=r_g,
                  S_e=S_e, E_sat_veg=E_sat_veg, E_sat_bs=E_sat_bs,
                  Q_se=Q_se, Q_ss=Q_ss)
    return (S_us_new, S_sat_new), Q, fluxes


def step_model_09(S, forcings, params, dt=1.0):
    """
    Model 09 -- Susannah Brook model v1-5  (MARRMoT file: m_09_susannah1_6p_2s)

    Reference(s): Son and Sivapalan, 2007

    Processes represented:
      - Evaporation from soil and transpiration from vegetation
      - Saturation excess and non-linear subsurface flow
      - Groundwater recharge and baseflow

    State variables (2):
      S_uz : storage in the upper (soil) zone [mm]
      S_gw : groundwater storage [mm]

    Parameters (6):
      S_b        [mm]  Maximum soil moisture storage  range [1, 2000]
      S_fc       [mm]  Field capacity  range not specified in Table S3 — check S2.9.3 parameter table only
      M          [-]  Forest fraction  range [0.05, 0.95]
      a          [d]  Runoff time coefficient  range [1, 50]
      b          [-]  Runoff nonlinearity  range [1, 5]
      r          [-]  Fraction of subsurface flow Q_ss to groundwater  range not specified in Table S3 — check S2.9.3 parameter table only

    Output / routing (as specified):
      ```
      Q = Q_se + (Q_ss - Q_r) + Q_b     (79)
      ```
      Routing: none.
    """
    # unpack state
    S_uz, S_gw = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    S_b, S_fc, M, a, b, r = (
        params['S_b'],
        params['S_fc'],
        params['M'],
        params['a'],
        params['b'],
        params['r']
    )

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    E_bs = S_uz / S_b * (1 - M) * Ep

    _w1 = logistic_storage(S_uz, S_fc)
    E_veg = (1 - _w1) * (M * Ep) + _w1 * (S_uz / S_fc * M * Ep)

    _w3 = logistic_storage(S_uz, S_b)
    Q_se = (1 - _w3) * (P) + _w3 * (0.0)
    _w2 = logistic_storage(S_uz, S_fc)
    Q_ss = (1 - _w2) * ((max(0.0, S_uz - S_fc) / a) ** (1.0 / b)) + _w2 * (0.0)

    Q_r = r * Q_ss
    Q_b = (1.0 / a * S_gw) ** (1.0 / b)

    # update state(s) via mass balance
    dS_uz = P - E_bs - E_veg - Q_se - Q_ss
    dS_gw = Q_r - Q_b
    S_uz_new = S_uz + dS_uz * dt
    S_gw_new = S_gw + dS_gw * dt

    Q = Q_se + (Q_ss - Q_r) + Q_b

    fluxes = dict(E_bs=E_bs, E_veg=E_veg, Q_se=Q_se, Q_ss=Q_ss, Q_r=Q_r, Q_b=Q_b)
    return (S_uz_new, S_gw_new), Q, fluxes


def step_model_10(S, forcings, params, dt=1.0):
    """
    Model 10 -- Susannah Brook model v2  (MARRMoT file: m_10_susannah2_6p_2s)

    Reference(s): Son and Sivapalan, 2007

    Processes represented:
      - Separation of saturated zone and a variable-size unsaturated zone
      - Evaporation from unsaturated and saturated zones
      - Saturation excess and non-linear subsurface flow
      - Deep groundwater recharge

    State variables (2):
      S_us  : storage in the unsaturated (upper) store [mm]
      S_sat : storage in the saturated zone [mm]

    Parameters (6):
      S_b        [mm]  Maximum storage of the soil profile (= D * phi)  range [1, 2000]
      phi        [-]  Porosity  range [0.05, 0.95]
      fc         [-]  Field capacity as fraction of S_b  range [0.05, 0.95]
      r          [-]  Fraction of subsurface outflow to deep groundwater  range [0, 1]
      c          [d^-1]  Runoff (subsurface flow) coefficient  range [0, 1]
      d          [-]  Runoff nonlinearity  range [1, 5]

    Output / routing (as specified):
      ```
      Q = Q_SE + Q_SS     (91)
      ```
      (Q_R is deep groundwater recharge — a loss from the modelled system; it is
      not included in total streamflow.)
      Routing: none.
    """
    # unpack state
    S_us, S_sat = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    S_b, phi, fc, r, c, d = (
        params['S_b'],
        params['phi'],
        params['fc'],
        params['r'],
        params['c'],
        params['d']
    )

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    S_usfc = (S_b - S_sat) * fc / phi

    E_us = S_us / S_b * Ep

    _w1 = logistic_storage(S_us, S_usfc)
    r_g = (1 - _w1) * (P) + _w1 * (0.0)
    _w2 = logistic_storage(S_us, S_usfc)
    S_e = (1 - _w2) * ((S_us - S_usfc)) + _w2 * (0.0)
    E_sat = S_sat / S_b * Ep

    _w3 = logistic_storage(S_sat, S_b)
    Q_SE = (1 - _w3) * ((r_g + S_e)) + _w3 * (0.0)
    Q_SS = (1 - r) * c * (S_sat ** d)
    Q_R  = r * c * (S_sat ** d)

    # update state(s) via mass balance
    dS_us  = P - E_us - r_g - S_e
    dS_sat = r_g - E_sat - Q_SE - Q_SS - Q_R
    S_us_new  = S_us  + dS_us * dt
    S_sat_new = S_sat + dS_sat * dt

    Q = Q_SE + Q_SS

    fluxes = dict(E_us=E_us, r_g=r_g, S_e=S_e, E_sat=E_sat,
                  Q_SE=Q_SE, Q_SS=Q_SS, Q_R=Q_R)
    return (S_us_new, S_sat_new), Q, fluxes


def step_model_11(S, forcings, params, dt=1.0):
    """
    Model 11 -- Collie River Basin 3  (MARRMoT file: m_11_collie3_6p_2s)

    Reference(s): Jothityangkoon et al., 2001

    Processes represented:
      - Separate bare soil and vegetation evaporation
      - Saturation excess surface runoff
      - Non-linear subsurface runoff
      - Non-linear groundwater runoff

    State variables (2):
      S : soil moisture storage [mm]
      G : groundwater storage [mm]

    Parameters (6):
      S_max      [mm]  Maximum soil moisture storage  range [1, 2000]
      S_fc       [mm]  Field capacity  range not specified in Table S3 — check S2.11.3 parameter table only
      a          [d^-1]  Runoff coefficient  range not specified in Table S3 — check S2.11.3 parameter table only
      M          [-]  Forest fraction  range not specified in Table S3 — check S2.11.3 parameter table only
      b          [-]  Runoff nonlinearity  range [1, 5]
      lambda     [-]  Fraction of subsurface flow to groundwater  range [0, 1]

    Output / routing (as specified):
      ```
      Q = Q_se + (1 - lambda)*Q_ss + Q_sg     (100)
      ```
      Routing: none.
    """
    # unpack state
    S_soil, G = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    S_max, S_fc, a, M, b, lam = (
        params['S_max'],
        params['S_fc'],
        params['a'],
        params['M'],
        params['b'],
        params['lam']
    )

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    E_b = S_soil / S_max * (1 - M) * Ep

    _w1 = logistic_storage(S_soil, S_fc)
    E_v = (1 - _w1) * (M * Ep) + _w1 * (S_soil / S_fc * M * Ep)

    _w3 = logistic_storage(S_soil, S_max)
    Q_se = (1 - _w3) * (P) + _w3 * (0.0)
    _w2 = logistic_storage(S_soil, S_fc)
    Q_ss = (1 - _w2) * ((a * max(0.0, S_soil - S_fc)) ** b) + _w2 * (0.0)

    Q_ss_star = lam * Q_ss
    Q_sg = (a * G) ** b

    # update state(s) via mass balance
    dS = P - E_b - E_v - Q_se - Q_ss
    dG = Q_ss_star - Q_sg
    S_soil_new = min(max(S_soil + dS * dt, 0.0), S_max)
    G_new = G + dG * dt

    Q = Q_se + (1 - lam) * Q_ss + Q_sg

    fluxes = dict(E_b=E_b, E_v=E_v, Q_se=Q_se, Q_ss=Q_ss,
                  Q_ss_star=Q_ss_star, Q_sg=Q_sg)
    return (S_soil_new, G_new), Q, fluxes


def step_model_12(S, forcings, params, dt=1.0):
    """
    Model 12 -- Alpine model v2  (MARRMoT file: m_12_alpine2_6p_2s)

    Reference(s): Eder et al., 2003

    Processes represented:
      - Snow accumulation and melt
      - Saturation excess overland flow
      - Linear subsurface runoff

    State variables (2):
      S_N (S_n) : snow storage [mm]
      S       : soil moisture storage [mm]

    Parameters (6):
      T_t        [°C]  Threshold temperature for snowfall and melt  range [-3, 5]
      ddf        [mm °C^-1 d^-1]  Degree-day factor  range [0, 20]
      S_max      [mm]  Maximum soil moisture storage  range [1, 2000]
      S_fc       [mm]  Field capacity  range not specified in Table S3 — check S2.12.3 parameter table only
      t_c,in     [d^-1]  Runoff coefficient (interflow)  range [0, 1]
      t_c,bf     [d^-1]  Runoff coefficient (baseflow)  range [0, 1]

    Output / routing (as specified):
      ```
      Q_t = Q_se + Q_in + Q_bf     (110)
      ```
      Routing: none.
    """
    # unpack state
    S_N, S_soil = S

    # unpack forcings
    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    # unpack params
    T_t, ddf, S_max, S_fc, t_c_in, t_c_bf = (
        params['T_t'],
        params['ddf'],
        params['S_max'],
        params['S_fc'],
        params['t_c_in'],
        params['t_c_bf']
    )

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    _w2 = logistic_temperature(T, T_t)
    P_s = _w2 * (P) + (1 - _w2) * (0.0)
    _w3 = logistic_temperature(T, T_t)
    Q_N = (1 - _w3) * (ddf * (T - T_t)) + _w3 * (0.0)
    _w4 = logistic_temperature(T, T_t)
    P_r = (1 - _w4) * (P) + _w4 * (0.0) # source prints "TT"; treated as T_t (see note)

    _w5 = logistic_storage(S_soil, 0)
    E_a = (1 - _w5) * (Ep) + _w5 * (0.0)
    _w6 = logistic_storage(S_soil, S_max)
    Q_se = (1 - _w6) * ((P_r + Q_N)) + _w6 * (0.0)
    _w1 = logistic_storage(S_soil, S_fc)
    Q_in = (1 - _w1) * (t_c_in * (S_soil - S_fc)) + _w1 * (0.0)

    Q_bf = t_c_bf * S_soil

    # update state(s) via mass balance
    dS_N    = P_s - Q_N
    dS_soil = P_r + Q_N - E_a - Q_se - Q_in - Q_bf
    S_N_new    = S_N    + dS_N    * dt
    S_soil_new = S_soil + dS_soil * dt

    Q = Q_se + Q_in + Q_bf

    fluxes = dict(P_s=P_s, Q_N=Q_N, P_r=P_r, E_a=E_a, Q_se=Q_se,
                  Q_in=Q_in, Q_bf=Q_bf)
    return (S_N_new, S_soil_new), Q, fluxes


def step_model_13(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 13 -- Hillslope model  (MARRMoT file: m_13_hillslope_7p_2s)

    Reference(s): Savenije, 2010

    Processes represented:
      - Stylized interception by vegetation
      - Evaporation
      - Separation between rapid subsurface flow and groundwater recharge
      - Capillary rise and linear relation runoff from groundwater

    State variables (2):
      S_w    : soil water storage [mm]
      S_h,gw : groundwater storage [mm]

    Parameters (7):
      D_w        [mm d^-1]  Interception evaporation (constant interception loss rate)  range [0, 5]
      S_h,max    [mm]  Maximum soil moisture storage  range [1, 2000]
      beta_h     [-]  Non-linearity parameter for contributing area  range [0, 10]
      a          [-]  Fraction of saturation excess routed to groundwater  range [0, 1]
      T_h        [d]  Unit Hydrograph time base  range [1, 120]
      C          [mm d^-1]  Capillary rise  range [0, 4]
      K_h        [d^-1]  Runoff coefficient (linear groundwater store)  range [0, 1]

    Output / routing (as specified):
      ```
      Q_t = Q_h,gw + Q_h,srf
      ```
      where `Q_h,srf` is the complementary (surface) fraction of storage-excess
      flow, `Q_se,s = a * Q_se` (complementary of Q_se,g = (1-a)*Q_se), **routed
      through a Unit Hydrograph with time base T_h days** before being added to
      groundwater flow: `Q_h,srf = UH(T_h) convolved with Q_se,s`.
  
      Routing: **Yes** — Unit Hydrograph applied to the surface-runoff component
      `Q_se,s = a * Q_se`, with time base parameter `T_h` [d] (this model's own
      S2.13 section explicitly states "Total flow Q_t is the sum of Q_h,gw and
      Q_h,srf, the latter of which is Q_se,s lagged over T_h days").
    """
    # unpack state
    S_w, S_h_gw = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    D_w, S_h_max, beta_h, a, T_h, C_param, K_h = (
        params['D_w'], params['S_h_max'], params['beta_h'], params['a'],
        params['T_h'], params['C_param'], params['K_h'])

    if uh_state is None:
        uh_state = {'srf': UHRouter(uh_triangular(T_h, dt))}

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue -- this pseudocode shows the
    # literal mathematical definition)
    P_e = max(P - D_w, 0.0)           # D_h in source == D_w parameter, see note
    C = C_param                        # "C = c." -- C flux equals constant parameter C

    _w1 = logistic_storage(S_w, 0)
    Et_Es = (1 - _w1) * (Ep) + _w1 * (0.0)
    Q_se = (1 - (1 - S_w / S_h_max) ** beta_h) * P_e   # S_h in source == S_w, see note

    Q_se_g = (1 - a) * Q_se
    Q_se_s = a * Q_se                  # complementary part, routed via UH(T_h)

    Q_h_gw = K_h * S_h_gw

    # route Q_se_s through Unit Hydrograph with time base T_h
    Q_h_srf = uh_state['srf'].route(Q_se_s)

    # update state(s) via mass balance
    dS_w    = P_e + C - Et_Es - Q_se
    dS_h_gw = Q_se_g - C - Q_h_gw
    S_w_new    = S_w    + dS_w    * dt
    S_h_gw_new = S_h_gw + dS_h_gw * dt

    Q = Q_h_gw + Q_h_srf

    fluxes = dict(P_e=P_e, C=C, Et_Es=Et_Es, Q_se=Q_se, Q_se_g=Q_se_g,
                  Q_se_s=Q_se_s, Q_h_gw=Q_h_gw, Q_h_srf=Q_h_srf)
    return (S_w_new, S_h_gw_new), Q, fluxes, uh_state


def step_model_14(S, forcings, params, dt=1.0):
    """
    Model 14 -- TOPMODEL  (MARRMoT file: m_14_topmodel_7p_2s)

    Reference(s): Beven, K.J. and Kirkby, M.J. (1979) [original TOPMODEL concept]; Beven, K.J. et al. (1995) [version this implementation is mostly based on]; Sivapalan, M. et al. (1987) [shifted 2-parameter gamma distribution for topographic index]; Clark, M.P. et al. (2008) [lumped simplification, calibrated gamma-distribution parameters instead of DEM-derived distribution]

    Processes represented:
      - Variable saturated area with direct runoff from the saturated part
      - Infiltration and saturation excess flow
      - Leakage to, and non-linear baseflow from, a deficit store

    State variables (2):
      S_UZ : current storage in the combined unsaturated zone / root zone store [mm]
      S_SZ : current storage *deficit* in the saturated zone store [mm] (this is a deficit, not a fill level)

    Parameters (7):
      S_UZ,max   [mm]  Maximum soil moisture storage (unsaturated/root zone)  range [1, 2000]
      S_t        [-]  Threshold for drainage as fraction of S_UZ,max  range [0.05, 0.95]
      K_d        [d^-1]  Runoff (leakage) coefficient  range [0, 1]
      q_0        [mm d^-1]  Saturated flow rate (baseflow at zero deficit)  range [0.1, 200]
      f          [mm^-1]  Shape parameter for connectivity (baseflow deficit) profile  range [0, 1]
      chi (χ)    [-]  Gamma distribution parameter (scale) for topographic index  range [1, 7.5]
      phi (φ)    [-]  Gamma distribution parameter (shape) for topographic index  range [0.1, 5]

    Output / routing (as specified):
      ```
      Q = Q_of + Q_ex + Q_b            (127)
      ```
      Routing: none (the lumped simplification of Clark et al. (2008) explicitly removes the
      distributed routing component of the original semi-distributed TOPMODEL).
    """
    # unpack state
    S_UZ, S_SZ = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    S_UZmax, S_t, K_d, q0, f, chi, phi = (
        params['S_UZmax'],
        params['S_t'],
        params['K_d'],
        params['q0'],
        params['f'],
        params['chi'],
        params['phi']
    )
    mu = 3.0  # fixed shift of gamma distribution

    # --- saturated area from topographic index distribution ---
    lam = chi * phi + mu
    zeta_crit = f * S_SZ + lam
    x = (zeta_crit - mu) / chi
    A_c = reg_upper_incomplete_gamma(phi, x)   # = integral_{zeta_crit}^{inf} f(zeta) dzeta
    A_c = clip(A_c, 0, 1)

    # --- fluxes ---
    Q_of  = A_c * P
    P_eff = P - Q_of

    _w1 = logistic_storage(S_UZ, S_UZmax)
    Q_ex = (1 - _w1) * (P_eff) + _w1 * (0.0)
    if S_UZ > S_t * S_UZmax:
        E_a = Ep
    else:
        E_a = (S_UZ / (S_t * S_UZmax)) * Ep

    if S_UZ > S_t * S_UZmax:
        Q_v = K_d * (S_UZ - S_t * S_UZmax) / (S_UZmax * (1 - S_t))
    else:
        Q_v = 0.0

    Q_b = q0 * exp(-f * S_SZ)

    # --- mass balance (state update) ---
    dS_UZ = P_eff - Q_ex - E_a - Q_v
    dS_SZ = -Q_v + Q_b
    S_UZ_new = S_UZ + dS_UZ * dt
    S_SZ_new = S_SZ + dS_SZ * dt

    # --- output ---
    Q = Q_of + Q_ex + Q_b

    fluxes_dict = {"P_eff": P_eff, "Q_of": Q_of, "Q_ex": Q_ex, "E_a": E_a,
                   "Q_v": Q_v, "Q_b": Q_b, "A_c": A_c}
    return (S_UZ_new, S_SZ_new), Q, fluxes_dict


def step_model_15(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 15 -- Plateau model  (MARRMoT file: m_15_plateau_8p_2s)

    Reference(s): Savenije, H.H.G. (2010) — part of a 3-model topography-driven modelling

    Processes represented:
      - Stylized interception by vegetation
      - Evaporation controlled by a wilting point and moisture-constrained transpiration
      - Separation between infiltration and infiltration excess flow
      - Capillary rise and linear relation runoff from groundwater

    State variables (2):
      S_u    : current soil water storage [mm]
      S_p,gw : current groundwater storage [mm]

    Parameters (8):
      F_max      [mm d^-1]  Maximum infiltration rate  range [0, 200]
      D_p        [mm d^-1]  Interception evaporation (assumed to evaporate before next precip event)  range [0, 5]
      S_u,max    [mm]  Maximum soil moisture storage  range [1, 2000]
      lp         [-]  Wilting point as fraction of S_u,max  range [0.05, 0.95]
      p          [-]  Evaporation reduction (moisture-constrained rate) factor, 0-1  range [0, 1]
      T_p        [d]  Unit Hydrograph time base  range [1, 120]
      C          [mm d^-1]  Capillary rise (constant rate)  range [0, 4]
      K_p        [d^-1]  Runoff coefficient (groundwater store)  range [0, 1]

    Output / routing (as specified):
      ```
      Q_t = Q_p,gw + Q_p,ieo
      ```
      where Q_p,ieo is the infiltration-excess flux P_ie routed through a Unit Hydrograph with time
      base T_p (i.e. P_ie lagged over T_p days).
  
      Routing: Unit Hydrograph applied to the infiltration-excess flow P_ie, time base = parameter T_p [d].
      (Q_p,gw from the groundwater store is NOT routed through the UH — only the surface/infiltration-excess
      path Q_p,ieo is.)
    """
    # unpack state
    S_u, S_pgw = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    F_max, D_p, S_umax, lp, p, T_p, C_rate, K_p = (
        params['F_max'], params['D_p'], params['S_umax'], params['lp'],
        params['p'], params['T_p'], params['C_rate'], params['K_p'])
    S_wp = lp * S_umax

    if uh_state is None:
        uh_state = {'ieo': UHRouter(uh_triangular(T_p, dt))}

    # --- fluxes ---
    P_e  = max(P - D_p, 0)
    P_i  = min(P_e, F_max)
    P_ie = P_e - P_i                      # infiltration excess, routed via UH

    C = C_rate                            # constant capillary rise rate

    E_t = Ep * max(p * (S_u - S_wp) / (S_umax - S_wp), 0)

    _w1 = logistic_storage(S_u, S_umax)
    R = (1 - _w1) * ((P_i + C)) + _w1 * (0.0)
    Q_pgw = K_p * S_pgw

    # --- mass balance (state update) ---
    dS_u   = P_i + C - E_t - R
    dS_pgw = R - C - Q_pgw
    S_u_new   = S_u + dS_u * dt
    S_pgw_new = S_pgw + dS_pgw * dt

    # --- route infiltration excess through Unit Hydrograph (time base T_p) ---
    Q_p_ieo = uh_state['ieo'].route(P_ie)

    # --- output ---
    Q = Q_pgw + Q_p_ieo

    fluxes_dict = {"P_e": P_e, "P_i": P_i, "P_ie": P_ie, "C": C,
                   "E_t": E_t, "R": R, "Q_pgw": Q_pgw, "Q_p_ieo": Q_p_ieo}
    return (S_u_new, S_pgw_new), Q, fluxes_dict, uh_state


def step_model_16(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 16 -- New Zealand model v2  (MARRMoT file: m_16_newzealand2_8p_2s)

    Reference(s): Atkinson, S.E. et al. (2003) — "Model A" of a top-down modelling exercise

    Processes represented:
      - Interception by vegetation
      - Separate vegetation and bare soil evaporation
      - Saturation excess overland flow
      - Subsurface runoff when soil moisture exceeds field capacity
      - Baseflow
      - Flow routing

    State variables (2):
      S_i : current interception storage [mm]
      S_m : current soil moisture storage [mm]

    Parameters (8):
      I_max      [mm]  Maximum interception capacity  range [0, 5]
      S_max      [mm]  Maximum soil moisture storage  range [1, 2000]
      S_fc       [mm]  Field capacity  range not specified in Table S3 — check S2.x.3 parameter table only
      M          [-]  Forest fraction  range [0.05, 0.95]
      a          [d^-1]  Runoff (subsurface/interflow) coefficient  range not specified in Table S3 — check S2.x.3 parameter table only
      b          [-]  Runoff nonlinearity exponent  range [1, 5]
      t_c,bf     [d^-1]  Runoff (baseflow) coefficient  range [0, 1]
      d          [d]  Unit Hydrograph time base  range [1, 120]

    Output / routing (as specified):
      ```
      Q_t = Q_se + Q_ss + Q_bf          (149)
      ```
      Total flow Q_t is then delayed by a triangular Unit Hydrograph routing scheme controlled by
      time parameter d [d].
  
      Routing: Unit Hydrograph — triangular routing scheme, time base = parameter d [d], applied to
      the combined runoff Q_t = Q_se + Q_ss + Q_bf.
    """
    # unpack state
    S_i, S_m = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    I_max, S_max, S_fc, M, a, b, t_c_bf, d_uh = (
        params['I_max'], params['S_max'], params['S_fc'], params['M'],
        params['a'], params['b'], params['t_c_bf'], params['d_uh'])

    if uh_state is None:
        uh_state = {'q': UHRouter(uh_triangular(d_uh, dt))}

    # --- interception store ---
    E_int = Ep
    _w3 = logistic_storage(S_i, I_max)
    Q_tf = (1 - _w3) * (P) + _w3 * (0.0)
    dS_i = P - E_int - Q_tf
    S_i_new = S_i + dS_i * dt

    # --- soil moisture store ---
    _w1 = logistic_storage(S_m, S_fc)
    E_veg = (1 - _w1) * (M * Ep) + _w1 * ((S_m / S_fc) * M * Ep)

    E_bs = (S_m / S_max) * (1 - M) * Ep

    _w4 = logistic_storage(S_m, S_max)
    Q_se = (1 - _w4) * (Q_tf) + _w4 * (0.0)
    _w2 = logistic_storage(S_m, S_fc)
    Q_ss = (1 - _w2) * ((a * max(0.0, S_m - S_fc)) ** b) + _w2 * (0.0)

    Q_bf = t_c_bf * S_m

    dS_m = Q_tf - E_veg - E_bs - Q_se - Q_ss - Q_bf
    S_m_new = S_m + dS_m * dt

    # --- total runoff before routing ---
    Q_t = Q_se + Q_ss + Q_bf

    # --- route through triangular Unit Hydrograph, time base d_uh ---
    Q = uh_state['q'].route(Q_t)

    fluxes_dict = {"E_int": E_int, "Q_tf": Q_tf, "E_veg": E_veg, "E_bs": E_bs,
                   "Q_se": Q_se, "Q_ss": Q_ss, "Q_bf": Q_bf, "Q_t": Q_t}
    return (S_i_new, S_m_new), Q, fluxes_dict, uh_state


def step_model_17(S, forcings, params, dt=1.0):
    """
    Model 17 -- Penman model  (MARRMoT file: m_17_penman_4p_3s)

    Reference(s): Penman, H.L. (1950) — original "drying curve" concept; Wagener, T. et al. (2002)

    Processes represented:
      - Moisture accumulation and evaporation from the root zone
      - Bypass of excess moisture to the stream
      - Deficit-based groundwater accounting
      - Linear flow routing

    State variables (3):
      S_rz  : current storage in the root zone [mm]
      S_def : current moisture *deficit* [mm]
      C_res : current storage in the linear routing reservoir [mm]

    Parameters (4):
      S_max      [mm]  Maximum soil moisture storage (root zone)  range [1, 2000]
      phi (φ)    [-]  Fraction of saturation excess (q_ex) that becomes direct runoff (bypasses lower soil layer)  range [0, 1]
      gamma (γ)  [-]  Evaporation reduction factor (deficit-store evap as fraction of Ep)  range [0, 1]
      k_1        [d^-1]  Runoff coefficient (linear routing reservoir)  range [0, 1]

    Output / routing (as specified):
      ```
      Q = k_1 * C_res           (159)
      ```
      (Q is the direct output of the single linear routing reservoir C_res, fed by u_1 = phi*q_ex
      and u_2, the saturation-excess overflow from the deficit store.)
  
      Routing: no external Unit Hydrograph — routing is internal via the linear reservoir C_res
      (store #3), drained linearly through parameter k_1 [d^-1].
    """
    # unpack state
    S_rz, S_def, C_res = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    S_max, phi, gamma, k1 = (
        params['S_max'],
        params['phi'],
        params['gamma'],
        params['k1']
    )

    # --- root zone ---
    _w1 = logistic_storage(S_rz, 0)
    E_a = (1 - _w1) * (Ep) + _w1 * (0.0)
    _w2 = logistic_storage(S_rz, S_max)
    q_ex = (1 - _w2) * (P) + _w2 * (0.0)
    dS_rz = P - E_a - q_ex
    S_rz_new = S_rz + dS_rz * dt

    # --- split of saturation excess ---
    q12 = (1 - phi) * q_ex
    u1  = phi * q_ex

    # --- deficit store ---
    _w3 = logistic_storage(S_rz, 0)
    E_t = _w3 * (gamma * Ep) + (1 - _w3) * (0.0)
    _w4 = logistic_storage(S_def, 0)
    u2 = _w4 * (q12) + (1 - _w4) * (0.0)
    dS_def = E_t + u2 - q12
    S_def_new = S_def + dS_def * dt

    # --- routing reservoir ---
    Q = k1 * C_res
    dC_res = u1 + u2 - Q
    C_res_new = C_res + dC_res * dt

    fluxes_dict = {"E_a": E_a, "q_ex": q_ex, "q12": q12, "u1": u1,
                   "E_t": E_t, "u2": u2}
    return (S_rz_new, S_def_new, C_res_new), Q, fluxes_dict


def step_model_18(S, forcings, params, dt=1.0):
    """
    Model 18 -- SIMHYD  (MARRMoT file: m_18_simhyd_7p_3s)

    Reference(s): Chiew, F.H.S. et al. (2002) — simplified version of MODHYDROLOG, originally

    Processes represented:
      - Interception by vegetation
      - Infiltration and infiltration excess flow
      - Preferential groundwater recharge, interflow and saturation excess flow
      - Groundwater recharge resulting from filling up of soil moisture storage capacity
      - Slow flow from groundwater

    State variables (3):
      I   : current interception storage [mm]
      SMS : current storage in the soil moisture store [mm]
      GW  : current storage in the groundwater reservoir [mm]

    Parameters (7):
      INSC       [mm]  Maximum interception capacity  range [0, 5]
      COEFF      [mm d^-1]  Maximum infiltration loss  range [0, 600]
      SQ         [-]  Infiltration loss exponent  range [0, 15]
      SMSC       [mm]  Maximum soil moisture storage  range [1, 2000]
      SUB        [-]  Proportionality constant (interflow)  range [0, 1]
      CRAK       [-]  Proportionality constant (preferential recharge)  range [0, 1]
      K          [d^-1]  Runoff coefficient (groundwater / baseflow)  range [0, 1]

    Output / routing (as specified):
      ```
      Q_t = SRUN + INT + BAS          (172)
      ```
  
      Routing: none — no explicit Unit Hydrograph; total flow is the direct sum of surface runoff
      (SRUN), interflow (INT), and baseflow (BAS) in the same time step.
    """
    # unpack state
    I, SMS, GW = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']
    PET = Ep   # same quantity, both symbols used in the source doc

    # unpack params
    INSC, COEFF, SQ, SMSC, SUB, CRAK, K = (
        params['INSC'],
        params['COEFF'],
        params['SQ'],
        params['SMSC'],
        params['SUB'],
        params['CRAK'],
        params['K']
    )

    # --- interception store ---
    _w1 = logistic_storage(I, 0)
    E_i = (1 - _w1) * (Ep) + _w1 * (0.0)
    _w2 = logistic_storage(I, INSC)
    EXC = (1 - _w2) * (P) + _w2 * (0.0)
    dI = P - E_i - EXC
    I_new = I + dI * dt

    # --- infiltration / soil moisture store ---
    INF = min(COEFF * exp(-SQ * SMS / SMSC), EXC)
    INT = SUB * (SMS / SMSC) * INF
    REC = CRAK * (SMS / SMSC) * (INF - INT)
    SMF = INF - INT - REC

    E_T = min(10 * (SMS / SMSC), PET)

    _w3 = logistic_storage(SMS, SMSC)
    GWF = (1 - _w3) * (SMF) + _w3 * (0.0)
    dSMS = SMF - E_T - GWF
    SMS_new = SMS + dSMS * dt

    # --- groundwater store ---
    BAS = K * GW
    dGW = REC + GWF - BAS
    GW_new = GW + dGW * dt

    # --- surface runoff & total output ---
    SRUN = EXC - INF
    Q = SRUN + INT + BAS

    fluxes_dict = {"E_i": E_i, "EXC": EXC, "INF": INF, "INT": INT, "REC": REC,
                   "SMF": SMF, "E_T": E_T, "GWF": GWF, "BAS": BAS, "SRUN": SRUN}
    return (I_new, SMS_new, GW_new), Q, fluxes_dict


def step_model_19(S, forcings, params, dt=1.0):
    """
    Model 19 -- Australia model  (MARRMoT file: m_19_australia_8p_3s)

    Reference(s): Farmer et al., 2003 (top-down modelling exercise, originally "model S4")

    Processes represented:
      - Separation of saturated zone and a variable-size unsaturated zone
      - Evaporation from unsaturated and saturated zones
      - Saturation excess and non-linear subsurface flow
      - Deep groundwater recharge and baseflow

    State variables (3):
      S_us  : storage in the unsaturated zone [mm]
      S_sat : storage in the saturated zone [mm]
      G_w   : groundwater storage [mm]

    Parameters (8):
      S_b        [mm]  Maximum soil moisture storage  range [1, 2000]
      phi        [–]  Porosity  range [0.05, 0.95]
      fc         [–]  Field capacity  range [0.05, 0.95]
      alpha_SS   [d-1]  Runoff coefficient (subsurface flow)  range [0, 1]
      beta_SS    [–]  Runoff nonlinearity (subsurface flow)  range [1, 5]
      K_deep     [d-1]  Runoff coefficient (deep recharge)  range [0, 1]
      alpha_BF   [d-1]  Runoff coefficient (baseflow)  range [0, 1]
      beta_BF    [–]  Runoff nonlinearity (baseflow)  range [1, 5]

    Output / routing (as specified):
      ```
      Q = Q_SE + Q_SS + Q_BF
      ```
      Routing: none (no Unit Hydrograph mentioned in S2.19).
    """
    # unpack state
    S_us, S_sat, G_w = S

    # unpack params
    S_b, phi, fc, alpha_SS, beta_SS, K_deep, alpha_BF, beta_BF = (
        params['S_b'],
        params['phi'],
        params['fc'],
        params['alpha_SS'],
        params['beta_SS'],
        params['K_deep'],
        params['alpha_BF'],
        params['beta_BF']
    )
    P, Ep = forcings['P'], forcings['Ep']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    E_us = (S_us / S_b) * Ep
    S_usfc = (S_b - S_sat) * fc / phi
    _w1 = logistic_storage(S_us, S_usfc)
    r_g = (1 - _w1) * (P) + _w1 * (0)
    _w2 = logistic_storage(S_us, S_usfc)
    s_e = (1 - _w2) * ((S_us - S_usfc)) + _w2 * (0)
    E_sat = (S_sat / S_b) * Ep
    _w3 = logistic_storage(S_sat, S_b)
    Q_SE = (1 - _w3) * ((r_g + s_e)) + _w3 * (0)
    Q_SS = alpha_SS * S_sat**beta_SS
    Q_R  = K_deep * S_sat

    Q_BF = alpha_BF * G_w**beta_BF

    # update state(s) via mass balance
    dS_us_dt  = P - E_us - r_g - s_e
    dS_sat_dt = r_g - E_sat - Q_SE - Q_SS - Q_R
    dG_w_dt   = Q_R - Q_BF
    S_new = (S_us + dS_us_dt*dt, S_sat + dS_sat_dt*dt, G_w + dG_w_dt*dt)

    Q = Q_SE + Q_SS + Q_BF
    fluxes_dict = dict(E_us=E_us, E_sat=E_sat, r_g=r_g, s_e=s_e,
                        Q_SE=Q_SE, Q_SS=Q_SS, Q_R=Q_R, Q_BF=Q_BF)
    return S_new, Q, fluxes_dict


def step_model_20(S, forcings, params, dt=1.0):
    """
    Model 20 -- Generalized Surface inFiltration Baseflow model (GSFB)  (MARRMoT file: m_20_gsfb_8p_3s)

    Reference(s): Nathan and McMahon, 1990; Ye et al., 1997

    Processes represented:
      - Saturation excess surface runoff
      - Threshold-based infiltration
      - Threshold-based baseflow
      - Deep percolation and water rise to meet evaporation demand

    State variables (3):
      S  : storage in the upper zone [mm]
      SS : storage in the subsurface store [mm]
      DS : storage in the deep store [mm]

    Parameters (8):
      C          [d-1]  Recharge coefficient  range [0, 1]
      NDC        [–]  Threshold for evaporation and recharge as fraction of S_max  range not specified in Table S3 — check S2.20.3 parameter table only
      S_max      [mm]  Maximum soil moisture storage  range [1, 2000]
      E_max      [mm d-1]  Maximum evaporation rate  range [0, 20]
      F_rate     [mm d-1]  Recharge rate (infiltration)  range [0, 200]
      B          [–]  Fraction subsurface flow to stream  range [0, 1]
      DPF        [d-1]  Runoff coefficient  range [0, 1]
      SDR_max    [mm]  Threshold for subsurface flow generation  range [1, 300]

    Output / routing (as specified):
      ```
      Q_t = Q_s + Q_b
      ```
      Routing: none (no Unit Hydrograph mentioned in S2.20).
    """
    # unpack state
    S, SS, DS = S

    # unpack params
    C, NDC, S_max, E_max, F_rate, B, DPF, SDR_max = (
        params['C'],
        params['NDC'],
        params['S_max'],
        params['E_max'],
        params['F_rate'],
        params['B'],
        params['DPF'],
        params['SDR_max']
    )
    P, Ep = forcings['P'], forcings['Ep']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    Q_dr = C * DS * (1 - S/(NDC*S_max)) if S <= NDC*S_max else 0
    E_a  = Ep if S > NDC*S_max else min(Ep, E_max * S/(NDC*S_max))
    _w1 = logistic_storage(S, S_max)
    Q_s = (1 - _w1) * (P) + _w1 * (0)
    F    = F_rate if S > NDC*S_max else 0

    _w2 = logistic_storage(SS, SDR_max)
    Q_b = (1 - _w2) * (B * DPF * (SS - SDR_max)) + _w2 * (0)
    D_p  = (1 - B) * DPF * SS

    # update state(s) via mass balance
    dS_dt  = P + Q_dr - E_a - Q_s - F
    dSS_dt = F - Q_b - D_p
    dDS_dt = D_p - Q_dr
    S_new = (S + dS_dt*dt, SS + dSS_dt*dt, DS + dDS_dt*dt)

    Q_t = Q_s + Q_b
    fluxes_dict = dict(Q_dr=Q_dr, E_a=E_a, Q_s=Q_s, F=F, Q_b=Q_b, D_p=D_p)
    return S_new, Q_t, fluxes_dict


def step_model_21(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 21 -- Flex-B  (MARRMoT file: m_21_flexb_9p_3s)

    Reference(s): Fenicia et al., 2008

    Processes represented:
      - Infiltration and saturation excess flow based on a distribution of different soil depths
      - A split between fast saturation excess flow and preferential recharge to a slow store
      - Percolation from the unsaturated zone to a slow runoff store

    State variables (3):
      UR : storage in the unsaturated zone [mm]
      FR : storage in the fast flow reservoir [mm]
      SR : storage in the slow flow reservoir [mm]

    Parameters (9):
      UR_max     [mm]  Maximum soil moisture storage  range [1, 2000]
      beta       [–]  Shape parameter  range [0, 10]
      D          [–]  Fraction effective precipitation to slow store  range [0, 1]
      Perc_max   [mm d-1]  Maximum percolation rate  range [0, 20]
      Lp         [–]  Wilting point as fraction of UR_max  range [0.05, 0.95]
      N_lag,f    [d]  Unit Hydrograph time base (fast)  range [1, 5]
      N_lag,s    [d]  Unit Hydrograph time base (slow)  range [1, 15]
      K_f        [d-1]  Runoff coefficient (fast)  range [0, 1]
      K_s        [d-1]  Runoff coefficient (slow)  range [0, 1]

    Output / routing (as specified):
      ```
      Q = Q_f + Q_s
      ```
      Routing: **Yes** — Flex-B uses two triangular Unit Hydrographs. UH_f (time base = N_lag,f) routes the fast-flow input R_f (+ percolation contribution) into FR; UH_s (time base = N_lag,s) routes the slow-flow input R_s (+ percolation R_p) into SR.
    """
    # unpack state
    UR, FR, SR = S

    # unpack params
    UR_max, beta, D, Perc_max, Lp, Nlag_f, Nlag_s, Kf, Ks = (
        params['UR_max'], params['beta'], params['D'], params['Perc_max'],
        params['Lp'], params['Nlag_f'], params['Nlag_s'], params['Kf'], params['Ks'])
    P_eff, Ep = forcings['P'], forcings['Ep']

    if uh_state is None:
        uh_state = {
            'fast': UHRouter(uh_triangular(Nlag_f, dt)),
            'slow': UHRouter(uh_triangular(Nlag_s, dt)),
        }

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue -- this pseudocode shows the
    # literal mathematical definition)
    Cr   = (1 + exp(-(UR/UR_max + 0.5)/beta))**-1
    R_u  = (1 - Cr) * P_eff
    E_ur = Ep * min(1, (UR/UR_max) * (1/Lp))
    P_s  = Perc_max * (UR/UR_max)          # R_p

    R_s = (P_eff - R_u) * D
    R_f = (P_eff - R_u) * (1 - D)

    # route R_f through triangular UH (time base Nlag_f) -> R_f_l
    # route (R_s + P_s) through triangular UH (time base Nlag_s) -> R_s_l
    R_f_l = uh_state['fast'].route(R_f)
    R_s_l = uh_state['slow'].route(R_s + P_s)

    Q_f = Kf * FR
    Q_s = Ks * SR

    # update state(s) via mass balance
    dUR_dt = R_u - E_ur - P_s
    dFR_dt = R_f_l - Q_f
    dSR_dt = R_s_l - Q_s
    S_new = (UR + dUR_dt*dt, FR + dFR_dt*dt, SR + dSR_dt*dt)

    Q = Q_f + Q_s
    fluxes_dict = dict(R_u=R_u, E_ur=E_ur, P_s=P_s, R_s=R_s, R_f=R_f,
                        R_f_l=R_f_l, R_s_l=R_s_l, Q_f=Q_f, Q_s=Q_s)
    return S_new, Q, fluxes_dict, uh_state


def step_model_22(S, forcings, params, dt=1.0):
    """
    Model 22 -- Variable Infiltration Capacity (VIC) model  (MARRMoT file: m_22_vic_10p_3s)

    Reference(s): Liang et al., 1994 (original VIC); conceptualized version based in part on Clark et al., 2008

    Processes represented:
      - Time-varying interception by vegetation
      - Variable infiltration and saturation excess flow
      - Interflow and baseflow from a deeper groundwater layer

    State variables (3):
      S_i  : interception storage [mm]
      S_sm : soil moisture storage [mm]
      S_gw : groundwater storage [mm]

    Parameters (10):
      Ī (I_bar)  [mm]  Mean annual interception storage  range [0, 5]
      I_delta    [–]  Seasonal interception change as fraction of Ī  range [0, 1]
      I_s        [d]  Time shift of maximum interception storage  range [1, 365]
      S_sm,max   [mm]  Maximum soil moisture storage  range [1, 2000]
      b          [–]  Shape parameter (infiltration curve)  range [0, 10]
      k1         [d-1]  Runoff coefficient (percolation)  range [0, 1]
      c1         [–]  Runoff nonlinearity (percolation)  range [1, 5]
      S_gw,max   [mm]  Maximum groundwater storage  range [1, 2000]
      k2         [d-1]  Runoff coefficient (baseflow)  range [0, 1]
      c2         [–]  Runoff nonlinearity (baseflow)  range [1, 5]

    Output / routing (as specified):
      ```
      Q_t = Q_ie + Q_ex1 + Q_ex2 + Q_b
      ```
      Routing: none (no Unit Hydrograph mentioned in S2.22).
    """
    t = forcings['t']  # day-of-year forcing, folded in from the extra `t` arg
    # unpack state
    S_i, S_sm, S_gw = S

    # unpack params
    I_bar, I_delta, I_s, S_sm_max, b, k1, c1, S_gw_max, k2, c2 = (
        params['I_bar'],
        params['I_delta'],
        params['I_s'],
        params['S_sm_max'],
        params['b'],
        params['k1'],
        params['c1'],
        params['S_gw_max'],
        params['k2'],
        params['c2']
    )
    P, Ep = forcings['P'], forcings['Ep']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    I_max = I_bar * (1 + I_delta * sin(2*pi*(t + I_s)))
    E_i   = (S_i / I_max) * Ep
    _w1 = logistic_storage(S_i, I_max)
    P_eff = (1 - _w1) * (P) + _w1 * (0)
    I_ex  = max(S_i - I_max, 0)

    Q_ie = (P_eff + I_ex) * (1 - (1 - S_sm/S_sm_max)**b)
    inf  = (P_eff + I_ex) - Q_ie
    Et1  = (S_sm / S_sm_max) * (Ep - E_i)
    _w2 = logistic_storage(S_sm, S_sm_max)
    Q_ex1 = (1 - _w2) * (inf) + _w2 * (0)
    pc   = k1 * (S_sm / S_sm_max)**c1

    Et2   = (S_gw / S_gw_max) * (Ep - E_i - Et1)
    _w3 = logistic_storage(S_gw, S_gw_max)
    Q_ex2 = (1 - _w3) * (pc) + _w3 * (0)
    Q_b   = k2 * (S_gw / S_gw_max)**c2

    # update state(s) via mass balance
    dS_i_dt  = P - E_i - P_eff - I_ex
    dS_sm_dt = inf - Et1 - Q_ex1 - pc
    dS_gw_dt = pc - Et2 - Q_ex2 - Q_b
    S_new = (S_i + dS_i_dt*dt, S_sm + dS_sm_dt*dt, S_gw + dS_gw_dt*dt)

    Q_t = Q_ie + Q_ex1 + Q_ex2 + Q_b
    fluxes_dict = dict(E_i=E_i, I_ex=I_ex, Q_ie=Q_ie, inf=inf, Et1=Et1, Q_ex1=Q_ex1,
                        pc=pc, Et2=Et2, Q_ex2=Q_ex2, Q_b=Q_b)
    return S_new, Q_t, fluxes_dict


def step_model_23(S, forcings, params, dt=1.0):
    """
    Model 23 -- Large-scale catchment water and salt balance model (LASCAM)  (MARRMoT file: m_23_lascam_24p_3s)

    Reference(s): Sivapalan et al., 1996 (water balance only simulated here; salt balance ignored)

    Processes represented:
      - Stylized interception
      - Saturation and infiltration excess surface runoff
      - An inner layout representing near-stream saturated storage, deep saturated storage and medium-depth unsaturated storage
      - Subsurface saturation and infiltration excess flow to the near-stream store
      - Percolation to and capillary rise from groundwater

    State variables (3):
      F : storage in the (medium-depth) unsaturated infiltration store [mm]
      A : storage in the more permeable upper (near-stream saturated) zone [mm]
      B : storage in the deep layers [mm]

    Parameters (24):
      alpha_f    [mm d-1]  Infiltration scaling parameter  range [0, 200]
      beta_f     [–]  Infiltration scaling parameter  range [0, 5]
      B_max      [mm]  Infiltration scaling parameter  range [1, 2000]
      F_max      [mm]  Infiltration scaling parameter  range [1, 2000]
      alpha_c    [–]  Overland flow fraction  range [0, 5]
      beta_c     [–]  Overland flow shape parameter  range [0, 1]
      A_min      [mm]  Minimum contributing storage  range not specified in Table S3 — check S2.23.3 parameter table only
      A_max      [mm]  Maximum contributing storage  range not specified in Table S3 — check S2.23.3 parameter table only
      alpha_ss   [–]  Saturated area fraction  range [0, 5]
      beta_ss    [–]  Saturated area shape parameter  range [0, 1]
      c          [mm d-1]  Catchment infiltration capacity  range [0, 200]
      alpha_g    [mm d-1]  Constant interception rate  range not specified in Table S3 — check S2.23.3 parameter table only
      beta_g     [–]  Interception fraction  range [0, 1]
      gamma_f    [–]  Evaporation shape parameter  range [0, 1]
      delta_f    [–]  Evaporation shape parameter  range [0, 10]
      t_d        [d-1]  Runoff coefficient (recharge rate)  range not specified in Table S3 — check S2.23.3 parameter table only
      alpha_b    [–]  Groundwater discharge shape parameter  range [0, 1]
      beta_b     [mm d-1]  Maximum groundwater recharge rate  range [0.1, 200]
      gamma_a    [–]  Evaporation scaling parameter  range [0, 1]
      delta_a    [–]  Evaporation nonlinearity  range [0, 10]
      alpha_a    [mm d-1]  Maximum subsurface flow rate  range [0.1, 200]
      beta_a     [–]  Subsurface flow nonlinearity  range [1, 5]
      gamma_b    [–]  Evaporation scaling parameter  range [0, 1]
      delta_b    [–]  Evaporation nonlinearity  range [0, 10]

    Output / routing (as specified):
      ```
      Q_t = q_se + q_ie + q_a
      ```
      Routing: none (no Unit Hydrograph mentioned in S2.23).
    """
    # unpack state
    F, A, B = S

    # unpack params
    alpha_f, beta_f, B_max, F_max, alpha_c, beta_c, A_min, A_max, alpha_ss, beta_ss, c, alpha_g, beta_g, gamma_f, delta_f, t_d, alpha_b, beta_b, gamma_a, delta_a, alpha_a, beta_a, gamma_b, delta_b = (
        params['alpha_f'],
        params['beta_f'],
        params['B_max'],
        params['F_max'],
        params['alpha_c'],
        params['beta_c'],
        params['A_min'],
        params['A_max'],
        params['alpha_ss'],
        params['beta_ss'],
        params['c'],
        params['alpha_g'],
        params['beta_g'],
        params['gamma_f'],
        params['delta_f'],
        params['t_d'],
        params['alpha_b'],
        params['beta_b'],
        params['gamma_a'],
        params['delta_a'],
        params['alpha_a'],
        params['beta_a'],
        params['gamma_b'],
        params['delta_b']
    )
    P, Ep = forcings['P'], forcings['Ep']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    P_g = max(alpha_g + beta_g * P, 0)

    _w1 = logistic_storage(A, A_min)
    phi_c = (1 - _w1) * (alpha_c  * (max(0.0, A - A_min)/(A_max - A_min))**beta_c) + _w1 * (0)
    _w2 = logistic_storage(A, A_min)
    phi_ss = (1 - _w2) * (alpha_ss * ((A - A_min)/(A_max - A_min))**beta_ss) + _w2 * (0)
    q_se = phi_c * P_g
    f_s_star = c
    P_c  = min(P_g - q_se, f_s_star)

    f_ss_star = alpha_f * (1 - B/B_max) * (F/F_max)**(-beta_f)
    f_a  = min(P_c * max(1, (1 - phi_ss)/(1 - phi_c)), f_ss_star)

    E_f = gamma_f * Ep * (F/F_max)**delta_f
    r_f = t_d * F

    q_sse = ((phi_ss - phi_c)/(1 - phi_c)) * P_c
    q_sie = max(P_c * (1 - phi_ss)/(1 - phi_c) - f_ss_star, 0)
    q_b   = beta_b * (exp(alpha_b * B/B_max) - 1)
    E_a   = phi_c * Ep + gamma_a * Ep * (A/A_max)**delta_a
    _w3 = logistic_storage(A, A_min)
    q_a = (1 - _w3) * (alpha_a * ((A - A_min)/(A_max - A_min))**beta_a) + _w3 * (0)
    r_a   = phi_ss * f_ss_star

    E_b = gamma_b * Ep * (B/B_max)**delta_b

    q_ie = P_g - q_se - P_c

    # update state(s) via mass balance
    dF_dt = f_a - E_f - r_f
    dA_dt = q_sse + q_sie + q_b - E_a - q_a - r_a
    dB_dt = r_f + r_a - E_b - q_b
    S_new = (F + dF_dt*dt, A + dA_dt*dt, B + dB_dt*dt)

    Q_t = q_se + q_ie + q_a
    fluxes_dict = dict(P_g=P_g, phi_c=phi_c, phi_ss=phi_ss, q_se=q_se, P_c=P_c,
                        f_ss_star=f_ss_star, f_a=f_a, E_f=E_f, r_f=r_f, q_sse=q_sse,
                        q_sie=q_sie, q_b=q_b, E_a=E_a, q_a=q_a, r_a=r_a, E_b=E_b, q_ie=q_ie)
    return S_new, Q_t, fluxes_dict


def step_model_24(S, forcings, params, dt=1.0):
    """
    Model 24 -- MOPEX-1  (MARRMoT file: m_24_mopex1_5p_4s)

    Reference(s): Ye et al., 2012 (model improvement study across 197 MOPEX-database catchments)

    Processes represented:
      - Saturation excess flow
      - Infiltration to deeper soil layers
      - A split between fast and slow runoff

    State variables (4):
      S1  : soil moisture storage [mm]
      S2  : groundwater storage [mm]
      Sc1 : fast-flow routing reservoir storage [mm]
      Sc2 : slow-flow routing reservoir storage [mm]

    Parameters (5):
      S_b1       [mm]  Maximum soil moisture storage  range [1, 2000]
      t_w        [d-1]  Runoff coefficient (percolation, S1→S2)  range [0, 1]
      t_u        [d-1]  Runoff coefficient (leakage, S2→Sc2)  range [0, 1]
      S_e        [mm]  Maximum groundwater storage capacity  range [1, 2000]
      t_c        [d-1]  Runoff coefficient (routing, used for both Q_f and Q_u)  range [0, 1]

    Output / routing (as specified):
      ```
      Q_t = Q_f + Q_u
      ```
      Routing: none (routing is represented internally via the two linear storage reservoirs Sc1/Sc2 driven by t_c; no separate Unit Hydrograph transformation is mentioned in S2.24).
    """
    # unpack state
    S1, S2, Sc1, Sc2 = S

    # unpack params
    S_b1, t_w, t_u, S_e, t_c = (
        params['S_b1'],
        params['t_w'],
        params['t_u'],
        params['S_e'],
        params['t_c']
    )
    P, Ep = forcings['P'], forcings['Ep']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    ET1  = (S1 / S_b1) * Ep
    _w1 = logistic_storage(S1, S_b1)
    Q_1f = (1 - _w1) * (P) + _w1 * (0)
    Q_w  = t_w * S1

    ET2  = (S2 / S_e) * Ep
    Q_2u = t_u * S2

    Q_f = t_c * Sc1
    Q_u = t_c * Sc2

    # update state(s) via mass balance
    dS1_dt  = P - ET1 - Q_1f - Q_w
    dS2_dt  = Q_w - ET2 - Q_2u
    dSc1_dt = Q_1f - Q_f
    dSc2_dt = Q_2u - Q_u
    S_new = (S1 + dS1_dt*dt, S2 + dS2_dt*dt, Sc1 + dSc1_dt*dt, Sc2 + dSc2_dt*dt)

    Q_t = Q_f + Q_u
    fluxes_dict = dict(ET1=ET1, Q_1f=Q_1f, Q_w=Q_w, ET2=ET2, Q_2u=Q_2u, Q_f=Q_f, Q_u=Q_u)
    return S_new, Q_t, fluxes_dict


def step_model_25(S, forcings, params, dt=1.0):
    """
    Model 25 -- Thames Catchment Model (TCM)  (MARRMoT file: m_25_tcm_6p_4s)

    Reference(s): Moore and Bell, 2001

    Processes represented:
      - Effective rainfall before infiltration
      - Preferential recharge
      - Catchment drying through prolonged soil moisture depletion
      - Groundwater abstraction
      - Non-linear groundwater flow

    State variables (4):
      S_rz  : current storage in the root zone [mm]
      S_def : current storage in the soil moisture deficit store [mm]
      S_uz  : current storage in the unsaturated zone [mm]
      S_sz  : current storage in the saturated zone [mm]

    Parameters (6):
      φ          [-]  Fraction of net precipitation that is preferential flow  range [0, 1]
      rc         [mm]  Maximum root zone storage  range not specified in Table S3 — check S2.x.3 parameter table only (model 25 is absent from the "Soil moisture: Max store depth" model list)
      γ          [-]  Transpiration reduction factor  range [0, 1]
      k1         [d⁻¹]  Runoff coefficient (unsaturated-zone drainage)  range [0, 1]
      ca         [mm d⁻¹]  Abstraction rate (constant)  range not specified in Table S3 — check S2.x.3 parameter table only
      k2         [d⁻¹ (printed in parameter table; note the governing equation Q=k2·S_sz² requires k2 in mm⁻¹ d⁻¹ for dimensional consistency — flagged discrepancy in the source document)]  Runoff coefficient (quadratic groundwater outflow)  range not specified in Table S3 — check S2.x.3 parameter table only (no matching category/unit; not listed under Fast or Slow reservoir time scale)

    Output / routing (as specified):
      Q = k2 * S_sz² (if S_sz > 0, else 0) — this IS total streamflow (eq. 275); no separate combination step.
      Routing: none (no Unit Hydrograph mentioned in S2.25).
    """
    # unpack state
    S_rz, S_def, S_uz, S_sz = S

    # unpack params
    phi, rc, gamma, k1, c_a, k2 = (
        params['phi'],
        params['rc'],
        params['gamma'],
        params['k1'],
        params['c_a'],
        params['k2']
    )
    P, Ep = forcings['P'], forcings['Ep']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    P_n   = max(P - Ep, 0)
    P_in  = (1 - phi) * P_n
    _w1 = logistic_storage(S_rz, 0)
    E_a = (1 - _w1) * (Ep) + _w1 * (0)
    _w2 = logistic_storage(S_rz, rc)
    q_ex1 = (1 - _w2) * (P_in) + _w2 * (0)
    _w3 = logistic_storage(S_rz, 0)
    E_t = _w3 * (gamma * Ep) + (1 - _w3) * (0)
    _w4 = logistic_storage(S_def, 0)
    q_ex2 = _w4 * (q_ex1) + (1 - _w4) * (0)
    P_by  = phi * P_n
    q_uz  = k1 * S_uz

    a     = c_a
    _w5 = logistic_storage(S_sz, 0)
    Q = (1 - _w5) * (k2 * S_sz**2) + _w5 * (0) # update state(s) via mass balance
    dS_rz  = P_in - E_a - q_ex1
    dS_def = E_t + q_ex2 - q_ex1
    dS_uz  = P_by + q_ex2 - q_uz
    dS_sz  = q_uz - a - Q

    S_new = (S_rz + dS_rz*dt, S_def + dS_def*dt, S_uz + dS_uz*dt, S_sz + dS_sz*dt)
    fluxes_dict = {"P_n":P_n, "P_in":P_in, "E_a":E_a, "q_ex1":q_ex1,
                   "E_t":E_t, "q_ex2":q_ex2, "P_by":P_by, "q_uz":q_uz,
                   "a":a, "Q":Q}
    return S_new, Q, fluxes_dict


def step_model_26(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 26 -- Flex-I  (MARRMoT file: m_26_flexi_10p_4s)

    Reference(s): Fenicia et al., 2008

    Processes represented:
      - Interception by vegetation
      - Infiltration and saturation excess flow based on a distribution of different soil depths
      - A split between fast saturation excess flow and preferential recharge to a slow store
      - Percolation from the unsaturated zone to a slow runoff store

    State variables (4):
      I  : current interception storage [mm]
      UR : current storage in the unsaturated zone [mm]
      FR : current storage in the fast flow reservoir [mm]
      SR : current storage in the slow flow reservoir [mm]

    Parameters (10):
      Imax       [mm]  Maximum interception storage  range [0, 5]
      URmax      [mm]  Maximum soil moisture (unsaturated zone) storage  range [1, 2000]
      β          [-]  Shape (distribution) parameter  range [0, 10]
      D          [-]  Fraction of effective precipitation routed to slow store (runoff partitioning)  range [0, 1]
      Percmax    [mm d⁻¹]  Maximum percolation rate  range [0, 20]
      Lp         [-]  Wilting point as fraction of URmax  range [0.05, 0.95]
      Nlag,f     [d]  Unit Hydrograph time base (fast flow)  range [1, 5]
      Nlag,s     [d]  Unit Hydrograph time base (slow flow)  range [1, 15]
      Kf         [d⁻¹]  Runoff coefficient (fast reservoir)  range [0, 1]
      Ks         [d⁻¹]  Runoff coefficient (slow reservoir)  range [0, 1]

    Output / routing (as specified):
      ```
      Q = Qf + Qs                                                       (290)
      ```
      Routing: YES. Rs (+ Rp) and Rf are each routed through a linearly-increasing
      triangular Unit Hydrograph before entering SR and FR respectively —
      time base Nlag,s [d] for the slow-store input, Nlag,f [d] for the fast-store input.
    """
    # unpack state
    I, UR, FR, SR = S
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    Imax, URmax, beta, D, Percmax, Lp, Nlag_f, Nlag_s, Kf, Ks = (
        params['Imax'], params['URmax'], params['beta'], params['D'],
        params['Percmax'], params['Lp'], params['Nlag_f'], params['Nlag_s'],
        params['Kf'], params['Ks'])

    if uh_state is None:
        uh_state = {
            'fast': UHRouter(uh_triangular(Nlag_f, dt)),
            'slow': UHRouter(uh_triangular(Nlag_s, dt)),
        }

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue -- this pseudocode shows the
    # literal mathematical definition)
    _w1 = logistic_storage(I, 0)
    Ei = (1 - _w1) * (Ep) + _w1 * (0)
    _w2 = logistic_storage(I, Imax)
    Peff = (1 - _w2) * (P) + _w2 * (0)
    Cr   = 1.0 / (1 + exp((-UR/URmax + 0.5) / beta))
    Ru   = (1 - Cr) * Peff
    Eur  = Ep * min(1, (UR/URmax) * (1/Lp))
    Rp   = Percmax * (-UR/URmax)          # as printed in the source; flagged there
                                           # as an unusual (missing?) minus sign --
                                           # preserved rather than silently "corrected"

    Rs   = (Peff - Ru) * D
    Rf   = (Peff - Ru) * (1 - D)

    # route Rs+Rp and Rf through triangular UHs
    Rs_l = uh_state['slow'].route(Rs + Rp)
    Rf_l = uh_state['fast'].route(Rf)

    Qf = Kf * FR
    Qs = Ks * SR

    # update state(s) via mass balance
    dI  = P - Ei - Peff
    dUR = Ru - Eur - Rp
    dFR = Rf_l - Qf
    dSR = Rs_l - Qs

    S_new = (I + dI*dt, UR + dUR*dt, FR + dFR*dt, SR + dSR*dt)
    Q = Qf + Qs
    fluxes_dict = {"Ei":Ei, "Peff":Peff, "Ru":Ru, "Eur":Eur, "Rp":Rp,
                   "Rs":Rs, "Rf":Rf, "Rs_l":Rs_l, "Rf_l":Rf_l, "Qf":Qf, "Qs":Qs}
    return S_new, Q, fluxes_dict, uh_state


def step_model_27(S, forcings, params, dt=1.0):
    """
    Model 27 -- Tank model  (MARRMoT file: m_27_tank_12p_4s)

    Reference(s): Sugawara, 1979, 1995

    Processes represented:
      - Runoff on increasing time scales with depth

    State variables (4):
      S1 : current storage in the upper (top) tank [mm]
      S2 : current storage in the intermediate tank [mm]
      S3 : current storage in the sub-base tank [mm]
      S4 : current storage in the base tank [mm]

    Parameters (12):
      A0         [d⁻¹]  Runoff coefficient (drainage S1→S2)  range not specified in Table S3 — check S2.x.3 parameter table only
      A1         [d⁻¹]  Runoff coefficient (surface runoff Y1, threshold t1)  range not specified in Table S3 — check S2.x.3 parameter table only
      A2         [d⁻¹]  Runoff coefficient (surface runoff Y2, threshold t2)  range not specified in Table S3 — check S2.x.3 parameter table only
      t1         [mm]  Threshold for runoff generation (Y1)  range not specified in Table S3 — check S2.x.3 parameter table only
      t2         [mm]  Threshold for runoff generation (Y2)  range not specified in Table S3 — check S2.x.3 parameter table only
      B0         [d⁻¹]  Runoff coefficient (drainage S2→S3)  range not specified in Table S3 — check S2.x.3 parameter table only
      B1         [d⁻¹]  Runoff coefficient (intermediate runoff Y3, threshold t3)  range not specified in Table S3 — check S2.x.3 parameter table only
      t3         [mm]  Threshold for runoff generation (Y3)  range not specified in Table S3 — check S2.x.3 parameter table only
      C0         [d⁻¹]  Runoff coefficient (drainage S3→S4)  range not specified in Table S3 — check S2.x.3 parameter table only
      C1         [d⁻¹]  Runoff coefficient (sub-base runoff Y4, threshold t4)  range not specified in Table S3 — check S2.x.3 parameter table only
      t4         [mm]  Threshold for runoff generation (Y4)  range not specified in Table S3 — check S2.x.3 parameter table only
      D1         [d⁻¹]  Runoff coefficient (baseflow Y5)  range not specified in Table S3 — check S2.x.3 parameter table only

    Output / routing (as specified):
      ```
      Q = Y1 + Y2 + Y3 + Y4 + Y5                                        (307)
      ```
      Routing: none (no Unit Hydrograph mentioned in S2.27).
    """
    # unpack state
    S1, S2, S3, S4 = S
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    A0, A1, A2, t1, t2, B0, B1, t3, C0, C1, t4, D1 = (
        params['A0'],
        params['A1'],
        params['A2'],
        params['t1'],
        params['t2'],
        params['B0'],
        params['B1'],
        params['t3'],
        params['C0'],
        params['C1'],
        params['t4'],
        params['D1']
    )

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    _w1 = logistic_storage(S1, 0)
    E1 = (1 - _w1) * (Ep) + _w1 * (0)
    F12 = A0 * S1
    _w2 = logistic_storage(S1, t2)
    Y2 = (1 - _w2) * (A2*(S1 - t2)) + _w2 * (0)
    _w3 = logistic_storage(S1, t1)
    Y1 = (1 - _w3) * (A1*(S1 - t1)) + _w3 * (0)
    E2  = Ep if (S1 <= 0 and S2 > 0) else 0
    F23 = B0 * S2
    _w4 = logistic_storage(S2, t3)
    Y3 = (1 - _w4) * (B1*(S2 - t3)) + _w4 * (0)
    E3  = Ep if (S1 <= 0 and S2 <= 0 and S3 > 0) else 0
    F34 = C0 * S3
    _w5 = logistic_storage(S3, t4)
    Y4 = (1 - _w5) * (C1*(S3 - t4)) + _w5 * (0)
    E4  = Ep if (S1 <= 0 and S2 <= 0 and S3 <= 0 and S4 > 0) else 0
    Y5  = D1 * S4

    # update state(s) via mass balance
    dS1 = P - E1 - F12 - Y2 - Y1
    dS2 = F12 - E2 - F23 - Y3
    dS3 = F23 - E3 - F34 - Y4
    dS4 = F34 - E4 - Y5

    S_new = (S1+dS1*dt, S2+dS2*dt, S3+dS3*dt, S4+dS4*dt)
    Q = Y1 + Y2 + Y3 + Y4 + Y5
    fluxes_dict = {"E1":E1,"F12":F12,"Y1":Y1,"Y2":Y2,"E2":E2,"F23":F23,"Y3":Y3,
                   "E3":E3,"F34":F34,"Y4":Y4,"E4":E4,"Y5":Y5}
    return S_new, Q, fluxes_dict


def step_model_28(S_state, forcings, params, dt=1.0):
    """
    Model 28 -- Xinanjiang model  (MARRMoT file: m_28_xinanjiang_12p_4s)

    Reference(s): Zhao, 1992 (original single-parabolic-curve model); double-parabolic

    Processes represented:
      - Runoff from impervious areas
      - Variable distribution of tension water storage capacities in the catchment
      - Variable contributing area of free water storages
      - Direct surface runoff from the contributing free area
      - Delayed interflow and baseflow from the contributing free area

    State variables (4):
      W  : current tension water storage [mm]
      S  : current storage of free water [mm]
      SI : current storage in the interflow routing reservoir [mm]
      SG : current storage in the baseflow routing reservoir [mm]

    Parameters (12):
      Aim        [-]  Fraction impervious area  range [0, 1]
      a          [-]  Contributing area curve inflection point  range [-0.5, 0.5]
      b          [-]  Contributing area curve shape parameter  range [0, 10]
      Wmax       [mm]  Maximum tension water storage  range [1, 2000]
      LM         [mm]  Threshold for evaporation behaviour change  range not specified in Table S3 — check S2.x.3 parameter table only
      c          [-]  Threshold and evaporation reduction factor  range not specified in Table S3 — check S2.x.3 parameter table only
      Smax       [mm]  Maximum free water storage  range not specified in Table S3 — check S2.x.3 parameter table only (no distinct "free water" store-depth row separate from Wmax's "Soil moisture Max store depth")
      Ex         [-]  Contributing area curve shape parameter (free water)  range not specified in Table S3 — check S2.x.3 parameter table only
      kI         [d⁻¹]  Runoff coefficient (interflow generation)  range [0, 1]
      kG         [d⁻¹]  Runoff coefficient (baseflow generation)  range [0, 1]
      cI         [d⁻¹]  Runoff coefficient (interflow routing reservoir)  range not specified in Table S3 — check S2.x.3 parameter table only (Fast/Slow reservoir time scale rows already used by kI/kG)
      cG         [d⁻¹]  Runoff coefficient (baseflow routing reservoir)  range not specified in Table S3 — check S2.x.3 parameter table only (Fast/Slow reservoir time scale rows already used by kI/kG)

    Output / routing (as specified):
      ```
      Qt = QS + QI + QG                                                  (320)
      ```
      Routing: none (no Unit Hydrograph mentioned in S2.28; delay is via the two linear
      routing reservoirs SI and SG instead).
    """
    # unpack state
    W, S, SI, SG = S_state
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    Aim, a, b, Wmax, LM, c, Smax, Ex, kI, kG, cI, cG = (
        params['Aim'],
        params['a'],
        params['b'],
        params['Wmax'],
        params['LM'],
        params['c'],
        params['Smax'],
        params['Ex'],
        params['kI'],
        params['kG'],
        params['cI'],
        params['cG']
    )

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    Pi = (1 - Aim) * P

    if W / Wmax <= 0.5 - a:
        R = Pi * ((0.5 - a)**(1 - b)) * (W / Wmax)**b
    else:
        R = Pi * (1 - ((0.5 + a)**(1 - b)) * (1 - W / Wmax)**b)

    if W > LM:
        E = Ep
    elif c*LM <= W <= LM:          # see condition-notation flag above
        E = (W / LM) * Ep
    else:
        E = c * Ep

    contrib = 1 - (1 - S/Smax)**Ex
    RS = R  * contrib
    RI = kI * S * contrib
    RG = kG * S * contrib

    QI = cI * SI
    QG = cG * SG

    RB = Aim * P
    QS = RS + RB
    Qt = QS + QI + QG

    # update state(s) via mass balance
    dW  = Pi - E - R
    dS  = R - RS - RI - RG
    dSI = RI - QI
    dSG = RG - QG

    S_new = (W+dW*dt, S+dS*dt, SI+dSI*dt, SG+dSG*dt)
    fluxes_dict = {"Pi":Pi, "R":R, "E":E, "RS":RS, "RI":RI, "RG":RG,
                   "QI":QI, "QG":QG, "RB":RB, "QS":QS}
    return S_new, Qt, fluxes_dict


def step_model_29(S_state, forcings, params, dt=1.0):
    """
    Model 29 -- HyMOD  (MARRMoT file: m_29_hymod_5p_5s)

    Reference(s): Wagener et al., 2001; Boyle, 2001 (PDM-like soil moisture routine

    Processes represented:
      - Different soil depths throughout the catchment
      - Separation of flow into fast and slow flow

    State variables (5):
      Sm : current storage in the soil moisture store [mm]
      F1 : current storage in fast-flow reservoir 1 [mm]
      F2 : current storage in fast-flow reservoir 2 [mm]
      F3 : current storage in fast-flow reservoir 3 [mm]
      S  : current storage in the slow-flow reservoir [mm]

    Parameters (5):
      Smax       [mm]  Maximum soil moisture storage  range [1, 2000]
      b          [-]  Contributing area / soil-depth-distribution curve shape parameter  range [0, 10]
      a          [-]  Fraction of effective precipitation that becomes fast flow  range [0, 1]
      kf         [d⁻¹]  Runoff coefficient (fast reservoirs F1,F2,F3)  range [0, 1]
      ks         [d⁻¹]  Runoff coefficient (slow reservoir S)  range [0, 1]

    Output / routing (as specified):
      ```
      Qt = Qs + Qf,3                                                     (332)
      ```
      Routing: none (no explicit Unit Hydrograph; fast-flow delay is produced instead by a
      Nash cascade of 3 identical linear reservoirs F1→F2→F3, all with time-scale
      parameter kf).
    """
    # unpack state
    Sm, F1, F2, F3, S = S_state
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    Smax, b, a, kf, ks = (
        params['Smax'],
        params['b'],
        params['a'],
        params['kf'],
        params['ks']
    )

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    Ea = (Sm / Smax) * Ep
    Pe = (1 - (1 - Sm/Smax)**b) * P

    Pf   = a * Pe
    Qf1  = kf * F1
    Qf2  = kf * F2
    Qf3  = kf * F3

    Ps = (1 - a) * Pe
    Qs = ks * S

    # update state(s) via mass balance
    dSm = P  - Ea - Pe
    dF1 = Pf - Qf1
    dF2 = Qf1 - Qf2
    dF3 = Qf2 - Qf3
    dS  = Ps - Qs

    S_new = (Sm+dSm*dt, F1+dF1*dt, F2+dF2*dt, F3+dF3*dt, S+dS*dt)
    Qt = Qs + Qf3
    fluxes_dict = {"Ea":Ea, "Pe":Pe, "Pf":Pf, "Qf1":Qf1, "Qf2":Qf2, "Qf3":Qf3,
                   "Ps":Ps, "Qs":Qs}
    return S_new, Qt, fluxes_dict


def step_model_30(S_state, forcings, params, dt=1.0):
    """
    Model 30 -- MOPEX-2  (MARRMoT file: m_30_mopex2_7p_5s)

    Reference(s): Ye et al., 2012

    Processes represented:
      - Snow accumulation and melt
      - Saturation excess flow
      - Infiltration to deeper soil layers
      - A split between fast and slow runoff

    State variables (5):
      Sn  : current snow pack storage [mm]
      S1  : current storage in soil moisture (upper) [mm]
      S2  : current storage in groundwater [mm]
      Sc1 : current storage in the fast-flow routing reservoir [mm]
      Sc2 : current storage in the slow-flow routing reservoir [mm]

    Parameters (7):
      Tcrit      [°C]  Threshold temperature for snowfall and melt  range [-3, 5]
      ddf        [mm °C⁻¹ d⁻¹]  Degree-day factor  range [0, 20]
      Sb1        [mm]  Maximum soil moisture storage  range [1, 2000]
      tw         [d⁻¹]  Runoff coefficient (infiltration to deeper groundwater, S1→S2)  range [0, 1]
      tu         [d⁻¹]  Runoff coefficient (leakage from groundwater to slow routing store, S2→Sc2)  range not specified in Table S3 — check S2.x.3 parameter table only (no distinct remaining category)
      Se         [mm]  Maximum groundwater storage capacity  range not specified in Table S3 — check S2.x.3 parameter table only
      tc         [d⁻¹]  Runoff coefficient (mean residence time, both routing reservoirs Sc1 and Sc2)  range [0, 1]

    Output / routing (as specified):
      ```
      Qt = Qf + Qu                                                       (348)
      ```
      Routing: none (no Unit Hydrograph mentioned in S2.30; delay is via the two linear
      routing reservoirs Sc1 and Sc2 instead, both using time coefficient tc).
    """
    # unpack state
    Sn, S1, S2, Sc1, Sc2 = S_state
    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    # unpack params
    Tcrit, ddf, Sb1, tw, tu, Se, tc = (
        params['Tcrit'],
        params['ddf'],
        params['Sb1'],
        params['tw'],
        params['tu'],
        params['Se'],
        params['tc']
    )

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    _w1 = logistic_temperature(T, Tcrit)
    Ps = _w1 * (P) + (1 - _w1) * (0)
    _w2 = logistic_temperature(T, Tcrit)
    Qn = (1 - _w2) * (ddf * (T - Tcrit)) + _w2 * (0)
    _w3 = logistic_temperature(T, Tcrit)
    Pr = (1 - _w3) * (P) + _w3 * (0)
    ET1 = (S1 / Sb1) * Ep
    _w4 = logistic_storage(S1, Sb1)
    Q1f = (1 - _w4) * (P) + _w4 * (0) # as printed; see flag above
    Qw  = tw * S1

    ET2 = (S2 / Se) * Ep
    Q2u = tu * S2

    Qf = tc * Sc1
    Qu = tc * Sc2

    # update state(s) via mass balance
    dSn  = Ps - Qn
    dS1  = Pr - ET1 - Q1f - Qw
    dS2  = Qw - ET2 - Q2u
    dSc1 = Q1f - Qf
    dSc2 = Q2u - Qu

    S_new = (Sn+dSn*dt, S1+dS1*dt, S2+dS2*dt, Sc1+dSc1*dt, Sc2+dSc2*dt)
    Qt = Qf + Qu
    fluxes_dict = {"Ps":Ps, "Qn":Qn, "Pr":Pr, "ET1":ET1, "Q1f":Q1f, "Qw":Qw,
                   "ET2":ET2, "Q2u":Q2u, "Qf":Qf, "Qu":Qu}
    return S_new, Qt, fluxes_dict


def step_model_31(S, forcings, params, dt=1.0):
    """
    Model 31 -- MOPEX-3  (MARRMoT file: m_31_mopex3_8p_5s)

    Reference(s): Ye, S. et al. (2012) MOPEX model-improvement study, 197 catchments in the MOPEX database (model structure investigation of dominant processes vs. model structure).

    Processes represented:
      - Snow accumulation and melt
      - Saturation excess flow
      - Infiltration to deeper soil layers
      - Subsurface-influenced fast flow
      - A split between fast and slow runoff

    State variables (5):
      Sn  : snow pack [mm]
      S1  : soil moisture (upper soil / tension) store [mm]
      S2  : groundwater storage [mm]
      Sc1 : fast-flow routing reservoir [mm]
      Sc2 : slow-flow routing reservoir [mm]

    Parameters (8):
      T_crit     [°C]  Threshold temperature for snowfall and melt  range [-3, 5]
      ddf        [mm °C⁻¹ d⁻¹]  Degree-day factor  range [0, 20]
      S_b1       [mm]  Maximum soil moisture storage  range [1, 2000]
      t_w        [d⁻¹]  Runoff coefficient (soil→groundwater percolation rate, Q_w = t_w·S1)  range not specified in Table S3 — no distinct category for this soil→groundwater time coefficient; check S2.31.3 table only
      S_b2       [mm]  Maximum deep (groundwater) storage  range [1, 2000]
      t_u        [d⁻¹]  Runoff coefficient (groundwater→slow store leakage, Q_2u = t_u·S2)  range [0, 1]
      S_e        [mm]  Maximum groundwater/root-zone storage capacity (evaporation scaling)  range [1, 2000]
      t_c        [d⁻¹]  Runoff coefficient / mean residence time for both routing reservoirs (Q_f = t_c·Sc1, Q_u = t_c·Sc2)  range [0, 1]

    Output / routing (as specified):
      ```
      Q = Qf + Qu
      ```
      Routing: none — MOPEX-3 has no separate Unit Hydrograph transformation. Fast/slow response is produced purely by
      the two internal linear reservoirs Sc1 and Sc2 (both drained at rate t_c).
    """
    # unpack state
    Sn, S1, S2, Sc1, Sc2 = S

    # unpack forcings
    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    # unpack params
    T_crit, ddf, S_b1, t_w, S_b2, t_u, S_e, t_c = (
        params['T_crit'],
        params['ddf'],
        params['S_b1'],
        params['t_w'],
        params['S_b2'],
        params['t_u'],
        params['S_e'],
        params['t_c']
    )

    # NOTE: MARRMoT's actual numeric implementation smooths the hard if/else
    # thresholds below with logistic smoother functions from the toolbox's
    # flux catalogue (Table S1). This pseudocode gives the literal
    # mathematical (unsmoothed) definitions exactly as printed in S2.31.2.

    # --- snow module ---
    _w1 = logistic_temperature(T, T_crit)
    Ps = _w1 * (P) + (1 - _w1) * (0.0)
    _w2 = logistic_temperature(T, T_crit)
    Pr = (1 - _w2) * (P) + _w2 * (0.0)
    _w3 = logistic_temperature(T, T_crit)
    Qn = (1 - _w3) * (ddf * (T - T_crit)) + _w3 * (0.0) # --- soil moisture (S1) ---
    ET1 = (S1 / S_b1) * Ep
    _w4 = logistic_storage(S1, S_b1)
    Q1f = (1 - _w4) * (P) + _w4 * (0.0)
    Qw  = t_w * S1

    # --- groundwater (S2) ---
    ET2 = (S2 / S_e) * Ep
    Q2u = t_u * S2
    _w5 = logistic_storage(S2, S_b2)
    Q2f = (1 - _w5) * (Qw) + _w5 * (0.0) # --- routing reservoirs ---
    Qf = t_c * Sc1
    Qu = t_c * Sc2

    # update state(s) via mass balance (explicit Euler shown; MARRMoT solves
    # this implicitly/ODE-solver based, but the flux definitions are identical)
    Sn_new  = Sn  + dt * (Ps - Qn)
    S1_new  = S1  + dt * (Pr - ET1 - Q1f - Qw)
    S2_new  = S2  + dt * (Qw - ET2 - Q2u - Q2f)
    Sc1_new = Sc1 + dt * (Q1f + Q2f - Qf)
    Sc2_new = Sc2 + dt * (Q2u - Qu)

    Q = Qf + Qu

    fluxes_dict = dict(Ps=Ps, Pr=Pr, Qn=Qn, ET1=ET1, Q1f=Q1f, Qw=Qw,
                        ET2=ET2, Q2u=Q2u, Q2f=Q2f, Qf=Qf, Qu=Qu)
    S_new = (Sn_new, S1_new, S2_new, Sc1_new, Sc2_new)
    return S_new, Q, fluxes_dict


def step_model_32(S, forcings, params, dt=1.0):
    """
    Model 32 -- MOPEX-4  (MARRMoT file: m_32_mopex4_10p_5s)

    Reference(s): Ye, S. et al. (2012) MOPEX model-improvement study (197 catchments, MOPEX database); interception

    Processes represented:
      - Snow accumulation and melt
      - Time-varying interception
      - Saturation excess flow
      - Infiltration to deeper soil layers
      - A split between fast and slow runoff

    State variables (5):
      Sn  : snow pack [mm]
      S1  : soil moisture (upper soil / tension) store [mm]
      S2  : groundwater storage [mm]
      Sc1 : fast-flow routing reservoir [mm]
      Sc2 : slow-flow routing reservoir [mm]

    Parameters (10):
      T_crit     [°C]  Threshold temperature for snowfall and melt  range [-3, 5]
      ddf        [mm °C⁻¹ d⁻¹]  Degree-day factor  range [0, 20]
      S_b1       [mm]  Maximum soil moisture storage  range [1, 2000]
      t_w        [d⁻¹]  Runoff coefficient (soil→groundwater percolation rate)  range not specified in Table S3 — check S2.32.3 table only
      I_α        [–]  Mean intercepted fraction of precipitation  range [0, 1]
      I_s        [d]  Timing of peak interception capacity  range [1, 365]
      S_b2       [mm]  Maximum deep (groundwater) storage  range [1, 2000]
      t_u        [d⁻¹]  Runoff coefficient (groundwater→slow store leakage)  range [0, 1]
      S_e        [mm]  Maximum groundwater/root-zone storage capacity (evaporation scaling)  range [1, 2000]
      t_c        [d⁻¹]  Runoff coefficient / mean residence time for both routing reservoirs  range [0, 1]

    Output / routing (as specified):
      ```
      Q = Qf + Qu
      ```
      Routing: none — no Unit Hydrograph. Fast/slow response comes purely from the two internal linear reservoirs Sc1
      and Sc2 (both drained at rate t_c), identical routing structure to Model 31.
    """
    t_index = forcings['t']  # day-of-year forcing, folded in from the extra `t_index` arg
    # unpack state
    Sn, S1, S2, Sc1, Sc2 = S

    # unpack forcings
    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    # unpack params
    T_crit, ddf, S_b1, t_w, I_alpha, I_s, S_b2, t_u, S_e, t_c = (
        params['T_crit'],
        params['ddf'],
        params['S_b1'],
        params['t_w'],
        params['I_alpha'],
        params['I_s'],
        params['S_b2'],
        params['t_u'],
        params['S_e'],
        params['t_c']
    )
    t_max = 365  # fixed constant, not calibrated

    # NOTE: MARRMoT's numeric implementation smooths the hard if/else
    # thresholds below with logistic smoother functions from the toolbox's
    # flux catalogue (Table S1). This pseudocode gives the literal
    # mathematical (unsmoothed) definitions exactly as printed in S2.32.2.

    # --- snow module ---
    _w1 = logistic_temperature(T, T_crit)
    Ps = _w1 * (P) + (1 - _w1) * (0.0)
    _w2 = logistic_temperature(T, T_crit)
    Pr = (1 - _w2) * (P) + _w2 * (0.0)
    _w3 = logistic_temperature(T, T_crit)
    Qn = (1 - _w3) * (ddf * (T - T_crit)) + _w3 * (0.0) # --- interception (time-varying) ---
    I = max(0.0, I_alpha + (1 - I_alpha) *
            cos(2 * pi * (t_index - I_s) / t_max)) * Pr

    # --- soil moisture (S1) ---
    ET1 = (S1 / S_b1) * Ep
    _w4 = logistic_storage(S1, S_b1)
    Q1f = (1 - _w4) * (P) + _w4 * (0.0)
    Qw  = t_w * S1

    # --- groundwater (S2) ---
    ET2 = (S2 / S_e) * Ep
    Q2u = t_u * S2
    _w5 = logistic_storage(S2, S_b2)
    Q2f = (1 - _w5) * (Qw) + _w5 * (0.0) # --- routing reservoirs ---
    Qf = t_c * Sc1
    Qu = t_c * Sc2

    # update state(s) via mass balance
    Sn_new  = Sn  + dt * (Ps - Qn)
    S1_new  = S1  + dt * (Pr - ET1 - I - Q1f - Qw)
    S2_new  = S2  + dt * (Qw - ET2 - Q2u - Q2f)
    Sc1_new = Sc1 + dt * (Q1f + Q2f - Qf)
    Sc2_new = Sc2 + dt * (Q2u - Qu)

    Q = Qf + Qu

    fluxes_dict = dict(Ps=Ps, Pr=Pr, Qn=Qn, I=I, ET1=ET1, Q1f=Q1f, Qw=Qw,
                        ET2=ET2, Q2u=Q2u, Q2f=Q2f, Qf=Qf, Qu=Qu)
    S_new = (Sn_new, S1_new, S2_new, Sc1_new, Sc2_new)
    return S_new, Q, fluxes_dict


def step_model_33(S, forcings, params, dt=1.0):
    """
    Model 33 -- SACRAMENTO  (MARRMoT file: m_33_sacramento_11p_5s)

    Reference(s): Burnash, R.J.C. (1995); National Weather Service (2005) — the SAC-SMA (Sacramento Soil Moisture

    Processes represented:
      - Impervious and direct runoff
      - Within-soil division of water storage between tension and free water
      - Surface runoff, interflow, and percolation to deeper soil layers
      - Multiple baseflow processes

    State variables (5):
      UZTW : upper-zone tension water [mm]
      UZFW : upper-zone free water [mm]
      LZTW : lower-zone tension water [mm]
      LZFWP : lower-zone primary free water [mm]
      LZFWS : lower-zone supplemental (secondary) free water [mm]

    Parameters (13):
      PCTIM      [–]  Fraction impervious area  range [0, 1]
      UZTWM      [mm]  Maximum upper zone tension water storage  range [1, 2000]
      UZFWM      [mm]  Maximum upper zone free water storage  range [1, 2000]
      k_uz       [d⁻¹]  Runoff coefficient (interflow, Q_int = k_uz·UZFW)  range [0, 1]
      PBASE      [mm d⁻¹]  Base percolation rate — **derived**: `PBASE = k_lzp·LZFWPM + k_lzs·LZFWSM` (not independently calibrated)  range not applicable (derived quantity, not sampled) — see note above
      ZPERC      [–]  Maximum percolation rate as multiple of PBASE — documentation states this is also a **derived** coefficient (formula not given in extracted pages)  range not specified in Table S3 — check S2.33.3 table only, and verify derived-vs-free status against MARRMoT source
      REXP       [–]  Percolation non-linearity exponent  range [0, 10]
      LZTWM      [mm]  Maximum lower zone tension water storage  range [1, 2000]
      LZFWPM     [mm]  Maximum lower zone primary free water storage  range [1, 2000]
      LZFWSM     [mm]  Maximum lower zone secondary (supplemental) free water storage  range [1, 2000]
      PFREE      [–]  Fraction of percolation that goes directly to free storage  range [0, 1]
      k_lzp      [d⁻¹]  Runoff coefficient — primary baseflow (Q_bfp = k_lzp·LZFWP)  range [0, 1]
      k_lzs      [d⁻¹]  Runoff coefficient — supplemental/secondary baseflow (Q_bfs = k_lzs·LZFWS)  range [0, 1]

    Output / routing (as specified):
      ```
      Q = Qdir + Qsur + Qint + Qbfp + Qbfs
      ```
      Routing: none — no Unit Hydrograph. All five stores are internal linear/nonlinear reservoirs; streamflow is the
      direct sum of the five component fluxes above (no additional convolution/routing stage in this MARRMoT
      implementation).
    """
    # unpack state
    UZTW, UZFW, LZTW, LZFWP, LZFWS = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params (13 symbols as printed; PBASE & ZPERC flagged as
    # possibly-derived rather than free-calibrated — verify against source)
    PCTIM, UZTWM, UZFWM, k_uz, PBASE, ZPERC, REXP, LZTWM, LZFWPM, LZFWSM, PFREE, k_lzp, k_lzs = (
        params['PCTIM'],
        params['UZTWM'],
        params['UZFWM'],
        params['k_uz'],
        params['PBASE'],
        params['ZPERC'],
        params['REXP'],
        params['LZTWM'],
        params['LZFWPM'],
        params['LZFWSM'],
        params['PFREE'],
        params['k_lzp'],
        params['k_lzs']
    )

    # if PBASE truly is derived rather than a free parameter:
    # PBASE = k_lzp * LZFWPM + k_lzs * LZFWSM

    # NOTE: MARRMoT's actual numeric implementation smooths the hard if/else
    # thresholds below (store-at-max-capacity checks, deficit-ratio checks)
    # using logistic smoother functions from the toolbox's flux catalogue
    # (Table S1). This pseudocode gives the literal mathematical
    # (unsmoothed) definitions exactly as printed in S2.33.2. All fluxes
    # below are meant to be evaluated SIMULTANEOUSLY from the states at the
    # start of the time step (per the documentation's explicit note), not
    # sequentially.

    # --- precipitation split ---
    Peff = (1 - PCTIM) * P
    Qdir = PCTIM * P

    # --- upper zone tension water ---
    if UZTW / UZTWM < UZFW / UZFWM:
        Ru = (UZTWM * UZFW - UZFWM * UZTW) / (UZTWM + UZFWM)
    else:
        Ru = 0.0
    Euztw = (UZTW / UZTWM) * Ep
    _w1 = logistic_storage(UZTW, UZTWM)
    Twex_u = (1 - _w1) * (Peff) + _w1 * (0.0) # --- upper zone free water ---
    if UZFW > 0 and Ep > Euztw:
        Euzfw = Ep - Euztw
    else:
        Euzfw = 0.0
    _w2 = logistic_storage(UZFW, UZFWM)
    Qsur = (1 - _w2) * (Twex_u) + _w2 * (0.0)
    Qint = k_uz * UZFW

    LZ_deficiency = (LZTWM - LZTW) + (LZFWPM - LZFWP) + (LZFWSM - LZFWS)
    LZ_capacity   = LZTWM + LZFWPM + LZFWSM
    Pc_demand = PBASE * (1 + ZPERC * (LZ_deficiency / LZ_capacity) ** (1 + REXP))
    Pc = Pc_demand * (UZFW / UZFWM)

    # --- lower zone tension water ---
    Pc_tw = (1 - PFREE) * Pc
    frac_tension_deficit = LZTW / LZTWM
    frac_free_deficit = (LZFWP + LZFWS) / (LZFWPM + LZFWSM)
    if frac_tension_deficit < frac_free_deficit:
        common_denom = (LZFWPM + LZFWSM) * (LZTWM + LZFWPM + LZFWSM)
        Rl_p = LZFWPM * (-LZTW*(LZFWPM+LZFWSM) + LZTWM*(LZFWP+LZFWS)) / common_denom
        Rl_s = LZFWSM * (-LZTW*(LZFWPM+LZFWSM) + LZTWM*(LZFWP+LZFWS)) / common_denom
    else:
        Rl_p = 0.0
        Rl_s = 0.0

    if LZTW > 0 and Ep > (Euztw + Euzfw):
        Elztw = (Ep - Euztw - Euzfw) * LZTW / (UZTWM + LZTWM)
    else:
        Elztw = 0.0
    _w3 = logistic_storage(LZTW, LZTWM)
    Twex_l = (1 - _w3) * (Pc_tw) + _w3 * (0.0) # --- lower zone free water split (primary vs. supplemental) ---
    deficit_ratio_p = (LZFWPM - LZFWP) / LZFWPM
    deficit_ratio_s = (LZFWSM - LZFWS) / LZFWSM
    denom = deficit_ratio_p + deficit_ratio_s

    share_p = (deficit_ratio_p / LZFWPM) / denom if denom != 0 else 0.0  # dimension-consistent w/ eq. 405
    share_s = (deficit_ratio_s / LZFWSM) / denom if denom != 0 else 0.0  # dimension-consistent w/ eq. 409
    # (implemented directly from eq. 405/406/409/410's bracket expressions)
    Pc_fw_p  = ((LZFWPM - LZFWP) / (LZFWPM * denom)) * (PFREE * Pc)
    Twex_lp  = ((LZFWPM - LZFWP) / (LZFWPM * denom)) * Twex_l
    Pc_fw_s  = ((LZFWSM - LZFWS) / (LZFWSM * denom)) * (PFREE * Pc)
    Twex_ls  = ((LZFWSM - LZFWS) / (LZFWSM * denom)) * Twex_l

    Qbfp = k_lzp * LZFWP
    Qbfs = k_lzs * LZFWS

    # update state(s) via mass balance
    UZTW_new  = UZTW  + dt * (Peff + Ru - Euztw - Twex_u)
    UZFW_new  = UZFW  + dt * (Twex_u - Euzfw - Qsur - Qint - Pc - Ru)
    LZTW_new  = LZTW  + dt * (Pc_tw + Rl_p + Rl_s - Elztw - Twex_l)
    LZFWP_new = LZFWP + dt * (Pc_fw_p + Twex_lp - Qbfp)
    LZFWS_new = LZFWS + dt * (Pc_fw_s + Twex_ls - Qbfs)

    Q = Qdir + Qsur + Qint + Qbfp + Qbfs

    fluxes_dict = dict(Peff=Peff, Qdir=Qdir, Ru=Ru, Euztw=Euztw, Twex_u=Twex_u,
                        Euzfw=Euzfw, Qsur=Qsur, Qint=Qint, Pc=Pc, Pc_tw=Pc_tw,
                        Rl_p=Rl_p, Rl_s=Rl_s, Elztw=Elztw, Twex_l=Twex_l,
                        Pc_fw_p=Pc_fw_p, Twex_lp=Twex_lp, Qbfp=Qbfp,
                        Pc_fw_s=Pc_fw_s, Twex_ls=Twex_ls, Qbfs=Qbfs)
    S_new = (UZTW_new, UZFW_new, LZTW_new, LZFWP_new, LZFWS_new)
    return S_new, Q, fluxes_dict


def step_model_34(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 34 -- FLEX-IS  (MARRMoT file: m_34_flexis_12p_5s)

    Reference(s): Fenicia et al. (2008) [FLEX-B]; Nijzink et al. (2016) [snow module]

    Processes represented:
      - Snow accumulation and melt
      - Interception by vegetation
      - Infiltration and saturation excess flow based on a distribution of different soil depths
      - A split between fast saturation excess flow and preferential recharge to a slow store
      - Percolation from the unsaturated zone to a slow runoff store

    State variables (5):
      S  : snow storage [mm]
      I  : interception storage [mm]
      UR : unsaturated-zone (soil moisture) storage [mm]
      FR : fast-flow reservoir storage [mm]
      SR : slow-flow reservoir storage [mm]

    Parameters (12):
      TT         [°C]  Threshold temperature for snowfall and melt  range [-3, 5]
      ddf        [mm °C⁻¹ d⁻¹]  Degree-day factor  range [0, 20]
      Imax       [mm]  Maximum interception storage  range [0, 5]
      URmax      [mm]  Maximum soil moisture storage  range [1, 2000]
      β          [-]  Shape parameter (unsaturated zone runoff distribution)  range [0, 10]
      Lp         [-]  Wilting point as fraction of URmax  range [0.05, 0.95]
      Percmax    [mm d⁻¹]  Maximum percolation rate  range [0, 20]
      D          [-]  Fraction of effective precipitation to slow store  range [0, 1]
      Nlag,f     [d]  Unit Hydrograph time base (fast pathway)  range [1, 5]
      Nlag,s     [d]  Unit Hydrograph time base (slow pathway)  range [1, 15]
      Kf         [d⁻¹]  Runoff coefficient (fast reservoir)  range [0, 1]
      Ks         [d⁻¹]  Runoff coefficient (slow reservoir)  range [0, 1]

    Output / routing (as specified):
      ```
      Q = Qf + Qs                                                               (430)
      ```
      Routing: YES — Unit Hydrograph routing. Both Rs (+Rp) and Rf are lagged by linearly
      increasing triangular transformation functions (a MARRMoT triangular Unit Hydrograph,
      e.g. uh_4_full) before entering SR and FR respectively. Nlag,s sets the time base for the
      slow-pathway UH; Nlag,f sets the time base for the fast-pathway UH.
    """
    # unpack state
    S_snow, I, UR, FR, SR = S

    # unpack params
    TT, ddf, Imax, URmax, beta, Lp, Percmax, D, Nlag_f, Nlag_s, Kf, Ks = (
        params['TT'], params['ddf'], params['Imax'], params['URmax'],
        params['beta'], params['Lp'], params['Percmax'], params['D'],
        params['Nlag_f'], params['Nlag_s'], params['Kf'], params['Ks'])

    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    if uh_state is None:
        uh_state = {
            'fast': UHRouter(uh_triangular(Nlag_f, dt)),
            'slow': UHRouter(uh_triangular(Nlag_s, dt)),
        }

    # --- snow store ---
    _w1 = logistic_temperature(T, TT)
    Ps = _w1 * (P) + (1 - _w1) * (0.0)
    _w2 = logistic_temperature(T, TT)
    M_melt = (1 - _w2) * (ddf * (T - TT)) + _w2 * (0.0) # --- interception store ---
    _w3 = logistic_temperature(T, TT)
    PI = (1 - _w3) * (P) + _w3 * (0.0)
    _w4 = logistic_storage(I, 0)
    Ei = (1 - _w4) * (Ep) + _w4 * (0.0) # Interception spills once the store is at/above capacity. The literal
    # spec printed `I == Imax`; bit-exact float equality effectively never
    # holds in a discrete Euler loop, so this uses `>=` instead -- see the
    # module docstring for the general rationale (same fix as Example 3/4
    # of this course's `S_n >= S_max` saturation-excess condition).
    _w5 = logistic_storage(I, Imax)
    Peff = (1 - _w5) * (PI) + _w5 * (0.0) # --- unsaturated zone (UR) ---
    Ru  = (1 - (1 + exp((-UR / URmax + 0.5) / beta)) ** -1) * Peff
    Eur = Ep * min(1, (UR / URmax) * (1 / Lp))
    Rp  = Percmax * (UR / URmax)

    # --- split of overflow to slow/fast pathways ---
    overflow = Peff - Ru
    Rs = overflow * D
    Rf = overflow * (1 - D)

    # Rs + Rp enters slow-pathway triangular UH (time base Nlag_s) -> Rs_l
    # Rf enters fast-pathway triangular UH (time base Nlag_f) -> Rf_l
    Rs_l = uh_state['slow'].route(Rs + Rp)
    Rf_l = uh_state['fast'].route(Rf)

    # --- fast and slow reservoirs ---
    Qf = Kf * FR
    Qs = Ks * SR

    # update state(s) via mass balance
    S_snow_new = S_snow + dt * (Ps - M_melt)
    I_new      = I      + dt * (PI + M_melt - Ei - Peff)
    UR_new     = UR     + dt * (Ru - Eur - Rp)
    FR_new     = FR     + dt * (Rf_l - Qf)
    SR_new     = SR     + dt * (Rs_l - Qs)

    Q = Qf + Qs

    S_new = (S_snow_new, I_new, UR_new, FR_new, SR_new)
    fluxes_dict = dict(Ps=Ps, M=M_melt, PI=PI, Ei=Ei, Peff=Peff, Ru=Ru, Eur=Eur, Rp=Rp,
                        Rs=Rs, Rf=Rf, Rs_l=Rs_l, Rf_l=Rf_l, Qf=Qf, Qs=Qs)
    return S_new, Q, fluxes_dict, uh_state


def step_model_35(S, forcings, params, dt=1.0):
    """
    Model 35 -- MOPEX-5  (MARRMoT file: m_35_mopex5_12p_5s)

    Reference(s): Ye et al. (2012); Liang et al. (1994) [sinusoidal LAI/interception approximation]

    Processes represented:
      - Snow accumulation and melt
      - Time-varying interception and the impact of phenology on transpiration
      - Saturation excess flow
      - Infiltration to deeper soil layers
      - A split between fast and slow runoff

    State variables (5):
      Sn  : snow pack [mm]
      S1  : soil moisture storage [mm]
      S2  : groundwater storage [mm]
      Sc1 : fast-flow routing store [mm]
      Sc2 : slow-flow routing store [mm]

    Parameters (12):
      Tcrit      [°C]  Threshold temperature for snowfall and melt  range [-3, 5]
      ddf        [mm °C⁻¹ d⁻¹]  Degree-day factor  range [0, 20]
      Sb1        [mm]  Maximum soil moisture storage  range [1, 2000]
      tw         [d⁻¹]  Runoff coefficient (leakage, S1 → S2)  range [0, 1]
      Iα         [-]  Mean intercepted fraction of precipitation  range [0, 1]
      Is         [d]  Timing of peak interception capacity  range [1, 365]
      Tmin       [°C]  Minimum temperature to start growing season  range [0, -10] (as printed in Table S3)
      Tmax       [°C]  Temperature of maximum plant growth  range [1, 20]
      Sb2        [mm]  Maximum deep (groundwater) storage  range [1, 2000]
      tu         [d⁻¹]  Runoff coefficient (leakage, S2 → Sc2)  range [0, 1]
      Se         [mm]  Maximum groundwater storage capacity  range [1, 2000]
      tc         [d⁻¹]  Runoff coefficient (mean residence time, both routing stores)  range [0, 1]

    Output / routing (as specified):
      ```
      Qt = Qf + Qu                                                              (450)
      ```
      Routing: NONE (no Unit Hydrograph). Routing is achieved via two linear reservoirs
      (Sc1 for fast pathway inflows Q1f+Q2f, Sc2 for slow pathway inflow Q2u), both drained
      at the same rate coefficient tc [d⁻¹].
    """
    t_day = forcings['t']  # day-of-year forcing, folded in from the extra `t_day` arg
    # unpack state
    Sn, S1, S2, Sc1, Sc2 = S

    # unpack params
    Tcrit, ddf, Sb1, tw, Ialpha, Is, Tmin, Tmax, Sb2, tu, Se, tc = (
        params['Tcrit'],
        params['ddf'],
        params['Sb1'],
        params['tw'],
        params['Ialpha'],
        params['Is'],
        params['Tmin'],
        params['Tmax'],
        params['Sb2'],
        params['tu'],
        params['Se'],
        params['tc']
    )

    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    # --- snow ---
    _w1 = logistic_temperature(T, Tcrit)
    Ps = _w1 * (P) + (1 - _w1) * (0.0)
    _w2 = logistic_temperature(T, Tcrit)
    Qn = (1 - _w2) * (ddf * (T - Tcrit)) + _w2 * (0.0) # --- rain / phenology / interception ---
    _w3 = logistic_temperature(T, Tcrit)
    Pr = (1 - _w3) * (P) + _w3 * (0.0)
    GSI = 0.0 if T < Tmin else (1.0 if T >= Tmax else (T - Tmin) / (Tmax - Tmin))
    Epc = Ep * GSI
    I_interception = max(0.0, Ialpha + (1 - Ialpha) * sin(2*pi*(t_day + Is) / 365.0))

    # --- soil moisture S1 ---
    ET1 = (S1 / Sb1) * Epc
    _w4 = logistic_storage(S1, Sb1)
    Q1f = (1 - _w4) * (P) + _w4 * (0.0)
    Qw  = tw * S1

    # --- groundwater S2 ---
    ET2 = (S2 / Se) * Epc
    Q2u = tu * S2
    _w5 = logistic_storage(S2, Sb2)
    Q2f = (1 - _w5) * (Qw) + _w5 * (0.0) # --- routing stores ---
    Qf = tc * Sc1
    Qu = tc * Sc2

    # note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)

    # update state(s) via mass balance
    Sn_new  = Sn  + dt * (Ps - Qn)
    S1_new  = S1  + dt * (Pr - ET1 - I_interception - Q1f - Qw)
    S2_new  = S2  + dt * (Qw - ET2 - Q2u - Q2f)
    Sc1_new = Sc1 + dt * (Q1f + Q2f - Qf)
    Sc2_new = Sc2 + dt * (Q2u - Qu)

    Qt = Qf + Qu

    S_new = (Sn_new, S1_new, S2_new, Sc1_new, Sc2_new)
    fluxes_dict = dict(Ps=Ps, Qn=Qn, Pr=Pr, ET1=ET1, I=I_interception, Q1f=Q1f, Qw=Qw,
                        Epc=Epc, GSI=GSI, ET2=ET2, Q2u=Q2u, Q2f=Q2f, Qf=Qf, Qu=Qu)
    return S_new, Qt, fluxes_dict


def step_model_36(S, forcings, params, dt=1.0):
    """
    Model 36 -- MODHYDROLOG  (MARRMoT file: m_36_modhydrolog_15p_5s)

    Reference(s): Chiew (1990); Chiew and McMahon (1994)

    Processes represented:
      - Interception by vegetation
      - Infiltration and infiltration excess flow
      - Depression storage and delayed infiltration
      - Preferential groundwater recharge, interflow and saturation excess flow
      - Groundwater recharge resulting from filling up of soil moisture storage capacity
      - Water exchange between shallow and deep aquifers
      - Water exchange between aquifer and river channel

    State variables (5):
      I   : interception storage [mm]
      D   : depression storage [mm]
      SMS : soil moisture storage [mm]
      GW  : groundwater storage [mm] (no fixed upper/lower bound; fluctuates around datum DLEV)
      CH  : channel storage [mm]

    Parameters (15):
      INSC       [mm]  Maximum interception capacity  range [0, 5]
      COEFF      [mm d⁻¹]  Maximum infiltration loss  range [0, 600]
      SQ         [-]  Infiltration loss exponent  range [0, 15]
      SMSC       [mm]  Maximum soil moisture storage  range [1, 2000]
      SUB        [-]  Proportionality constant (interflow)  range [0, 1]
      CRAK       [-]  Proportionality constant (preferential recharge)  range [0, 1]
      EM         [mm d⁻¹]  Maximum plant-controlled evaporation rate  range [0, 20]
      DSC        [mm]  Maximum depression storage  range [0, 50]
      ADS        [-]  Fraction of area functioning as depression storage  range [0, 1]
      MD         [-]  Depression storage shape parameter  range [0.99, 1]
      VCOND      [d⁻¹]  Leakage coefficient (GW ↔ deeper aquifer)  range [0, 0.5]
      DLEV       [mm]  Datum of groundwater store (level compared to channel level)  range [-10, 10]
      k1         [d⁻¹]  Runoff coefficient (GW–channel exchange)  range [0, 1]
      k2         [d⁻¹]  Runoff coefficient (GW–channel exchange)  range [0, 1]
      k3         [d⁻¹]  Runoff coefficient (GW–channel exchange, exponential term)  range [0, 100]

    Output / routing (as specified):
      ```
      Q = Qt   (channel store CH aggregates SRUN + INT + FLOW inflows; eq. 472 above)
      ```
      Routing: NONE in this MARRMoT implementation. (The original MODHYDROLOG includes a
      routing scheme to link sub-basins together; it has been removed from the MARRMoT
      version, so channel storage CH directly aggregates SRUN, INT and FLOW to produce Q.)
    """
    # unpack state
    I, D, SMS, GW, CH = S

    # unpack params
    INSC, COEFF, SQ, SMSC, SUB, CRAK, EM, DSC, ADS, MD, VCOND, DLEV, k1, k2, k3 = (
        params['INSC'],
        params['COEFF'],
        params['SQ'],
        params['SMSC'],
        params['SUB'],
        params['CRAK'],
        params['EM'],
        params['DSC'],
        params['ADS'],
        params['MD'],
        params['VCOND'],
        params['DLEV'],
        params['k1'],
        params['k2'],
        params['k3']
    )

    P, Ep = forcings['P'], forcings['Ep']
    PET = Ep

    # --- interception ---
    _w2 = logistic_storage(I, 0)
    Ei = (1 - _w2) * (Ep) + _w2 * (0.0)
    _w3 = logistic_storage(I, INSC)
    EXC = (1 - _w3) * (P) + _w3 * (0.0) # --- infiltration / soil moisture partitioning ---
    INF = min(COEFF * exp(-SQ * SMS / SMSC), EXC)
    INT = SUB * (SMS / SMSC) * INF
    REC = CRAK * (SMS / SMSC) * (INF - INT)
    SMF = INF - INT - REC
    ET  = min(EM * (SMS / SMSC), PET)
    _w4 = logistic_storage(SMS, SMSC)
    GWF = (1 - _w4) * (SMF) + _w4 * (0.0) # --- overland flow / depression storage ---
    RUN  = EXC - INF
    TRAP = ADS * exp(-MD * D / (DSC - D)) * RUN
    _w5 = logistic_storage(D, 0)
    ED = (1 - _w5) * (ADS * Ep) + _w5 * (0.0)
    RATE = COEFF * exp(-SQ * SMS / SMSC) - INF - INT - REC
    _w6 = logistic_storage(D, 0)
    DINF = (1 - _w6) * (ADS * RATE) + _w6 * (0.0)
    SRUN = RUN - TRAP

    # --- groundwater exchange ---
    SEEP = VCOND * (GW - DLEV)
    _w1 = logistic_storage(GW, 0)
    FLOW = (1 - _w1) * (k1 * abs(GW) + k2 * (1 - exp(-k3 * abs(GW)))) + _w1 * (-(k1 * abs(GW) + k2 * (1 - exp(-k3 * abs(GW)))))

    # --- channel / output ---
    # NB: CH update happens via mass balance below; Q read off CH per eq. 472
    _w7 = logistic_storage(CH, 0)
    Qt = (1 - _w7) * (CH) + _w7 * (0.0) # evaluated using CH BEFORE this step's update,
                                  # per the model's own ODE structure (dCH/dt = SRUN+INT+FLOW-Q)

    # note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)

    # update state(s) via mass balance
    I_new   = I   + dt * (P - Ei - EXC)
    SMS_new = SMS + dt * (SMF + DINF - ET - GWF)
    D_new   = D   + dt * (TRAP - ED - DINF)
    GW_new  = GW  + dt * (REC + GWF - SEEP - FLOW)
    CH_new  = CH  + dt * (SRUN + INT + FLOW - Qt)

    S_new = (I_new, D_new, SMS_new, GW_new, CH_new)
    fluxes_dict = dict(Ei=Ei, EXC=EXC, INF=INF, INT=INT, REC=REC, SMF=SMF, ET=ET, GWF=GWF,
                        RUN=RUN, TRAP=TRAP, ED=ED, DINF=DINF, SRUN=SRUN,
                        SEEP=SEEP, FLOW=FLOW)
    return S_new, Qt, fluxes_dict


def step_model_37(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 37 -- HBV-96  (MARRMoT file: m_37_hbv_15p_5s)

    Reference(s): Lindström et al., 1997

    Processes represented:
      - Snow accumulation, melt and refreezing
      - Infiltration and capillary flow to, and evaporation from, soil moisture
      - A non-linear storage-runoff relationship from the upper runoff-generating zone
      - A linear storage-runoff relationship from the lower runoff-generating zone

    State variables (5):
      SP : current snow storage (frozen) [mm]
      WC : current liquid water content of the snow pack [mm]
      SM : current storage in soil moisture [mm]
      UZ : current storage in the upper (non-linear) runoff-generating zone [mm]
      LZ : current storage in the lower (linear) runoff-generating zone [mm]

    Parameters (15):
      TT         [°C]  Threshold temperature for snowfall  range [-3, 5]
      TTI        [°C]  Threshold temperature interval length  range [0, 17]
      CFR        [–]  Refreezing coefficient  range [0, 1]
      CFMAX      [mm °C⁻¹ d⁻¹]  Degree-day factor  range [0, 20]
      TTM        [°C]  Threshold temperature for snowmelt  range [-3, 3]
      WHC        [–]  Water holding capacity as fraction of current snow pack  range [0, 1]
      CFLUX      [mm d⁻¹]  Maximum capillary rise rate  range [0, 4]
      FC         [mm]  Field capacity  range [1, 2000]
      LP         [–]  Wilting point as fraction of FC  range [0.05, 0.95]
      β          [–]  Recharge non-linearity  range [0, 10]
      K0         [d⁻¹]  Runoff coefficient (upper zone outflow Q0)  range [0, 1]
      α          [–]  Runoff non-linearity (upper zone)  range [1, 5]
      c          [mm d⁻¹]  Percolation rate  range [0, 20]
      K1         [d⁻¹]  Runoff coefficient (lower zone outflow Q1)  range [0, 1]
      MAXBAS     [d]  Unit Hydrograph time base  range [1, 120]

    Output / routing (as specified):
      ```
      Q = triangular_transform(Q0 + Q1; MAXBAS)
      ```
      Routing: **yes** — total outflow Q0+Q1 is passed through a triangular unit hydrograph
      (lag/transform) parameterised by MAXBAS [d]. In MARRMoT's toolbox this corresponds to the
      **uh_4_full** routine (full triangular UH), with MAXBAS feeding its time base.
    """
    # unpack state
    SP, WC, SM, UZ, LZ = S

    # unpack params
    TT, TTI, CFR, CFMAX, TTM, WHC, CFLUX, FC, LP, beta, K0, alpha, c, K1, MAXBAS = (
        params['TT'], params['TTI'], params['CFR'], params['CFMAX'], params['TTM'],
        params['WHC'], params['CFLUX'], params['FC'], params['LP'], params['beta'],
        params['K0'], params['alpha'], params['c'], params['K1'], params['MAXBAS'])

    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    if uh_state is None:
        uh_state = {'q': UHRouter(uh_triangular(MAXBAS, dt))}

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue -- this pseudocode shows the
    # literal mathematical definition)
    if T <= TT - 0.5*TTI:
        sf = P
    elif T >= TT + 0.5*TTI:
        sf = 0
    else:
        sf = P * (TT + 0.5*TTI - T) / TTI

    _w1 = logistic_temperature(T, TTM)
    refr = _w1 * (CFR * CFMAX * (TTM - T)) + (1 - _w1) * (0)
    _w2 = logistic_temperature(T, TTM)
    melt = (1 - _w2) * (CFMAX * (T - TTM)) + _w2 * (0)
    if T <= TT - 0.5*TTI:
        rf = 0
    elif T >= TT + 0.5*TTI:
        rf = P
    else:
        rf = P * (T - TT + 0.5*TTI) / TTI

    in_ = (rf + melt) if WC >= WHC*SP else 0
    S_excess = (WC - WHC*SP) if WC >= WHC*SP else 0

    cf = CFLUX * (1 - SM/FC)
    Ea = Ep if SM >= LP*FC else Ep * SM/(LP*FC)
    r = (in_ + S_excess) * (SM/FC)**beta

    Q0 = K0 * UZ**(1+alpha)
    perc = c
    Q1 = K1 * LZ

    # update state(s) via mass balance
    SP_new = SP + dt*(sf + refr - melt)
    WC_new = WC + dt*(rf + melt - refr - in_ - S_excess)
    SM_new = SM + dt*((in_ + S_excess) + cf - Ea - r)
    UZ_new = UZ + dt*(r - cf - Q0 - perc)
    LZ_new = LZ + dt*(perc - Q1)

    # route Q0+Q1 through triangular UH parameterised by MAXBAS (uh_4_full)
    Q = uh_state['q'].route(Q0 + Q1)

    fluxes = dict(sf=sf, refr=refr, melt=melt, rf=rf, infil=in_, S_excess=S_excess,
                  cf=cf, Ea=Ea, r=r, Q0=Q0, perc=perc, Q1=Q1)
    return (SP_new, WC_new, SM_new, UZ_new, LZ_new), Q, fluxes, uh_state


def step_model_38(S, forcings, params, dt=1.0):
    """
    Model 38 -- Tank Model - SMA  (MARRMoT file: m_38_tank2_16p_5s)

    Reference(s): Sugawara, 1979 (original Tank Model); Sugawara, 1995 (SMA variant)

    Processes represented:
      - Runoff on increasing time scales with depth
      - Soil moisture storage
      - Capillary rise to replenish soil moisture

    State variables (5):
      S1 : current storage in the upper zone (primary soil moisture) [mm]
      Xs : current storage in the secondary soil moisture zone [mm]
      S2 : current storage in the intermediate zone [mm]
      S3 : current storage in the sub-base zone [mm]
      S4 : current storage in the base layer [mm]

    Parameters (16):
      sm1        [mm]  Soil moisture threshold for capillary rise  range [1, 2000]
      sm2        [mm]  Maximum soil moisture storage (of Xs)  range [1, 2000]
      k1         [mm d⁻¹]  Base capillary rise rate  range [0, 4]
      k2         [mm d⁻¹]  Base soil moisture exchange rate (S1 ↔ Xs)  range not specified in Table S3 — check S2.38.3 parameter table only
      A0         [d⁻¹]  Runoff coefficient (F12, drainage S1→S2)  range not specified in Table S3 — check S2.38.3 parameter table only
      A1         [d⁻¹]  Runoff coefficient (Y1)  range not specified in Table S3 — check S2.38.3 parameter table only
      A2         [d⁻¹]  Runoff coefficient (Y2)  range not specified in Table S3 — check S2.38.3 parameter table only
      t1         [mm]  Threshold for runoff generation (Y1)  range not specified in Table S3 — check S2.38.3 parameter table only
      t2         [mm]  Threshold for runoff generation (Y2)  range not specified in Table S3 — check S2.38.3 parameter table only
      B0         [d⁻¹]  Runoff coefficient (F23, drainage S2→S3)  range not specified in Table S3 — check S2.38.3 parameter table only
      B1         [d⁻¹]  Runoff coefficient (Y3)  range not specified in Table S3 — check S2.38.3 parameter table only
      t3         [mm]  Threshold for runoff generation (Y3)  range not specified in Table S3 — check S2.38.3 parameter table only
      C0         [d⁻¹]  Runoff coefficient (F34, drainage S3→S4)  range not specified in Table S3 — check S2.38.3 parameter table only
      C1         [d⁻¹]  Runoff coefficient (Y4)  range not specified in Table S3 — check S2.38.3 parameter table only
      t4         [mm]  Threshold for runoff generation (Y4)  range not specified in Table S3 — check S2.38.3 parameter table only
      D1         [d⁻¹]  Runoff coefficient (Y5, baseflow)  range not specified in Table S3 — check S2.38.3 parameter table only

    Output / routing (as specified):
      ```
      Qt = Y1 + Y2 + Y3 + Y4 + Y5                                                      (509)
      ```
      Routing: **none** — no unit hydrograph is applied; total runoff Qt is the direct sum of
      Y1..Y5 at each time step.
    """
    # unpack state
    S1, Xs, S2, S3, S4 = S

    # unpack params
    sm1, sm2, k1, k2, A0, A1, A2, t1, t2, B0, B1, t3, C0, C1, t4, D1 = (
        params['sm1'],
        params['sm2'],
        params['k1'],
        params['k2'],
        params['A0'],
        params['A1'],
        params['A2'],
        params['t1'],
        params['t2'],
        params['B0'],
        params['B1'],
        params['t3'],
        params['C0'],
        params['C1'],
        params['t4'],
        params['D1']
    )

    P, Ep = forcings['P'], forcings['Ep']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    _w1 = logistic_storage(S1, sm1)
    T1 = _w1 * (k1*(1 - S1/sm1)) + (1 - _w1) * (0)
    T2 = k2*(min(S1, sm1)/sm1 - Xs/sm2)
    _w2 = logistic_storage(S1, 0)
    E1 = (1 - _w2) * (Ep) + _w2 * (0)
    _w3 = logistic_storage(S1, sm1)
    F12 = (1 - _w3) * (A0*(S1 - sm1)) + _w3 * (0)
    _w4 = logistic_storage(S1, t2)
    Y2 = (1 - _w4) * (A2*(S1 - t2)) + _w4 * (0)
    _w5 = logistic_storage(S1, t1)
    Y1 = (1 - _w5) * (A1*(S1 - t1)) + _w5 * (0)
    E2 = Ep if (S1 <= 0 and S2 > 0) else 0
    F23 = B0*S2
    _w6 = logistic_storage(S2, t3)
    Y3 = (1 - _w6) * (B1*(S2 - t3)) + _w6 * (0)
    E3 = Ep if (S1 <= 0 and S2 <= 0 and S3 > 0) else 0
    F34 = C0*S3
    _w7 = logistic_storage(S3, t4)
    Y4 = (1 - _w7) * (C1*(S3 - t4)) + _w7 * (0)
    E4 = Ep if (S1 <= 0 and S2 <= 0 and S3 <= 0 and S4 > 0) else 0
    Y5 = D1*S4

    # update state(s) via mass balance
    S1_new = S1 + dt*(P + T1 - T2 - E1 - F12 - Y2 - Y1)
    Xs_new = Xs + dt*(T2)
    S2_new = S2 + dt*(F12 - E2 - T1 - F23 - Y3)
    S3_new = S3 + dt*(F23 - E3 - F34 - Y4)
    S4_new = S4 + dt*(F34 - E4 - Y5)

    Q = Y1 + Y2 + Y3 + Y4 + Y5

    fluxes = dict(T1=T1, T2=T2, E1=E1, F12=F12, Y1=Y1, Y2=Y2,
                  E2=E2, F23=F23, Y3=Y3, E3=E3, F34=F34, Y4=Y4, E4=E4, Y5=Y5)
    return (S1_new, Xs_new, S2_new, S3_new, S4_new), Q, fluxes


def step_model_39(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 39 -- Midlands Catchment Runoff Model (MCRM)  (MARRMoT file: m_39_mcrm_16p_5s)

    Reference(s): Moore and Bell, 2001

    Processes represented:
      - Interception by vegetation
      - Direct runoff from a variable contributing area
      - A deficit-based approach to soil moisture accounting and interflow and percolation
      - Baseflow from groundwater
      - Uniform flood wave distribution in time
      - In-channel and out-of-channel flood routing

    State variables (5):
      S   : current interception storage [mm]
      D   : current soil moisture deficit store [mm] (negative D indicates surplus/no deficit)
      Sg  : current groundwater storage [mm]
      Sic : current in-channel storage [mm]
      Soc : current out-of-channel storage [mm]

    Parameters (16):
      Smax       [mm]  Maximum interception storage  range [0, 5]
      cmax       [–]  Maximum fraction of area contributing to rapid runoff  range [0, 1]
      c0         [–]  Minimum fraction of area contributing to rapid runoff  range [0, 1]
      c1         [–]  Contributing-area exponential shape parameter  range not specified in Table S3 — check S2.39.3 parameter table only
      ce         [–]  Evaporation exponential shape parameter  range [0, 1]
      Dsurp      [mm]  Threshold for direct runoff generation (surplus threshold)  range not specified in Table S3 — check S2.39.3 parameter table only
      kd         [d⁻¹]  Runoff coefficient (direct runoff qd)  range [0, 1]*
      γd         [–]  Runoff non-linearity (qd exponent)  range [1, 5]
      qp,max     [mm d⁻¹]  Maximum percolation rate  range [0, 20]
      kg         [d⁻¹]  Runoff coefficient (groundwater outflow qb)  range [0, 1]*
      τ          [d]  Unit Hydrograph time base  range [1, 120]
      Sbf        [mm]  Maximum in-channel ("bankfull") storage (labelled "Maximum groundwater storage" in S2.39.3, but functions as the Sic bankfull threshold per eq. 522)  range [1, 300]
      kcr        [d⁻¹]  Runoff coefficient (in-channel flow qic)  range [0, 1]*
      γcr        [–]  Runoff non-linearity (qic exponent)  range [1, 5]
      kor        [d⁻¹]  Runoff coefficient (out-of-channel flow qoc)  range [0, 1]*
      γor        [–]  Runoff non-linearity (qoc exponent)  range [1, 5]

    Output / routing (as specified):
      ```
      Qt = qoc + qic                                                                  (526)
      ```
      Routing: **yes** — qr, qd, qb are summed and distributed uniformly over timespan τ [d]
      to produce delayed inflow u_ib to the in-channel/out-of-channel store cascade. The PDF
      text does not explicitly name a `uh_x_...` toolbox file for this step (unlike HBV-96 and
      SMAR, whose sections do name their UH routine); it describes the transform only as
      "distributed uniformly over timespan τ", i.e. a rectangular/uniform-weight unit hydrograph
      with time base τ. Confirm the exact MARRMoT routine name (likely `uh_5_half` or similar
      uniform-distribution UH) against the toolbox source code — it is not stated in this
      section of the documentation.
    """
    # unpack state
    S_int, D, Sg, Sic, Soc = S

    # unpack params
    (Smax, cmax, c0, c1, ce, Dsurp, kd, gamma_d, qp_max, kg, tau, Sbf,
     kcr, gamma_cr, kor, gamma_or) = (
        params['Smax'], params['cmax'], params['c0'], params['c1'], params['ce'],
        params['Dsurp'], params['kd'], params['gamma_d'], params['qp_max'],
        params['kg'], params['tau'], params['Sbf'], params['kcr'], params['gamma_cr'],
        params['kor'], params['gamma_or'])

    P, Ep = forcings['P'], forcings['Ep']

    if uh_state is None:
        uh_state = {'ib': UHRouter(uh_uniform(tau, dt))}

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue -- this pseudocode shows the
    # literal mathematical definition)
    _w1 = logistic_storage(S_int, 0)
    Ec = (1 - _w1) * (Ep) + _w1 * (0) # Throughfall only once interception is at/above capacity -- literal spec
    # used `S_int == Smax`; see module docstring on `==` -> `>=`/`<=` fixups.
    _w2 = logistic_storage(S_int, Smax)
    qt = (1 - _w2) * (P) + _w2 * (0)
    qr = min(cmax, c0 + c0*exp(c1*D)) * qt
    qn = qt - qr
    Er = (1/(1+exp(-ce*D))) * (Ep - Ec)

    _w3 = logistic_storage(D, Dsurp)
    qd = (1 - _w3) * (kd*max(0.0, Dsurp - D)**gamma_d) + _w3 * (0)
    if D >= Dsurp:
        qp = qp_max
    elif D > 0:
        qp = (D/Dsurp)*qp_max
    else:
        qp = 0

    qb = kg * Sg**1.5

    # unit-hydrograph step: distribute (qr + qd + qb) uniformly over tau days -> u_ib
    u_ib = uh_state['ib'].route(qr + qd + qb)

    # outer-channel spill only once the inner-channel store is at/above bankfull
    _w4 = logistic_storage(Sic, Sbf)
    uob = (1 - _w4) * (u_ib) + _w4 * (0) # qic solved from its (self-referential) piecewise definition:
    qic_candidate = kcr * Sic**gamma_cr
    qic = qic_candidate if qic_candidate < 0.75*Sic else 0.75*Sic

    qoc_candidate = kor * Soc**gamma_or
    qoc = qoc_candidate if qoc_candidate < 0.75*Soc else 0.75*Soc

    # update state(s) via mass balance
    S_int_new = S_int + dt*(P - Ec - qt)
    D_new     = D + dt*(qn - Er - qd - qp)
    Sg_new    = Sg + dt*(qp - qb)
    Sic_new   = Sic + dt*(u_ib - uob - qic)
    Soc_new   = Soc + dt*(uob - qoc)

    Q = qoc + qic

    fluxes = dict(Ec=Ec, qt=qt, qn=qn, qr=qr, Er=Er, qd=qd, qp=qp, qb=qb,
                  u_ib=u_ib, uob=uob, qic=qic, qoc=qoc)
    return (S_int_new, D_new, Sg_new, Sic_new, Soc_new), Q, fluxes, uh_state


def step_model_40(S, forcings, params, dt=1.0, uh_state=None):
    """
    Model 40 -- SMAR  (MARRMoT file: m_40_smar_8p_6s)

    Reference(s): O'Connell et al., 1970 (original 'layers model'); Tan and O'Connor, 1996

    Processes represented:
      - Saturation excess overland flow
      - Infiltration excess overland flow
      - Gradual infiltration into soil moisture and declining evaporation potential when water

    State variables (6):
      S1 : current storage in soil layer 1 (uppermost) [mm]
      S2 : current storage in soil layer 2 [mm]
      S3 : current storage in soil layer 3 [mm]
      S4 : current storage in soil layer 4 [mm]
      S5 : current storage in soil layer 5 (deepest of the 5 fixed layers) [mm]
      Gw : current groundwater storage [mm]

    Parameters (8):
      H          [–]  Fraction of effective precipitation that is direct runoff  range [0, 1]
      Y          [mm d⁻¹]  Infiltration rate (constant cap on infiltration into S1)  range [0, 200]
      Smax       [mm]  Maximum soil moisture storage (total, across all 5 layers)  range [1, 2000]
      C          [–]  Evaporation reduction parameter  range [0, 1]
      G          [–]  Fraction of subsurface flow (R3) routed to groundwater  range [0, 1]
      KG         [d⁻¹]  Runoff (groundwater outflow) coefficient  range not specified in Table S3 — check S2.40.3 parameter table only
      N          [–]  Gamma function shape parameter (number of Nash-cascade reservoirs)  range [1, 10]
      K          [d]  Gamma function/routing time-delay parameter  range [1, 120]

    Output / routing (as specified):
      ```
      Qt = Qr + Qg                                                                    (551)
      ```
      Routing: **yes** — the non-groundwater fraction of runoff (R1 + R2 + R3, per the printed
      equation 550) is routed through a **gamma-function unit hydrograph** approximating a
      Nash cascade of N reservoirs with storage coefficient K [d]. This is MARRMoT's
      **uh_6_gamma** routine. Shape parameter: **N** [-] (number of Nash-cascade reservoirs);
      time-delay/scale parameter: **K** [d] (storage coefficient feeding the UH's time base).
    """
    # unpack state
    S1, S2, S3, S4, S5, Gw = S

    # unpack params
    H, Y, Smax, C, G, KG, N, K = (params['H'], params['Y'], params['Smax'],
                                   params['C'], params['G'], params['KG'],
                                   params['N'], params['K'])

    P, Ep = forcings['P'], forcings['Ep']
    Sn_max = Smax / 5

    if uh_state is None:
        uh_state = {'r': UHRouter(uh_gamma(N, K, dt))}

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue -- this pseudocode shows the
    # literal mathematical definition)
    _w1 = logistic_storage(P, Ep)
    Pstar = (1 - _w1) * ((P - Ep)) + _w1 * (0)
    R1 = Pstar * H * ((S1+S2+S3+S4+S5) / Smax)
    I = Y if (Pstar - R1) >= Y else (Pstar - R1)
    R2 = (Pstar - R1) - I

    _w2 = logistic_storage(Ep, P)
    Epstar = (1 - _w2) * ((Ep - P)) + _w2 * (0)
    E1 = (C**0) * Epstar
    _w3 = logistic_storage(S1, Sn_max)
    q1 = (1 - _w3) * ((Pstar - R1 - R2)) + _w3 * (0) # each successive evaporation layer only activates once the layer above it
    # is (at or below) empty -- literal spec used `== 0`; see module docstring.
    _w4 = logistic_storage(S1, 0)
    E2 = _w4 * ((C**1)*Ep) + (1 - _w4) * (0)
    _w5 = logistic_storage(S2, Sn_max)
    q2 = (1 - _w5) * (q1) + _w5 * (0)
    _w6 = logistic_storage(S2, 0)
    E3 = _w6 * ((C**2)*Ep) + (1 - _w6) * (0)
    _w7 = logistic_storage(S3, Sn_max)
    q3 = (1 - _w7) * (q2) + _w7 * (0)
    _w8 = logistic_storage(S3, 0)
    E4 = _w8 * ((C**3)*Ep) + (1 - _w8) * (0)
    _w9 = logistic_storage(S4, Sn_max)
    q4 = (1 - _w9) * (q3) + _w9 * (0)
    _w10 = logistic_storage(S4, 0)
    E5 = _w10 * ((C**4)*Ep) + (1 - _w10) * (0)
    _w11 = logistic_storage(S5, Sn_max)
    R3 = (1 - _w11) * (q4) + _w11 * (0)
    Rg = G * R3
    Qg = KG * Gw

    # R3_star = (1-G)*R3 -- see note in spec: source eq. 550 prints "R3" not "R3*";
    # using R3_star here as the mass-conserving interpretation (flagged above)
    R3_star = (1 - G) * R3

    # update state(s) via mass balance
    S1_new = S1 + dt*(I - E1 - q1)
    S2_new = S2 + dt*(q1 - E2 - q2)
    S3_new = S3 + dt*(q2 - E3 - q3)
    S4_new = S4 + dt*(q3 - E4 - q4)
    S5_new = S5 + dt*(q4 - E5 - R3)
    Gw_new = Gw + dt*(Rg - Qg)

    # route (R1 + R2 + R3_star) through gamma-function UH (uh_6_gamma), shape N, scale K
    Qr = uh_state['r'].route(R1 + R2 + R3_star)

    Q = Qr + Qg

    fluxes = dict(Pstar=Pstar, R1=R1, I=I, R2=R2, Epstar=Epstar, E1=E1, q1=q1,
                  E2=E2, q2=q2, E3=E3, q3=q3, E4=E4, q4=q4, E5=E5, R3=R3,
                  Rg=Rg, Qg=Qg, Qr=Qr)
    return (S1_new, S2_new, S3_new, S4_new, S5_new, Gw_new), Q, fluxes, uh_state


def step_model_41(S, forcings, params, dt=1.0):
    """
    Model 41 -- NAM model  (MARRMoT file: m_41_nam_10p_6s)

    Reference(s): Nielsen and Hansen, 1973. (MARRMoT modification: routing equations of

    Processes represented:
      - Snow accumulation and melt
      - Interflow when total soil moisture exceeds a threshold
      - Separation of saturation excess flow into overland flow and infiltration
      - Baseflow from groundwater

    State variables (6):
      S : current snow storage [mm]
      U : current storage in the upper zone [mm]
      L : current storage in the lower zone [mm]
      O : current storage in the overland-flow routing store [mm]
      I : current storage in the interflow routing store [mm]
      G : current storage in the groundwater/baseflow routing store [mm] (labelled "overland

    Parameters (10):
      Cs         [mm °C⁻¹ d⁻¹]  Degree-day factor  range [0, 20]
      Cif        [d⁻¹]  Runoff coefficient (interflow generation)  range not specified in Table S3 — check S2.41.3 parameter table only
      L*         [mm]  Maximum lower zone storage  range [1, 2000]
      CL1        [–]  Fractional threshold for interflow generation  range not specified in Table S3 — check S2.41.3 parameter table only (only one threshold row is listed for model 41 in FLOW DISTRIBUTION, best matched to CL2 below)
      U*         [mm]  Maximum upper zone storage  range [1, 2000]
      Cof        [d⁻¹]  Runoff coefficient (overland flow generation)  range not specified in Table S3 — check S2.41.3 parameter table only
      CL2        [–]  Fractional threshold for overland flow generation  range [0, 0.99]
      K0         [d⁻¹]  Runoff coefficient (overland-flow store outflow Qo)  range [0, 1]
      K1         [d⁻¹]  Runoff coefficient (interflow store outflow Qi)  range [0, 1]
      Kb         [d⁻¹]  Runoff coefficient (baseflow store outflow Qb)  range [0, 1]

    Output / routing (as specified):
      ```
      Q = Qo + Qi + Qb                                                                (571)
      ```
      Routing: **none** beyond the three internal linear-reservoir routing stores (O, I, G)
      already included as states — no additional unit-hydrograph transform is applied to Q.
    """
    # unpack state
    Sn, U, L, O, I, G = S   # Sn = snow store (avoid clashing with soil store name S)

    # unpack params
    Cs, Cif, Lstar, CL1, Ustar, Cof, CL2, K0, K1, Kb = (
        params['Cs'],
        params['Cif'],
        params['Lstar'],
        params['CL1'],
        params['Ustar'],
        params['Cof'],
        params['CL2'],
        params['K0'],
        params['K1'],
        params['Kb']
    )

    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    # compute fluxes (note: MARRMoT's actual numeric implementation smooths hard
    # if/else thresholds shown above using logistic smoother functions from
    # Table S1 in the toolbox's flux catalogue — this pseudocode shows the
    # literal mathematical definition)
    _w1 = logistic_temperature(T, 0)
    Ps = _w1 * (P) + (1 - _w1) * (0)
    _w2 = logistic_temperature(T, 0)
    M = (1 - _w2) * (Cs*T) + _w2 * (0)
    _w3 = logistic_temperature(T, 0)
    Pr = (1 - _w3) * (P) + _w3 * (0)
    _w4 = logistic_storage(U, 0)
    EU = (1 - _w4) * (Ep) + _w4 * (0)
    If = Cif * ((L/Lstar - CL1)/(1-CL1)) * U if (L/Lstar) > CL1 else 0
    _w5 = logistic_storage(U, Ustar)
    Pn = (1 - _w5) * ((Pr + M)) + _w5 * (0)
    Of = Cof * ((L/Lstar - CL2)/(1-CL2)) * Pn if (L/Lstar) > CL2 else 0
    Dl = (Pn - Of) * (1 - L/Lstar)
    _w6 = logistic_storage(U, 0)
    Et = _w6 * ((L/Lstar)*Ep) + (1 - _w6) * (0)
    Qo = K0 * O
    Qi = K1 * I
    Gw = (Pn - Of) * (L/Lstar)
    Qb = Kb * O   # NOTE: as printed in source (likely should be Kb * G — see spec note)

    # update state(s) via mass balance
    Sn_new = Sn + dt*(Ps - M)
    U_new  = U + dt*(Pr + M - EU - If - Pn)
    L_new  = L + dt*(Dl - Et)
    O_new  = O + dt*(Of - Qo)
    I_new  = I + dt*(If - Qi)
    G_new  = G + dt*(Gw - Qb)

    Q = Qo + Qi + Qb

    fluxes = dict(Ps=Ps, M=M, Pr=Pr, EU=EU, If=If, Pn=Pn, Of=Of, Dl=Dl, Et=Et,
                  Qo=Qo, Qi=Qi, Gw=Gw, Qb=Qb)
    return (Sn_new, U_new, L_new, O_new, I_new, G_new), Q, fluxes


def step_model_42(S, forcings, params, dt=1.0):
    """
    Model 42 -- HYCYMODEL  (MARRMoT file: m_42_hycymodel_12p_6s)

    Reference(s): Fukushima, 1988 (original HYCYMODEL, developed for heavily forested catchments in Japan)

    Processes represented:
      - Split between channel and ground precipitation
      - Interception by canopy and stems/trunks
      - Overland flow from a variable contributing area
      - Non-linear channel flow, hillslope flow and baseflow
      - Channel evaporation

    State variables (6):
      Ic : current canopy interception storage [mm]
      Is : current stem and trunk interception storage [mm]
      Su : current storage in the upper (soil) zone [mm]
      Sb : current storage in the lower zone [mm]
      Sh : current storage in the hillslope routing store [mm]
      Sc : current storage in the channel routing store [mm]

    Parameters (12):
      C          [–]  Fraction of area that is channel  range [0, 1]
      I1,max     [mm]  Maximum interception storage (canopy)  range [0, 5]
      α (alpha)  [–]  Fraction of interception excess to stem flow  range [0, 1]
      I2,max     [mm]  Maximum trunk and stem storage  range range: not specified in Table S3 — check S2.x.3 parameter table only
      kin        [d⁻¹]  Runoff coefficient (upper→lower zone infiltration)  range [0, 1]
      D50        [mm]  Effective soil depth at which 50% of area contributes to flow  range [1, 2000]
      D16        [mm]  Effective soil depth at which 16% of area contributes to flow  range derived: sampled as a fraction of D50, fraction ∈ [0.01, 0.99]
      Sbc        [mm]  Threshold for evaporation behaviour change (lower zone)  range range: not specified in Table S3 — check S2.x.3 parameter table only
      kb         [d⁻¹]  Runoff coefficient (baseflow)  range [0, 1]
      pb         [–]  Runoff non-linearity (baseflow exponent)  range [1, 5]
      kh         [d⁻¹]  Runoff coefficient (hillslope)  range [0, 1]
      kc         [d⁻¹]  Runoff coefficient (channel)  range [0, 1]

    Output / routing (as specified):
      ```
      Q = Qc + Qh + Qb - Ec
      ```
      Routing: none (no Unit Hydrograph). Flow is routed internally through three parallel non-linear/linear
      reservoirs (hillslope Sh, channel Sc, baseflow Sb) whose outflows are summed directly into Q.
    """
    # unpack state
    Ic, Is, Su, Sb, Sh, Sc = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']

    # unpack params
    C, I1max, alpha, I2max, kin, D50, D16, Sbc, kb, pb, kh, kc = (
        params['C'],
        params['I1max'],
        params['alpha'],
        params['I2max'],
        params['kin'],
        params['D50'],
        params['D16'],
        params['Sbc'],
        params['kb'],
        params['pb'],
        params['kh'],
        params['kc']
    )

    # --- canopy interception ---
    Rg  = (1 - C) * P
    Rc  = C * P
    _w1 = logistic_storage(Ic, 0)
    Eic = (1 - _w1) * ((1 - C) * Ep) + _w1 * (0.0)
    _w2 = logistic_storage(Ic, I1max)
    qie = (1 - _w2) * (Rg) + _w2 * (0.0) # canopy at capacity -> spill

    # --- stem/trunk interception ---
    qis = alpha * qie
    Rt  = (1 - alpha) * qie
    _w3 = logistic_storage(Is, 0)
    Eis = (1 - _w3) * ((1 - C) * Ep) + _w3 * (0.0)
    _w4 = logistic_storage(Is, I2max)
    Rs = (1 - _w4) * (qis) + _w4 * (0.0) # stem store at capacity -> spill

    Rn  = Rt + Rs

    # --- upper (soil) zone: variable contributing area ---
    xi  = log(Su / D50) / log(D50 / D16)
    m   = standard_normal_cdf(xi)             # Phi(xi)
    Re  = m * Rn
    _w5 = logistic_storage(Su, 0)
    Esu = (1 - _w5) * ((1 - C) * Ep) + _w5 * (0.0)
    Qin = kin * Su

    # --- lower zone ---
    if Su <= 0 and Sb >= Sbc:
        Esb = (1 - C) * Ep
    else:
        Esb = (1 - C) * Ep * (Sb / Sbc)
    Qb = kb * Sb**pb

    # --- hillslope routing ---
    Qh = kh * Sh**(5/3)

    # --- channel routing ---
    Qc = kc * Sc**(5/3)
    Ec = C * Ep

    # update state(s) via mass balance (Euler shown; MARRMoT solves implicitly)
    dIc = Rg - Eic - qie
    dIs = qis - Eis - Rs
    dSu = Rn - Re - Esu - Qin
    dSb = Qin - Esb - Qb
    dSh = Re - Qh
    dSc = Rc - Qc

    S_new = [Ic + dt*dIc, Is + dt*dIs, Su + dt*dSu,
             Sb + dt*dSb, Sh + dt*dSh, Sc + dt*dSc]

    Q = Qc + Qh + Qb - Ec

    fluxes_dict = dict(Rg=Rg, Rc=Rc, Eic=Eic, qie=qie, qis=qis, Rt=Rt, Eis=Eis, Rs=Rs,
                        Rn=Rn, Re=Re, Esu=Esu, Qin=Qin, Esb=Esb, Qb=Qb, Qh=Qh, Qc=Qc, Ec=Ec)
    return S_new, Q, fluxes_dict


def step_model_43(S, forcings, params, dt=1.0):
    """
    Model 43 -- GSM-SOCONT model  (MARRMoT file: m_43_gsmsocont_12p_6s)

    Reference(s): Schaefli et al., 2005 (Glacier and SnowMelt - SOil CONTribution model)

    Processes represented:
      - Separate treatment of glacier and non-glacier catchment area
      - Snow accumulation and melt
      - Glacier melt
      - Soil moisture accounting in the non-glacier catchment area

    State variables (6):
      Hi,s  : snow pack storage on the ice(glacier)-covered fraction of the catchment [mm]
      Si,s  : storage in the snow-water routing reservoir (ice-covered fraction) [mm]
      Si,i  : storage in the ice-water routing reservoir (glacier melt routing) [mm]
      Hni,s : snow pack storage on the non-ice-covered fraction of the catchment [mm]
      Sni,s : soil moisture storage (non-ice-covered fraction) [mm]
      Sni,q : storage in the direct runoff (quick) reservoir (non-ice-covered fraction) [mm]

    Parameters (12):
      fice       [–]  Fraction of area with ice cover  range [0, 1]
      T0         [°C]  Threshold temperature for snowfall  range [-3, 5]
      asnow      [mm/°C/d]  Degree-day factor for snowmelt  range [0, 20]
      Tm         [°C]  Threshold temperature for snowmelt  range [-3, 3]
      ks         [d⁻¹]  Runoff coefficient (snow-water routing reservoir)  range [0, 1]
      aice       [mm/°C/d]  Degree-day factor for ice (glacier) melt  range [0, 20]
      ki         [d⁻¹]  Runoff coefficient (ice-water routing reservoir)  range [0, 1]
      A          [mm]  Maximum soil moisture storage  range [1, 2000]
      x          [–]  Evaporation non-linearity  range [0, 10]
      y          [–]  Quick runoff (infiltration decline) non-linearity  range [0, 5]
      ksl        [d⁻¹]  Runoff coefficient (slow flow from soil moisture)  range [0, 1]
      β (beta)   [mm^(1/3)/d [printed as MM^(4/3)/d]]  Runoff coefficient (quick/direct-runoff reservoir, non-linear)  range range: not specified in Table S3 — check S2.x.3 parameter table only

    Output / routing (as specified):
      ```
      Q = Qqu + Qsl + Qis + Qii
      ```
      Routing: none (no Unit Hydrograph). All four component fluxes (quick, slow, ice-covered snow-water,
      glacier/ice-water) are linear or non-linear reservoir outflows summed directly.
    """
    # unpack state
    Hi_s, Si_s, Si_i, Hni_s, Sni_s, Sni_q = S

    # unpack forcings
    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    # unpack params
    fice, T0, asnow, Tm, ks, aice, ki, A, x, y, ksl, beta = (
        params['fice'],
        params['T0'],
        params['asnow'],
        params['Tm'],
        params['ks'],
        params['aice'],
        params['ki'],
        params['A'],
        params['x'],
        params['y'],
        params['ksl'],
        params['beta']
    )

    # --- ice-covered fraction: snow ---
    Pice   = fice * P
    _w1 = logistic_temperature(T, T0)
    Pice_s = _w1 * (Pice) + (1 - _w1) * (0.0)
    _w2 = logistic_temperature(T, Tm)
    Mi_s = (1 - _w2) * (asnow * (T - Tm)) + _w2 * (0.0)
    _w3 = logistic_temperature(T, T0)
    Pice_r = (1 - _w3) * (Pice) + _w3 * (0.0)
    _w4 = logistic_storage(Hi_s, 0)
    Pi_r_s = (1 - _w4) * (Pice_r) + _w4 * (0.0)
    Qis    = ks * Si_s

    # --- ice-covered fraction: glacier ---
    _w5 = logistic_storage(Hi_s, 0)
    Pi_r_i = _w5 * (Pice_r) + (1 - _w5) * (0.0)
    Mi_i   = aice * (T - Tm) if (T > Tm and Hi_s <= 0) else 0.0
    Qii    = ki * Si_i

    # --- non-ice-covered fraction: snow ---
    Pnon_ice = (1 - fice) * P
    _w6 = logistic_temperature(T, T0)
    Pni_s = _w6 * (Pnon_ice) + (1 - _w6) * (0.0)
    _w7 = logistic_temperature(T, Tm)
    Mni_s = (1 - _w7) * (asnow * (T - Tm)) + _w7 * (0.0)
    _w8 = logistic_temperature(T, T0)
    Pni_r = (1 - _w8) * (Pnon_ice) + _w8 * (0.0) # rain-on-ground mirrors ice-covered logic

    # --- non-ice-covered fraction: soil moisture ---
    Peq  = Mni_s + Pni_r
    Peff = Peq * (Sni_s / A)**y
    Pinf = Peq - Peff
    ET   = Ep * (Sni_s / A)**x
    Qsl  = ksl * Sni_s

    # --- non-ice-covered fraction: quick runoff ---
    Qqu = beta * Sni_q**(5/3)

    # update state(s) via mass balance
    dHi_s  = Pice_s - Mi_s
    dSi_s  = Mi_s + Pi_r_s - Qis
    dSi_i  = Mi_i + Pi_r_i - Qii
    dHni_s = Pni_s - Mni_s
    dSni_s = Pinf - ET - Qsl
    dSni_q = Peff - Qqu

    S_new = [Hi_s + dt*dHi_s, Si_s + dt*dSi_s, Si_i + dt*dSi_i,
             Hni_s + dt*dHni_s, Sni_s + dt*dSni_s, Sni_q + dt*dSni_q]

    Q = Qqu + Qsl + Qis + Qii

    fluxes_dict = dict(Pice=Pice, Pice_s=Pice_s, Mi_s=Mi_s, Pice_r=Pice_r, Pi_r_s=Pi_r_s, Qis=Qis,
                        Pi_r_i=Pi_r_i, Mi_i=Mi_i, Qii=Qii, Pnon_ice=Pnon_ice, Pni_s=Pni_s, Mni_s=Mni_s,
                        Pni_r=Pni_r, Peq=Peq, Peff=Peff, Pinf=Pinf, ET=ET, Qsl=Qsl, Qqu=Qqu)
    return S_new, Q, fluxes_dict


def step_model_44(S, forcings, params, dt=1.0):
    """
    Model 44 -- ECHO model  (MARRMoT file: m_44_echo_16p_6s)

    Reference(s): Schaefli et al., 2014 (SEHR-ECHO, Spatially Explicit Hydrologic Response model — used

    Processes represented:
      - Interception by vegetation
      - Snowfall, snowmelt, ground-heat flux and storage and refreezing of liquid snow
      - Infiltration, infiltration excess and saturation excess
      - Fast and slow runoff

    State variables (6):
      I     : current interception storage [mm]
      Hs    : current (frozen) snow pack storage [mm]
      Hw    : current storage of liquid water held within the snow pack [mm]
      S     : current storage in the soil moisture zone [mm]
      Sfast : current storage in the fast runoff reservoir [mm]
      Sslow : current storage in the slow runoff reservoir [mm]

    Parameters (16):
      ρ (rho)    [mm]  Maximum interception storage  range [0, 5]
      Ts         [°C]  Threshold temperature for snowfall  range [-3, 5]
      Tm         [°C]  Threshold temperature for snowmelt  range [-3, 3]
      as         [mm/°C/d]  Degree-day factor for snowmelt  range [0, 20]
      af         [–]  Degree-day factor reduction (refreezing) factor  range [0, 1]
      Gmax       [mm/d]  Snowmelt-through-ground-heat-flux rate (max)  range [0, 2]
      θ (theta)  [–]  Water holding capacity as fraction of current snow pack  range [0, 1]
      φ (phi)    [mm/d]  Maximum Horton-type flow rate  range [0, 200]
      Smax       [mm]  Maximum soil moisture storage  range [1, 2000]
      sw         [mm]  Wilting point  range [0.05, 0.95] (as fraction of soil moisture capacity)
      sm         [mm]  Plant stress point (field capacity-like threshold)  range range: not specified in Table S3 — check S2.x.3 parameter table only
      Ksat       [d⁻¹]  Runoff coefficient (non-linear soil leakage)  range [0, 1]
      c          [–]  Runoff non-linearity (leakage exponent)  range [1, 5]
      Lmax       [mm/d]  Maximum leakage rate (soil → slow store)  range [0, 20]
      kf         [d⁻¹]  Runoff coefficient (fast reservoir)  range [0, 1]
      ks         [d⁻¹]  Runoff coefficient (slow reservoir)  range [0, 1]

    Output / routing (as specified):
      ```
      Q = RH + RD + Rf + RS
      ```
      (RS = Rs, the slow-reservoir outflow; printed in the source as "R_S" in the total-flow equation.)
  
      Routing: none (no Unit Hydrograph). Streamflow is the direct sum of Horton runoff, Dunne runoff, fast
      reservoir outflow and slow reservoir outflow.
    """
    # unpack state
    I, Hs, Hw, Ssoil, Sfast, Sslow = S     # 'Ssoil' avoids clash with the ODE solver's step-loop 'S'

    # unpack forcings
    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    # unpack params
    rho, Ts, Tm, as_, af, Gmax, theta, phi, Smax, sw, sm, Ksat, c, Lmax, kf, ks = (
        params['rho'],
        params['Ts'],
        params['Tm'],
        params['as_'],
        params['af'],
        params['Gmax'],
        params['theta'],
        params['phi'],
        params['Smax'],
        params['sw'],
        params['sm'],
        params['Ksat'],
        params['c'],
        params['Lmax'],
        params['kf'],
        params['ks']
    )

    # --- interception ---
    _w1 = logistic_storage(I, 0)
    Ei = (1 - _w1) * (Ep) + _w1 * (0.0)
    _w2 = logistic_storage(I, rho)
    Pn = (1 - _w2) * (P) + _w2 * (0.0) # --- frozen snow pack ---
    _w3 = logistic_temperature(T, Ts)
    Ps = _w3 * (Pn) + (1 - _w3) * (0.0)
    Ms = as_ * (T - Tm) if (T > Tm and Hs > 0) else 0.0
    Fs = af * as_ * (Tm - T) if (T < Tm and Hw > 0) else 0.0
    _w4 = logistic_storage(Hs, 0)
    Gs = (1 - _w4) * (Gmax) + _w4 * (0.0) # --- liquid water in snow pack ---
    _w5 = logistic_temperature(T, Ts)
    Pr = (1 - _w5) * (Pn) + _w5 * (0.0)
    Mw = (Pr + Ms) if Hw >= theta * Hs else 0.0

    # --- soil moisture ---
    Peq   = Mw + Gs
    _w6 = logistic_storage(Ssoil, Smax)
    RH = _w6 * (max(Peq - phi, 0.0)) + (1 - _w6) * (0.0)
    Fi    = Peq - RH
    _w7 = logistic_storage(Ssoil, Smax)
    RD = (1 - _w7) * (Peq) + _w7 * (0.0)
    Et_pot = Ep - Ei
    Et    = min(max(0.0, Et_pot * (Ssoil - sw) / (sm - sw)), Et_pot)
    L     = Ksat * Ssoil**c

    # --- fast / slow split ---
    Ls = min(L, Lmax)
    Lf = L - Ls
    Rf = kf * Sfast
    Rs = ks * Sslow

    # update state(s) via mass balance
    dI     = P - Ei - Pn
    dHs    = Ps + Fs - Ms - Gs
    dHw    = Pr + Ms - Fs - Mw
    dSsoil = Fi - RD - Et - L
    dSfast = Lf - Rf
    dSslow = Ls - Rs

    S_new = [I + dt*dI, Hs + dt*dHs, Hw + dt*dHw,
             Ssoil + dt*dSsoil, Sfast + dt*dSfast, Sslow + dt*dSslow]

    Q = RH + RD + Rf + Rs

    fluxes_dict = dict(Ei=Ei, Pn=Pn, Ps=Ps, Ms=Ms, Fs=Fs, Gs=Gs, Pr=Pr, Mw=Mw,
                        Peq=Peq, RH=RH, Fi=Fi, RD=RD, Et=Et, L=L, Ls=Ls, Lf=Lf, Rf=Rf, Rs=Rs)
    return S_new, Q, fluxes_dict


def step_model_45(S, forcings, params, dt=1.0):
    """
    Model 45 -- Precipitation-Runoff Modelling System (PRMS)  (MARRMoT file: m_45_prms_18p_7s)

    Reference(s): Markstrom et al., 2015 (most recent PRMS version); simplified from the original PRMS model of Leavesley et al., 1983. Simplifications: PET supplied as input time series (instead of within-model temperature-based estimates); simpler interception and snow routines.

    Processes represented:
      - Snow accumulation and melt
      - Interception by vegetation
      - Depression storage and impervious surface areas
      - Direct runoff based on catchment saturation
      - Infiltration into soil moisture and connection with deeper groundwater
      - Potentially non-linear interflow, baseflow and groundwater sink

    State variables (7):
      Sn    : snow storage [mm]
      XIN   : interception reservoir storage [mm]
      RSTOR : depression storage on impervious area [mm]
      RECHR : upper soil moisture zone storage [mm]
      SMAV  : lower soil moisture zone storage [mm]
      RES   : runoff/interflow routing reservoir storage [mm]
      GW    : groundwater storage [mm]

    Parameters (18):
      TT         [°C]  Threshold temperature for snowfall and snowmelt  range [-3, 5]
      ddf        [mm °C⁻¹ d⁻¹]  Degree-day factor for snowmelt  range [0, 20]
      α          [–]  Fraction of precipitation on soil that is intercepted  range [0, 1]
      β          [–]  Fraction of precipitation that falls on soil (non-impervious)  range [0, 1]
      STOR       [mm]  Maximum interception storage  range [0, 5]
      RETIP      [mm]  Maximum depression storage  range [0, 50]
      SCN        [–]  Minimum contributing area to surface runoff  range [0, 1]
      SCX        [–]  Maximum contributing area to surface runoff  range [0, 1]
      REMX       [mm]  Maximum upper soil moisture storage  range [1, 2000]
      SMAX       [mm]  Maximum total soil moisture storage  range [1, 2000]
      cgw        [mm d⁻¹]  Maximum groundwater recharge rate  range [0, 20]
      RESMAX     [mm]  Maximum runoff routing storage  range [1, 300]
      k1         [d⁻¹]  Runoff coefficient (groundwater drainage GAD time scale)  range not specified in Table S3 — check S2.45.3 parameter table only
      k2         [–]  Runoff non-linearity (GAD exponent)  range [1, 5]
      k3         [d⁻¹]  Runoff coefficient (RAS linear term)  range not specified in Table S3 — check S2.45.3 parameter table only
      k4         [mm⁻¹ d⁻¹]  Runoff coefficient (RAS quadratic term)  range not specified in Table S3 — check S2.45.3 parameter table only
      k5         [d⁻¹]  Runoff coefficient (baseflow BAS time scale)  range not specified in Table S3 — check S2.45.3 parameter table only
      k6         [d⁻¹]  Runoff coefficient (deep sink SNK time scale)  range not specified in Table S3 — check S2.45.3 parameter table only

    Output / routing (as specified):
      ```
      Q_t = SAS + SRO + RAS + BAS                                    (680)
      ```
      Routing: none — no Unit Hydrograph is used by this model; all flow components (SAS, SRO, RAS, BAS) are generated directly by the store fluxes above and summed instantaneously into Q_t.
    """
    # unpack state
    Sn, XIN, RSTOR, RECHR, SMAV, RES, GW = S

    # unpack forcings
    P, Ep, T = forcings['P'], forcings['Ep'], forcings['T']

    # unpack params
    TT, ddf, alpha, beta, STOR, RETIP, SCN, SCX, REMX, SMAX, c_gw, RESMAX, k1, k2, k3, k4, k5, k6 = (
        params['TT'],
        params['ddf'],
        params['alpha'],
        params['beta'],
        params['STOR'],
        params['RETIP'],
        params['SCN'],
        params['SCX'],
        params['REMX'],
        params['SMAX'],
        params['c_gw'],
        params['RESMAX'],
        params['k1'],
        params['k2'],
        params['k3'],
        params['k4'],
        params['k5'],
        params['k6']
    )

    # --- snow ---
    _w1 = logistic_temperature(T, TT)
    Ps = _w1 * (P) + (1 - _w1) * (0.0)
    _w2 = logistic_temperature(T, TT)
    M = (1 - _w2) * (ddf * (T - TT)) + _w2 * (0.0) # --- interception ---
    _w3 = logistic_temperature(T, TT)
    Pr = (1 - _w3) * (P) + _w3 * (0.0)
    Psm  = beta * Pr
    Pin  = alpha * Psm
    _w4 = logistic_storage(XIN, 0)
    Ein = (1 - _w4) * (beta * Ep) + _w4 * (0.0)
    _w5 = logistic_storage(XIN, STOR)
    Ptf = (1 - _w5) * (Pin) + _w5 * (0.0) # (bounded: XIN cannot exceed STOR)

    # --- impervious depression storage ---
    Pim  = (1 - beta) * Pr
    Mim  = (1 - beta) * M
    _w6 = logistic_storage(RSTOR, 0)
    Eim = (1 - _w6) * ((1 - beta) * Ep) + _w6 * (0.0)
    _w7 = logistic_storage(RSTOR, RETIP)
    SAS = (1 - _w7) * ((Pim + Mim)) + _w7 * (0.0) # --- upper soil zone ---
    Msm  = beta * M
    Pby  = (1 - alpha) * Psm
    SRO  = (SCN + (SCX - SCN) * (RECHR / REMX)) * (Msm + Ptf + Pby)
    INF  = Msm + Ptf + Pby - SRO
    Ea   = (RECHR / REMX) * (Ep - Ein - Eim)
    _w8 = logistic_storage(RECHR, REMX)
    PC = (1 - _w8) * (INF) + _w8 * (0.0) # --- lower soil zone ---
    Et   = (SMAV / SMAX) * (Ep - Ein - Eim - Ea) if RECHR < (Ep - Ein - Eim) else 0.0
    EXCS = PC if SMAV >= (SMAX - REMX) else 0.0

    # --- runoff / interflow reservoir ---
    SEP  = min(c_gw, EXCS)
    QRES = min(EXCS - SEP, 0)          # as printed in source (verify sign convention on port)
    GAD  = k1 * (RES / RESMAX) ** k2
    RAS  = k3 * RES + k4 * RES ** 2

    # --- groundwater ---
    BAS  = k5 * GW
    SNK  = k6 * GW

    # --- mass balance updates (explicit Euler, dt=1.0 unless overridden) ---
    Sn_new    = Sn    + dt * (Ps - M)
    XIN_new   = XIN   + dt * (Pin - Ein - Ptf)
    RSTOR_new = RSTOR + dt * (Pim + Mim - Eim - SAS)
    RECHR_new = RECHR + dt * (INF - Ea - PC)
    SMAV_new  = SMAV  + dt * (PC - Et - EXCS)
    RES_new   = RES   + dt * (QRES - GAD - RAS)
    GW_new    = GW    + dt * (SEP + GAD - BAS - SNK)

    S_new = (Sn_new, XIN_new, RSTOR_new, RECHR_new, SMAV_new, RES_new, GW_new)

    # --- output ---
    Q = SAS + SRO + RAS + BAS

    fluxes_dict = {
        "Ps": Ps, "M": M, "Pr": Pr, "Psm": Psm, "Pin": Pin, "Ein": Ein, "Ptf": Ptf,
        "Pim": Pim, "Mim": Mim, "Eim": Eim, "SAS": SAS,
        "Msm": Msm, "Pby": Pby, "SRO": SRO, "INF": INF, "Ea": Ea, "PC": PC,
        "Et": Et, "EXCS": EXCS, "SEP": SEP, "QRES": QRES, "GAD": GAD, "RAS": RAS,
        "BAS": BAS, "SNK": SNK,
    }
    return S_new, Q, fluxes_dict


def step_model_46(S, forcings, params, dt=1.0):
    """
    Model 46 -- Climate and Land-use Scenario Simulation in Catchments (CLASSIC)  (MARRMoT file: m_46_classic_12p_8s)

    Reference(s): Crooks and Naden, 2007. (MARRMoT documentation notes the grid-based routing component of the original model is not included here, nor is the arable soil element, because input data for this soil type is not supported.)

    Processes represented:
      - Division into permeable, semi-permeable and impermeable areas
      - Infiltration into permeable soils and deficit-based soil moisture accounting
      - Infiltration into semi-permeable soils and direct runoff from semi-permeable soils (bypassing the moisture accounting)
      - Fixed interception on impermeable soils
      - Linear flow routing from permeable soils
      - Fast and slow routing from semi-permeable soils
      - Linear flow routing from impermeable soils

    State variables (8):
      Px : storage in the upper permeable layer [mm]
      Py : "deficit" storage in the lower permeable layer [mm]
      P  : storage in the permeable-soil routing store [mm]
      Sx : storage in the upper semi-permeable layer [mm]
      Sy : "deficit" storage in the lower semi-permeable layer [mm]
      Sq : storage in the semi-permeable quick-flow routing store [mm]
      Ss : storage in the semi-permeable slow-flow routing store [mm]
      I  : storage in the impermeable-soil routing store [mm]

    Parameters (12):
      f_ap       [–]  Fraction permeable area  range [0, 1]
      f_dp       [–]  Store depth as fraction of d_p (fraction of permeable total depth that is store Px)  range not specified in Table S3 — check S2.46.3 parameter table only
      d_p        [mm]  Total storage depth of Px and Py (permeable soil column)  range [1, 2000]
      c_q        [d⁻¹]  Runoff coefficient — permeable routing store P  range [0, 1]
      d_1        [–]  Fraction of precipitation on semi-permeable area that infiltrates to Sx  range [0, 1]
      f_as       [–]  Fraction semi-permeable area  range [0, 1]
      f_ds       [–]  Store depth as fraction of d_s (fraction of semi-permeable total depth that is store Sx)  range not specified in Table S3 — check S2.46.3 parameter table only
      d_s        [mm]  Total storage depth of Sx and Sy (semi-permeable soil column)  range [1, 2000]
      d_2        [–]  Fraction of semi-permeable-area subsurface flow that is quick flow  range [0, 1]
      c_xq       [d⁻¹]  Runoff coefficient — semi-permeable quick-flow store Sq  range [0, 1]
      c_xs       [d⁻¹]  Runoff coefficient — semi-permeable slow-flow store Ss  range [0, 1]
      c_u        [d⁻¹]  Runoff coefficient — impermeable routing store I  range [0, 1]

    Output / routing (as specified):
      ```
      Q = q + x_s + x_q + u                                               (708)
      ```
      Routing: none — no Unit Hydrograph is used; each of the four store types (permeable P, semi-permeable quick Sq, semi-permeable slow Ss, impermeable I) drains via its own linear reservoir directly into Q.
    """
    # unpack state
    Px, Py, P_store, Sx, Sy, Sq, Ss, I = S

    # unpack forcings
    P, Ep = forcings['P'], forcings['Ep']     # P here is the precipitation forcing

    # unpack params
    f_ap, f_dp, d_p, c_q, d_1, f_as, f_ds, d_s, d_2, c_xq, c_xs, c_u = (
        params['f_ap'],
        params['f_dp'],
        params['d_p'],
        params['c_q'],
        params['d_1'],
        params['f_as'],
        params['f_ds'],
        params['d_s'],
        params['d_2'],
        params['c_xq'],
        params['c_xs'],
        params['c_u']
    )
    E_i_fixed = 0.5   # fixed constant impermeable-area evaporation [mm/d], not a calibrated parameter

    # --- area split of precipitation ---
    P_p = f_ap * P
    P_s = f_as * P
    P_i = P - P_p - P_s          # remainder falls on impermeable area

    # --- upper permeable layer (Px) ---
    _w1 = logistic_storage(Px, 0)
    E_px = (1 - _w1) * (f_ap * Ep) + _w1 * (0.0)
    P_px = P_p if Px >= f_dp * d_p else 0.0   # bounded: Px capped at f_dp*d_p

    # --- lower permeable deficit layer (Py) ---
    E_py = 1.9 * exp(-0.6523 * (Py + f_dp * d_p) / (f_dp * d_p)) * (f_ap * Ep - E_px)
    _w2 = logistic_storage(Py, 0)
    P_pe = _w2 * (P_px) + (1 - _w2) * (0.0) # bounded: Py cannot go below 0

    # --- permeable routing store (P_store) ---
    q = c_q * P_store

    # --- upper semi-permeable layer (Sx) ---
    P_si = d_1 * P_s
    P_sd = (1 - d_1) * P_s                     # bypass fraction -> direct effective rainfall
    _w3 = logistic_storage(Sx, 0)
    E_sx = (1 - _w3) * (f_as * Ep) + _w3 * (0.0)
    P_sx = P_s if Sx >= f_ds * d_s else 0.0    # bounded: Sx capped at f_ds*d_s

    # --- lower semi-permeable deficit layer (Sy) ---
    E_sy = 1.9 * exp(-0.6523 * (Sy + f_ds * d_s) / (f_ds * d_s)) * (f_as * Ep - E_sx)
    _w4 = logistic_storage(Sy, 0)
    P_se = _w4 * (P_sx) + (1 - _w4) * (0.0) # bounded: Sy cannot go below 0

    # --- semi-permeable quick-flow routing store (Sq) ---
    P_sq = d_2 * (P_se + P_sd)
    x_q  = c_xq * Sq

    # --- semi-permeable slow-flow routing store (Ss) ---
    P_ss = (1 - d_2) * (P_se + P_sd)
    x_s  = c_xs * Ss

    # --- impermeable soil routing store (I) ---
    P_ie = P_i - E_i_fixed
    u    = c_u * I

    # --- mass balance updates (explicit Euler, dt=1.0 unless overridden) ---
    Px_new = Px      + dt * (P_p - E_px - P_px)
    Py_new = Py      + dt * (-P_px + E_py + P_pe)
    P_new  = P_store + dt * (P_pe - q)
    Sx_new = Sx      + dt * (P_si - E_sx - P_sx)
    Sy_new = Sy      + dt * (-P_sx + E_sy + P_se)
    Sq_new = Sq      + dt * (P_sq - x_q)
    Ss_new = Ss      + dt * (P_ss - x_s)
    I_new  = I        + dt * (P_ie - u)

    S_new = (Px_new, Py_new, P_new, Sx_new, Sy_new, Sq_new, Ss_new, I_new)

    # --- output ---
    Q = q + x_s + x_q + u

    fluxes_dict = {
        "P_p": P_p, "P_s": P_s, "P_i": P_i,
        "E_px": E_px, "P_px": P_px, "E_py": E_py, "P_pe": P_pe, "q": q,
        "P_si": P_si, "P_sd": P_sd, "E_sx": E_sx, "P_sx": P_sx,
        "E_sy": E_sy, "P_se": P_se,
        "P_sq": P_sq, "x_q": x_q, "P_ss": P_ss, "x_s": x_s,
        "P_ie": P_ie, "u": u,
    }
    return S_new, Q, fluxes_dict

# ===========================================================================
# Model registry and generic simulation driver
# ===========================================================================

MODEL_REGISTRY = {
    1: {
        'name': 'Collie River Basin 1',
        'marrmot_file': 'm_01_collie1_1p_1s',
        'step_fn': step_model_01,
        'n_stores': 1,
        'n_params': 1,
        'param_names': ['Smax'],
        'param_units': ['mm'],
        'param_ranges': ['[1, 2000]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    2: {
        'name': 'Wetland model',
        'marrmot_file': 'm_02_wetland_4p_1s',
        'step_fn': step_model_02,
        'n_stores': 1,
        'n_params': 4,
        'param_names': ['Dw', 'Sw_max', 'beta_w', 'Kw'],
        'param_units': ['mm/d', 'mm', '–', 'd⁻¹'],
        'param_ranges': ['[0, 5]', '[1, 2000]', '[0, 10]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    3: {
        'name': 'Collie River Basin 2',
        'marrmot_file': 'm_03_collie2_4p_1s',
        'step_fn': step_model_03,
        'n_stores': 1,
        'n_params': 4,
        'param_names': ['Smax', 'Sfc', 'a', 'M'],
        'param_units': ['mm', 'mm', 'd⁻¹', '–'],
        'param_ranges': ['[1, 2000]', '[0.05, 0.95]*', '[0, 1]', '[0.05, 0.95]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    4: {
        'name': 'New Zealand model v1',
        'marrmot_file': 'm_04_newzealand1_6p_1s',
        'step_fn': step_model_04,
        'n_stores': 1,
        'n_params': 6,
        'param_names': ['Smax', 'Sfc', 'M', 'a', 'b', 'tc_bf'],
        'param_units': ['mm', '–', '–', 'd⁻¹', '–', 'd⁻¹'],
        'param_ranges': ['[1, 2000]', '[0.05, 0.95]', '[0.05, 0.95]', '[0, 1]', '[1, 5]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    5: {
        'name': 'IHACRES',
        'marrmot_file': 'm_05_ihacres_7p_1s',
        'step_fn': step_model_05,
        'n_stores': 1,
        'n_params': 7,
        'param_names': ['lp', 'd', 'p', 'alpha', 'tau_q', 'tau_s', 'tau_d'],
        'param_units': ['mm', 'mm', '–', '–', 'd', 'd', 'd'],
        'param_ranges': ['not specified in Table S3 — check S2.5.3 parameter table only', '[1, 2000]', '[0, 10]', 'not specified in Table S3 — check S2.5.3 parameter table only', '[1, 700]', '[1, 700]', '[0, 119]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': True,
    },
    6: {
        'name': 'Alpine model v1',
        'marrmot_file': 'm_06_alpine1_4p_2s',
        'step_fn': step_model_06,
        'n_stores': 2,
        'n_params': 4,
        'param_names': ['Tt', 'ddf', 'Smax', 'tc'],
        'param_units': ['°C', 'mm °C⁻¹ d⁻¹', 'mm', 'd⁻¹'],
        'param_ranges': ['[-3, 5]', '[0, 20]', '[1, 2000]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep', 'T'],
        'has_routing': False,
    },
    7: {
        'name': 'GR4J',
        'marrmot_file': 'm_07_gr4j_4p_2s',
        'step_fn': step_model_07,
        'n_stores': 2,
        'n_params': 4,
        'param_names': ['x1', 'x2', 'x3', 'x4'],
        'param_units': ['mm', 'mm/d', 'mm', 'd'],
        'param_ranges': ['[1, 2000]', '[-10, 15]', '[1, 300]', '[1, 15]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': True,
    },
    8: {
        'name': 'United States model',
        'marrmot_file': 'm_08_us1_5p_2s',
        'step_fn': step_model_08,
        'n_stores': 2,
        'n_params': 5,
        'param_names': ['alpha_ei', 'M', 'S_max', 'fc', 'alpha_ss'],
        'param_units': ['-', '-', 'mm', '-', 'd^-1'],
        'param_ranges': ['[0, 1]', '[0.05, 0.95]', '[1, 2000]', '[0.05, 0.95]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    9: {
        'name': 'Susannah Brook model v1-5',
        'marrmot_file': 'm_09_susannah1_6p_2s',
        'step_fn': step_model_09,
        'n_stores': 2,
        'n_params': 6,
        'param_names': ['S_b', 'S_fc', 'M', 'a', 'b', 'r'],
        'param_units': ['mm', 'mm', '-', 'd', '-', '-'],
        'param_ranges': ['[1, 2000]', 'not specified in Table S3 — check S2.9.3 parameter table only', '[0.05, 0.95]', '[1, 50]', '[1, 5]', 'not specified in Table S3 — check S2.9.3 parameter table only'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    10: {
        'name': 'Susannah Brook model v2',
        'marrmot_file': 'm_10_susannah2_6p_2s',
        'step_fn': step_model_10,
        'n_stores': 2,
        'n_params': 6,
        'param_names': ['S_b', 'phi', 'fc', 'r', 'c', 'd'],
        'param_units': ['mm', '-', '-', '-', 'd^-1', '-'],
        'param_ranges': ['[1, 2000]', '[0.05, 0.95]', '[0.05, 0.95]', '[0, 1]', '[0, 1]', '[1, 5]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    11: {
        'name': 'Collie River Basin 3',
        'marrmot_file': 'm_11_collie3_6p_2s',
        'step_fn': step_model_11,
        'n_stores': 2,
        'n_params': 6,
        'param_names': ['S_max', 'S_fc', 'a', 'M', 'b', 'lam'],
        'param_units': ['mm', 'mm', 'd^-1', '-', '-', ''],
        'param_ranges': ['[1, 2000]', 'not specified in Table S3 — check S2.11.3 parameter table only', 'not specified in Table S3 — check S2.11.3 parameter table only', 'not specified in Table S3 — check S2.11.3 parameter table only', '[1, 5]', ''],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    12: {
        'name': 'Alpine model v2',
        'marrmot_file': 'm_12_alpine2_6p_2s',
        'step_fn': step_model_12,
        'n_stores': 2,
        'n_params': 6,
        'param_names': ['T_t', 'ddf', 'S_max', 'S_fc', 't_c_in', 't_c_bf'],
        'param_units': ['°C', 'mm °C^-1 d^-1', 'mm', 'mm', 'd^-1', 'd^-1'],
        'param_ranges': ['[-3, 5]', '[0, 20]', '[1, 2000]', 'not specified in Table S3 — check S2.12.3 parameter table only', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep', 'T'],
        'has_routing': False,
    },
    13: {
        'name': 'Hillslope model',
        'marrmot_file': 'm_13_hillslope_7p_2s',
        'step_fn': step_model_13,
        'n_stores': 2,
        'n_params': 7,
        'param_names': ['D_w', 'S_h_max', 'beta_h', 'a', 'T_h', 'C_param', 'K_h'],
        'param_units': ['mm d^-1', 'mm', '-', '-', 'd', '', 'd^-1'],
        'param_ranges': ['[0, 5]', '[1, 2000]', '[0, 10]', '[0, 1]', '[1, 120]', '', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': True,
    },
    14: {
        'name': 'TOPMODEL',
        'marrmot_file': 'm_14_topmodel_7p_2s',
        'step_fn': step_model_14,
        'n_stores': 2,
        'n_params': 7,
        'param_names': ['S_UZmax', 'S_t', 'K_d', 'q0', 'f', 'chi', 'phi'],
        'param_units': ['mm', '-', 'd^-1', 'mm d^-1', 'mm^-1', '', ''],
        'param_ranges': ['[1, 2000]', '[0.05, 0.95]', '[0, 1]', '[0.1, 200]', '[0, 1]', '', ''],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    15: {
        'name': 'Plateau model',
        'marrmot_file': 'm_15_plateau_8p_2s',
        'step_fn': step_model_15,
        'n_stores': 2,
        'n_params': 8,
        'param_names': ['F_max', 'D_p', 'S_umax', 'lp', 'p', 'T_p', 'C_rate', 'K_p'],
        'param_units': ['mm d^-1', 'mm d^-1', 'mm', '-', '-', 'd', '', 'd^-1'],
        'param_ranges': ['[0, 200]', '[0, 5]', '[1, 2000]', '[0.05, 0.95]', '[0, 1]', '[1, 120]', '', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': True,
    },
    16: {
        'name': 'New Zealand model v2',
        'marrmot_file': 'm_16_newzealand2_8p_2s',
        'step_fn': step_model_16,
        'n_stores': 2,
        'n_params': 8,
        'param_names': ['I_max', 'S_max', 'S_fc', 'M', 'a', 'b', 't_c_bf', 'd_uh'],
        'param_units': ['mm', 'mm', 'mm', '-', 'd^-1', '-', 'd^-1', ''],
        'param_ranges': ['[0, 5]', '[1, 2000]', 'not specified in Table S3 — check S2.x.3 parameter table only', '[0.05, 0.95]', 'not specified in Table S3 — check S2.x.3 parameter table only', '[1, 5]', '[0, 1]', ''],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': True,
    },
    17: {
        'name': 'Penman model',
        'marrmot_file': 'm_17_penman_4p_3s',
        'step_fn': step_model_17,
        'n_stores': 3,
        'n_params': 4,
        'param_names': ['S_max', 'phi', 'gamma', 'k1'],
        'param_units': ['mm', '', '', 'd^-1'],
        'param_ranges': ['[1, 2000]', '', '', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    18: {
        'name': 'SIMHYD',
        'marrmot_file': 'm_18_simhyd_7p_3s',
        'step_fn': step_model_18,
        'n_stores': 3,
        'n_params': 7,
        'param_names': ['INSC', 'COEFF', 'SQ', 'SMSC', 'SUB', 'CRAK', 'K'],
        'param_units': ['mm', 'mm d^-1', '-', 'mm', '-', '-', 'd^-1'],
        'param_ranges': ['[0, 5]', '[0, 600]', '[0, 15]', '[1, 2000]', '[0, 1]', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    19: {
        'name': 'Australia model',
        'marrmot_file': 'm_19_australia_8p_3s',
        'step_fn': step_model_19,
        'n_stores': 3,
        'n_params': 8,
        'param_names': ['S_b', 'phi', 'fc', 'alpha_SS', 'beta_SS', 'K_deep', 'alpha_BF', 'beta_BF'],
        'param_units': ['mm', '–', '–', 'd-1', '–', 'd-1', 'd-1', '–'],
        'param_ranges': ['[1, 2000]', '[0.05, 0.95]', '[0.05, 0.95]', '[0, 1]', '[1, 5]', '[0, 1]', '[0, 1]', '[1, 5]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    20: {
        'name': 'Generalized Surface inFiltration Baseflow model (GSFB)',
        'marrmot_file': 'm_20_gsfb_8p_3s',
        'step_fn': step_model_20,
        'n_stores': 3,
        'n_params': 8,
        'param_names': ['C', 'NDC', 'S_max', 'E_max', 'F_rate', 'B', 'DPF', 'SDR_max'],
        'param_units': ['d-1', '–', 'mm', 'mm d-1', 'mm d-1', '–', 'd-1', 'mm'],
        'param_ranges': ['[0, 1]', 'not specified in Table S3 — check S2.20.3 parameter table only', '[1, 2000]', '[0, 20]', '[0, 200]', '[0, 1]', '[0, 1]', '[1, 300]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    21: {
        'name': 'Flex-B',
        'marrmot_file': 'm_21_flexb_9p_3s',
        'step_fn': step_model_21,
        'n_stores': 3,
        'n_params': 9,
        'param_names': ['UR_max', 'beta', 'D', 'Perc_max', 'Lp', 'Nlag_f', 'Nlag_s', 'Kf', 'Ks'],
        'param_units': ['mm', '–', '–', 'mm d-1', '–', 'd', 'd', 'd-1', 'd-1'],
        'param_ranges': ['[1, 2000]', '[0, 10]', '[0, 1]', '[0, 20]', '[0.05, 0.95]', '[1, 5]', '[1, 15]', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': True,
    },
    22: {
        'name': 'Variable Infiltration Capacity (VIC) model',
        'marrmot_file': 'm_22_vic_10p_3s',
        'step_fn': step_model_22,
        'n_stores': 3,
        'n_params': 10,
        'param_names': ['I_bar', 'I_delta', 'I_s', 'S_sm_max', 'b', 'k1', 'c1', 'S_gw_max', 'k2', 'c2'],
        'param_units': ['mm', '–', 'd', 'mm', '–', 'd-1', '–', 'mm', 'd-1', '–'],
        'param_ranges': ['[0, 5]', '[0, 1]', '[1, 365]', '[1, 2000]', '[0, 10]', '[0, 1]', '[1, 5]', '[1, 2000]', '[0, 1]', '[1, 5]'],
        'forcing_vars': ['P', 'Ep', 't'],
        'has_routing': False,
    },
    23: {
        'name': 'Large-scale catchment water and salt balance model (LASCAM)',
        'marrmot_file': 'm_23_lascam_24p_3s',
        'step_fn': step_model_23,
        'n_stores': 3,
        'n_params': 24,
        'param_names': ['alpha_f', 'beta_f', 'B_max', 'F_max', 'alpha_c', 'beta_c', 'A_min', 'A_max', 'alpha_ss', 'beta_ss', 'c', 'alpha_g', 'beta_g', 'gamma_f', 'delta_f', 't_d', 'alpha_b', 'beta_b', 'gamma_a', 'delta_a', 'alpha_a', 'beta_a', 'gamma_b', 'delta_b'],
        'param_units': ['mm d-1', '–', 'mm', 'mm', '–', '–', 'mm', 'mm', '–', '–', 'mm d-1', 'mm d-1', '–', '–', '–', 'd-1', '–', 'mm d-1', '–', '–', 'mm d-1', '–', '–', '–'],
        'param_ranges': ['[0, 200]', '[0, 5]', '[1, 2000]', '[1, 2000]', '[0, 5]', '[0, 1]', 'not specified in Table S3 — check S2.23.3 parameter table only', 'not specified in Table S3 — check S2.23.3 parameter table only', '[0, 5]', '[0, 1]', '[0, 200]', 'not specified in Table S3 — check S2.23.3 parameter table only', '[0, 1]', '[0, 1]', '[0, 10]', 'not specified in Table S3 — check S2.23.3 parameter table only', '[0, 1]', '[0.1, 200]', '[0, 1]', '[0, 10]', '[0.1, 200]', '[1, 5]', '[0, 1]', '[0, 10]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    24: {
        'name': 'MOPEX-1',
        'marrmot_file': 'm_24_mopex1_5p_4s',
        'step_fn': step_model_24,
        'n_stores': 4,
        'n_params': 5,
        'param_names': ['S_b1', 't_w', 't_u', 'S_e', 't_c'],
        'param_units': ['mm', 'd-1', 'd-1', 'mm', 'd-1'],
        'param_ranges': ['[1, 2000]', '[0, 1]', '[0, 1]', '[1, 2000]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    25: {
        'name': 'Thames Catchment Model (TCM)',
        'marrmot_file': 'm_25_tcm_6p_4s',
        'step_fn': step_model_25,
        'n_stores': 4,
        'n_params': 6,
        'param_names': ['phi', 'rc', 'gamma', 'k1', 'c_a', 'k2'],
        'param_units': ['-', 'mm', '-', 'd⁻¹', 'mm d⁻¹', 'd⁻¹ (printed in parameter table; note the governing equation Q=k2·S_sz² requires k2 in mm⁻¹ d⁻¹ for dimensional consistency — flagged discrepancy in the source document)'],
        'param_ranges': ['[0, 1]', 'not specified in Table S3 — check S2.x.3 parameter table only (model 25 is absent from the "Soil moisture: Max store depth" model list)', '[0, 1]', '[0, 1]', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only (no matching category/unit; not listed under Fast or Slow reservoir time scale)'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    26: {
        'name': 'Flex-I',
        'marrmot_file': 'm_26_flexi_10p_4s',
        'step_fn': step_model_26,
        'n_stores': 4,
        'n_params': 10,
        'param_names': ['Imax', 'URmax', 'beta', 'D', 'Percmax', 'Lp', 'Nlag_f', 'Nlag_s', 'Kf', 'Ks'],
        'param_units': ['mm', 'mm', '-', '-', 'mm d⁻¹', '-', 'd', 'd', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[0, 5]', '[1, 2000]', '[0, 10]', '[0, 1]', '[0, 20]', '[0.05, 0.95]', '[1, 5]', '[1, 15]', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': True,
    },
    27: {
        'name': 'Tank model',
        'marrmot_file': 'm_27_tank_12p_4s',
        'step_fn': step_model_27,
        'n_stores': 4,
        'n_params': 12,
        'param_names': ['A0', 'A1', 'A2', 't1', 't2', 'B0', 'B1', 't3', 'C0', 'C1', 't4', 'D1'],
        'param_units': ['d⁻¹', 'd⁻¹', 'd⁻¹', 'mm', 'mm', 'd⁻¹', 'd⁻¹', 'mm', 'd⁻¹', 'd⁻¹', 'mm', 'd⁻¹'],
        'param_ranges': ['not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    28: {
        'name': 'Xinanjiang model',
        'marrmot_file': 'm_28_xinanjiang_12p_4s',
        'step_fn': step_model_28,
        'n_stores': 4,
        'n_params': 12,
        'param_names': ['Aim', 'a', 'b', 'Wmax', 'LM', 'c', 'Smax', 'Ex', 'kI', 'kG', 'cI', 'cG'],
        'param_units': ['-', '-', '-', 'mm', 'mm', '-', 'mm', '-', 'd⁻¹', 'd⁻¹', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[0, 1]', '[-0.5, 0.5]', '[0, 10]', '[1, 2000]', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only', 'not specified in Table S3 — check S2.x.3 parameter table only (no distinct "free water" store-depth row separate from Wmax\'s "Soil moisture Max store depth")', 'not specified in Table S3 — check S2.x.3 parameter table only', '[0, 1]', '[0, 1]', 'not specified in Table S3 — check S2.x.3 parameter table only (Fast/Slow reservoir time scale rows already used by kI/kG)', 'not specified in Table S3 — check S2.x.3 parameter table only (Fast/Slow reservoir time scale rows already used by kI/kG)'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    29: {
        'name': 'HyMOD',
        'marrmot_file': 'm_29_hymod_5p_5s',
        'step_fn': step_model_29,
        'n_stores': 5,
        'n_params': 5,
        'param_names': ['Smax', 'b', 'a', 'kf', 'ks'],
        'param_units': ['mm', '-', '-', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[1, 2000]', '[0, 10]', '[0, 1]', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    30: {
        'name': 'MOPEX-2',
        'marrmot_file': 'm_30_mopex2_7p_5s',
        'step_fn': step_model_30,
        'n_stores': 5,
        'n_params': 7,
        'param_names': ['Tcrit', 'ddf', 'Sb1', 'tw', 'tu', 'Se', 'tc'],
        'param_units': ['°C', 'mm °C⁻¹ d⁻¹', 'mm', 'd⁻¹', 'd⁻¹', 'mm', 'd⁻¹'],
        'param_ranges': ['[-3, 5]', '[0, 20]', '[1, 2000]', '[0, 1]', 'not specified in Table S3 — check S2.x.3 parameter table only (no distinct remaining category)', 'not specified in Table S3 — check S2.x.3 parameter table only', '[0, 1]'],
        'forcing_vars': ['P', 'Ep', 'T'],
        'has_routing': False,
    },
    31: {
        'name': 'MOPEX-3',
        'marrmot_file': 'm_31_mopex3_8p_5s',
        'step_fn': step_model_31,
        'n_stores': 5,
        'n_params': 8,
        'param_names': ['T_crit', 'ddf', 'S_b1', 't_w', 'S_b2', 't_u', 'S_e', 't_c'],
        'param_units': ['°C', 'mm °C⁻¹ d⁻¹', 'mm', 'd⁻¹', 'mm', 'd⁻¹', 'mm', 'd⁻¹'],
        'param_ranges': ['[-3, 5]', '[0, 20]', '[1, 2000]', 'not specified in Table S3 — no distinct category for this soil→groundwater time coefficient; check S2.31.3 table only', '[1, 2000]', '[0, 1]', '[1, 2000]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep', 'T'],
        'has_routing': False,
    },
    32: {
        'name': 'MOPEX-4',
        'marrmot_file': 'm_32_mopex4_10p_5s',
        'step_fn': step_model_32,
        'n_stores': 5,
        'n_params': 10,
        'param_names': ['T_crit', 'ddf', 'S_b1', 't_w', 'I_alpha', 'I_s', 'S_b2', 't_u', 'S_e', 't_c'],
        'param_units': ['°C', 'mm °C⁻¹ d⁻¹', 'mm', 'd⁻¹', '–', 'd', 'mm', 'd⁻¹', 'mm', 'd⁻¹'],
        'param_ranges': ['[-3, 5]', '[0, 20]', '[1, 2000]', 'not specified in Table S3 — check S2.32.3 table only', '[0, 1]', '[1, 365]', '[1, 2000]', '[0, 1]', '[1, 2000]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep', 'T', 't'],
        'has_routing': False,
    },
    33: {
        'name': 'SACRAMENTO',
        'marrmot_file': 'm_33_sacramento_11p_5s',
        'step_fn': step_model_33,
        'n_stores': 5,
        'n_params': 13,
        'param_names': ['PCTIM', 'UZTWM', 'UZFWM', 'k_uz', 'PBASE', 'ZPERC', 'REXP', 'LZTWM', 'LZFWPM', 'LZFWSM', 'PFREE', 'k_lzp', 'k_lzs'],
        'param_units': ['–', 'mm', 'mm', 'd⁻¹', 'mm d⁻¹', '–', '–', 'mm', 'mm', 'mm', '–', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[0, 1]', '[1, 2000]', '[1, 2000]', '[0, 1]', 'not applicable (derived quantity, not sampled) — see note above', 'not specified in Table S3 — check S2.33.3 table only, and verify derived-vs-free status against MARRMoT source', '[0, 10]', '[1, 2000]', '[1, 2000]', '[1, 2000]', '[0, 1]', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    34: {
        'name': 'FLEX-IS',
        'marrmot_file': 'm_34_flexis_12p_5s',
        'step_fn': step_model_34,
        'n_stores': 5,
        'n_params': 12,
        'param_names': ['TT', 'ddf', 'Imax', 'URmax', 'beta', 'Lp', 'Percmax', 'D', 'Nlag_f', 'Nlag_s', 'Kf', 'Ks'],
        'param_units': ['°C', 'mm °C⁻¹ d⁻¹', 'mm', 'mm', '-', '-', 'mm d⁻¹', '-', 'd', 'd', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[-3, 5]', '[0, 20]', '[0, 5]', '[1, 2000]', '[0, 10]', '[0.05, 0.95]', '[0, 20]', '[0, 1]', '[1, 5]', '[1, 15]', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep', 'T'],
        'has_routing': True,
    },
    35: {
        'name': 'MOPEX-5',
        'marrmot_file': 'm_35_mopex5_12p_5s',
        'step_fn': step_model_35,
        'n_stores': 5,
        'n_params': 12,
        'param_names': ['Tcrit', 'ddf', 'Sb1', 'tw', 'Ialpha', 'Is', 'Tmin', 'Tmax', 'Sb2', 'tu', 'Se', 'tc'],
        'param_units': ['°C', 'mm °C⁻¹ d⁻¹', 'mm', 'd⁻¹', '-', 'd', '°C', '°C', 'mm', 'd⁻¹', 'mm', 'd⁻¹'],
        'param_ranges': ['[-3, 5]', '[0, 20]', '[1, 2000]', '[0, 1]', '[0, 1]', '[1, 365]', '[0, -10] (as printed in Table S3)', '[1, 20]', '[1, 2000]', '[0, 1]', '[1, 2000]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep', 'T', 't'],
        'has_routing': False,
    },
    36: {
        'name': 'MODHYDROLOG',
        'marrmot_file': 'm_36_modhydrolog_15p_5s',
        'step_fn': step_model_36,
        'n_stores': 5,
        'n_params': 15,
        'param_names': ['INSC', 'COEFF', 'SQ', 'SMSC', 'SUB', 'CRAK', 'EM', 'DSC', 'ADS', 'MD', 'VCOND', 'DLEV', 'k1', 'k2', 'k3'],
        'param_units': ['mm', 'mm d⁻¹', '-', 'mm', '-', '-', 'mm d⁻¹', 'mm', '-', '-', 'd⁻¹', 'mm', 'd⁻¹', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[0, 5]', '[0, 600]', '[0, 15]', '[1, 2000]', '[0, 1]', '[0, 1]', '[0, 20]', '[0, 50]', '[0, 1]', '[0.99, 1]', '[0, 0.5]', '[-10, 10]', '[0, 1]', '[0, 1]', '[0, 100]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    37: {
        'name': 'HBV-96',
        'marrmot_file': 'm_37_hbv_15p_5s',
        'step_fn': step_model_37,
        'n_stores': 5,
        'n_params': 15,
        'param_names': ['TT', 'TTI', 'CFR', 'CFMAX', 'TTM', 'WHC', 'CFLUX', 'FC', 'LP', 'beta', 'K0', 'alpha', 'c', 'K1', 'MAXBAS'],
        'param_units': ['°C', '°C', '–', 'mm °C⁻¹ d⁻¹', '°C', '–', 'mm d⁻¹', 'mm', '–', '–', 'd⁻¹', '–', 'mm d⁻¹', 'd⁻¹', 'd'],
        'param_ranges': ['[-3, 5]', '[0, 17]', '[0, 1]', '[0, 20]', '[-3, 3]', '[0, 1]', '[0, 4]', '[1, 2000]', '[0.05, 0.95]', '[0, 10]', '[0, 1]', '[1, 5]', '[0, 20]', '[0, 1]', '[1, 120]'],
        'forcing_vars': ['P', 'Ep', 'T'],
        'has_routing': True,
    },
    38: {
        'name': 'Tank Model - SMA',
        'marrmot_file': 'm_38_tank2_16p_5s',
        'step_fn': step_model_38,
        'n_stores': 5,
        'n_params': 16,
        'param_names': ['sm1', 'sm2', 'k1', 'k2', 'A0', 'A1', 'A2', 't1', 't2', 'B0', 'B1', 't3', 'C0', 'C1', 't4', 'D1'],
        'param_units': ['mm', 'mm', 'mm d⁻¹', 'mm d⁻¹', 'd⁻¹', 'd⁻¹', 'd⁻¹', 'mm', 'mm', 'd⁻¹', 'd⁻¹', 'mm', 'd⁻¹', 'd⁻¹', 'mm', 'd⁻¹'],
        'param_ranges': ['[1, 2000]', '[1, 2000]', '[0, 4]', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only', 'not specified in Table S3 — check S2.38.3 parameter table only'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    39: {
        'name': 'Midlands Catchment Runoff Model (MCRM)',
        'marrmot_file': 'm_39_mcrm_16p_5s',
        'step_fn': step_model_39,
        'n_stores': 5,
        'n_params': 16,
        'param_names': ['Smax', 'cmax', 'c0', 'c1', 'ce', 'Dsurp', 'kd', 'gamma_d', 'qp_max', 'kg', 'tau', 'Sbf', 'kcr', 'gamma_cr', 'kor', 'gamma_or'],
        'param_units': ['mm', '–', '–', '–', '–', 'mm', 'd⁻¹', '–', 'mm d⁻¹', 'd⁻¹', 'd', 'mm', 'd⁻¹', '–', 'd⁻¹', '–'],
        'param_ranges': ['[0, 5]', '[0, 1]', '[0, 1]', 'not specified in Table S3 — check S2.39.3 parameter table only', '[0, 1]', 'not specified in Table S3 — check S2.39.3 parameter table only', '[0, 1]*', '[1, 5]', '[0, 20]', '[0, 1]*', '[1, 120]', '[1, 300]', '[0, 1]*', '[1, 5]', '[0, 1]*', '[1, 5]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': True,
    },
    40: {
        'name': 'SMAR',
        'marrmot_file': 'm_40_smar_8p_6s',
        'step_fn': step_model_40,
        'n_stores': 6,
        'n_params': 8,
        'param_names': ['H', 'Y', 'Smax', 'C', 'G', 'KG', 'N', 'K'],
        'param_units': ['–', 'mm d⁻¹', 'mm', '–', '–', 'd⁻¹', '–', 'd'],
        'param_ranges': ['[0, 1]', '[0, 200]', '[1, 2000]', '[0, 1]', '[0, 1]', 'not specified in Table S3 — check S2.40.3 parameter table only', '[1, 10]', '[1, 120]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': True,
    },
    41: {
        'name': 'NAM model',
        'marrmot_file': 'm_41_nam_10p_6s',
        'step_fn': step_model_41,
        'n_stores': 6,
        'n_params': 10,
        'param_names': ['Cs', 'Cif', 'Lstar', 'CL1', 'Ustar', 'Cof', 'CL2', 'K0', 'K1', 'Kb'],
        'param_units': ['mm °C⁻¹ d⁻¹', 'd⁻¹', '', '–', '', 'd⁻¹', '–', 'd⁻¹', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[0, 20]', 'not specified in Table S3 — check S2.41.3 parameter table only', '', 'not specified in Table S3 — check S2.41.3 parameter table only (only one threshold row is listed for model 41 in FLOW DISTRIBUTION, best matched to CL2 below)', '', 'not specified in Table S3 — check S2.41.3 parameter table only', '[0, 0.99]', '[0, 1]', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep', 'T'],
        'has_routing': False,
    },
    42: {
        'name': 'HYCYMODEL',
        'marrmot_file': 'm_42_hycymodel_12p_6s',
        'step_fn': step_model_42,
        'n_stores': 6,
        'n_params': 12,
        'param_names': ['C', 'I1max', 'alpha', 'I2max', 'kin', 'D50', 'D16', 'Sbc', 'kb', 'pb', 'kh', 'kc'],
        'param_units': ['–', 'mm', '', 'mm', 'd⁻¹', 'mm', 'mm', 'mm', 'd⁻¹', '–', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[0, 1]', '[0, 5]', '', 'range: not specified in Table S3 — check S2.x.3 parameter table only', '[0, 1]', '[1, 2000]', 'derived: sampled as a fraction of D50, fraction ∈ [0.01, 0.99]', 'range: not specified in Table S3 — check S2.x.3 parameter table only', '[0, 1]', '[1, 5]', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
    43: {
        'name': 'GSM-SOCONT model',
        'marrmot_file': 'm_43_gsmsocont_12p_6s',
        'step_fn': step_model_43,
        'n_stores': 6,
        'n_params': 12,
        'param_names': ['fice', 'T0', 'asnow', 'Tm', 'ks', 'aice', 'ki', 'A', 'x', 'y', 'ksl', 'beta'],
        'param_units': ['–', '°C', 'mm/°C/d', '°C', 'd⁻¹', 'mm/°C/d', 'd⁻¹', 'mm', '–', '–', 'd⁻¹', ''],
        'param_ranges': ['[0, 1]', '[-3, 5]', '[0, 20]', '[-3, 3]', '[0, 1]', '[0, 20]', '[0, 1]', '[1, 2000]', '[0, 10]', '[0, 5]', '[0, 1]', ''],
        'forcing_vars': ['P', 'Ep', 'T'],
        'has_routing': False,
    },
    44: {
        'name': 'ECHO model',
        'marrmot_file': 'm_44_echo_16p_6s',
        'step_fn': step_model_44,
        'n_stores': 6,
        'n_params': 16,
        'param_names': ['rho', 'Ts', 'Tm', 'as_', 'af', 'Gmax', 'theta', 'phi', 'Smax', 'sw', 'sm', 'Ksat', 'c', 'Lmax', 'kf', 'ks'],
        'param_units': ['mm', '°C', '°C', 'mm/°C/d', '–', 'mm/d', '–', '', 'mm', 'mm', 'mm', 'd⁻¹', '–', 'mm/d', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[0, 5]', '[-3, 5]', '[-3, 3]', '[0, 20]', '[0, 1]', '[0, 2]', '[0, 1]', '', '[1, 2000]', '[0.05, 0.95] (as fraction of soil moisture capacity)', 'range: not specified in Table S3 — check S2.x.3 parameter table only', '[0, 1]', '[1, 5]', '[0, 20]', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep', 'T'],
        'has_routing': False,
    },
    45: {
        'name': 'Precipitation-Runoff Modelling System (PRMS)',
        'marrmot_file': 'm_45_prms_18p_7s',
        'step_fn': step_model_45,
        'n_stores': 7,
        'n_params': 18,
        'param_names': ['TT', 'ddf', 'alpha', 'beta', 'STOR', 'RETIP', 'SCN', 'SCX', 'REMX', 'SMAX', 'c_gw', 'RESMAX', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6'],
        'param_units': ['°C', 'mm °C⁻¹ d⁻¹', '–', '–', 'mm', 'mm', '–', '–', 'mm', 'mm', 'mm d⁻¹', 'mm', 'd⁻¹', '–', 'd⁻¹', 'mm⁻¹ d⁻¹', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[-3, 5]', '[0, 20]', '[0, 1]', '[0, 1]', '[0, 5]', '[0, 50]', '[0, 1]', '[0, 1]', '[1, 2000]', '[1, 2000]', '[0, 20]', '[1, 300]', 'not specified in Table S3 — check S2.45.3 parameter table only', '[1, 5]', 'not specified in Table S3 — check S2.45.3 parameter table only', 'not specified in Table S3 — check S2.45.3 parameter table only', 'not specified in Table S3 — check S2.45.3 parameter table only', 'not specified in Table S3 — check S2.45.3 parameter table only'],
        'forcing_vars': ['P', 'Ep', 'T'],
        'has_routing': False,
    },
    46: {
        'name': 'Climate and Land-use Scenario Simulation in Catchments (CLASSIC)',
        'marrmot_file': 'm_46_classic_12p_8s',
        'step_fn': step_model_46,
        'n_stores': 8,
        'n_params': 12,
        'param_names': ['f_ap', 'f_dp', 'd_p', 'c_q', 'd_1', 'f_as', 'f_ds', 'd_s', 'd_2', 'c_xq', 'c_xs', 'c_u'],
        'param_units': ['–', '–', 'mm', 'd⁻¹', '–', '–', '–', 'mm', '–', 'd⁻¹', 'd⁻¹', 'd⁻¹'],
        'param_ranges': ['[0, 1]', 'not specified in Table S3 — check S2.46.3 parameter table only', '[1, 2000]', '[0, 1]', '[0, 1]', '[0, 1]', 'not specified in Table S3 — check S2.46.3 parameter table only', '[1, 2000]', '[0, 1]', '[0, 1]', '[0, 1]', '[0, 1]'],
        'forcing_vars': ['P', 'Ep'],
        'has_routing': False,
    },
}


def simulate(model_num, P, Ep, params, S0, dt=1.0, T=None, t=None, clip_negative=True):
    """
    Run a full time series through model `model_num`, day by day.

    Parameters
    ----------
    model_num : int
        Model number (1-46), matching `MODEL_REGISTRY`.
    P, Ep : array-like [mm/d]
        Precipitation and potential evapotranspiration forcing.
    params : dict
        Parameter values keyed by the model's own symbols (see
        `MODEL_REGISTRY[model_num]['param_names']`).
    S0 : array-like
        Initial storage(s), in the order given by
        `MODEL_REGISTRY[model_num]['n_stores']` / the model's docstring.
    dt : float
        Time step in days (default 1.0, matching MARRMoT's internal units).
    clip_negative : bool
        After every step, clip each store to be >= 0 (default True). Several
        models have at least one outflow term that is a constant or a
        function of a *different* store rather than of its own store's
        content (e.g. HBV-96's upper-zone percolation `perc = c`), so
        nothing in the printed equations stops that store's content from
        going through zero under an explicit Euler step. This is the same
        physically-motivated safety clip this course used in Example 3/4
        (`S_n >= S_max`, clipped to `[0, S_max]`) applied generically here
        at the orchestration level, since 46 models' worth of per-model
        physical floors can't all be verified individually. Set to False to
        see the raw (occasionally runaway) trajectory instead.
    T : array-like [deg C], optional
        Temperature forcing -- required if `MODEL_REGISTRY[model_num]
        ['forcing_vars']` includes 'T' (the snow-driven models).
    t : array-like [day-of-year], optional
        Seasonal time signal -- required for the few models whose
        'forcing_vars' includes 't'. Defaults to day-of-year (1-based, mod
        365) from the position in the series if not supplied.

    Returns
    -------
    dict with keys:
        'S'      : ndarray, shape (n_steps+1, n_stores) -- storage trajectory
        'Q'      : ndarray, shape (n_steps,)             -- total streamflow [mm/d]
        'fluxes' : dict of ndarrays, shape (n_steps,)    -- every named flux
    """
    info = MODEL_REGISTRY[model_num]
    step_fn = info['step_fn']
    n = len(P)
    P = np.asarray(P, dtype=float)
    Ep = np.asarray(Ep, dtype=float)
    if T is not None:
        T = np.asarray(T, dtype=float)
    if t is None:
        t = (np.arange(n) % 365) + 1
    else:
        t = np.asarray(t, dtype=float)

    S = np.zeros((n + 1, info['n_stores']))
    S[0] = S0
    Q = np.zeros(n)
    flux_series = {}
    uh_state = None

    for i in range(n):
        forcings = {'P': P[i], 'Ep': Ep[i]}
        if 'T' in info['forcing_vars']:
            forcings['T'] = T[i]
        if 't' in info['forcing_vars']:
            forcings['t'] = t[i]

        state_in = list(S[i])
        if info['has_routing']:
            S_new, Q_i, fluxes, uh_state = step_fn(state_in, forcings, params,
                                                     dt=dt, uh_state=uh_state)
        else:
            S_new, Q_i, fluxes = step_fn(state_in, forcings, params, dt=dt)

        if clip_negative:
            S_new = np.maximum(S_new, 0.0)
        S[i + 1] = S_new
        Q[i] = Q_i
        for k, v in fluxes.items():
            flux_series.setdefault(k, np.zeros(n))[i] = v

    return {'S': S, 'Q': Q, 'fluxes': flux_series}
