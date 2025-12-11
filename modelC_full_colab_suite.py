!pip install qutip emcee corner

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================
# SHARED MODEL C PARAMETERS
# ============================================

# Curvature dressing parameters
m0 = 1.0        # base mass scale (arb.)
c_R = 1.0e22    # curvature coupling
Gamma0 = 1.0e-2 # reference Γ_grav at R0
R0 = 1.0e-23    # reference curvature

# Correlation parameter (true value used in tests)
rho_true = 0.6


# ============================================
# HELPER FUNCTIONS
# ============================================

def gamma_grav_from_R(R):
    """
    Curvature-screened gravitational decoherence:
        m_eff^2 = m0**2 + c_R * abs(R)
        R_c ~ 1/m_eff
        Γ_grav ∝ R_c**3 ∝ (m0**2 + c_R*abs(R))**(-3/2)
    Normalized such that Γ_grav(R0) = Gamma0.
    """
    R = np.array(R)
    m_eff_ref2 = m0**2 + c_R * np.abs(R0)
    m_eff2     = m0**2 + c_R * np.abs(R)
    return Gamma0 * (m_eff_ref2 / m_eff2)**1.5


def fit_exp_decay(t, y, min_points=20, cutoff_frac=0.8):
    """
    Fit |y(t)| ≈ A exp(-Gamma t) and return (Gamma, R^2).
    """
    y_abs = np.abs(y)
    A0 = y_abs[0] if y_abs[0] > 0 else 1.0
    y_norm = y_abs / A0

    eps = 1e-12
    mask = (y_norm > eps) & (t < cutoff_frac * t[-1])
    if np.sum(mask) < min_points:
        mask = (y_norm > eps)

    t_fit = t[mask]
    log_y = np.log(y_norm[mask])

    slope, intercept = np.polyfit(t_fit, log_y, 1)
    Gamma = -slope

    pred   = slope * t_fit + intercept
    ss_res = np.sum((log_y - pred)**2)
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return Gamma, R2


# =========================================================
# PART 1: QUBIT LINDLAD TEST (math + curvature scaling)
# =========================================================

def run_qubit_tests():
    print("=== PART 1: MODEL C QUANTUM QUBIT TEST ===\n")

    # qubit operators
    sx = qt.sigmax()
    sz = qt.sigmaz()
    H  = qt.qzero(2)

    # initial |+>
    psi_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    rho0     = psi_plus * psi_plus.dag()

    # environment rate (Lindblad)
    Gamma_env_qubit = 5.0e-3

    # curvature points (choose ones where Γ_grav is resolvable)
    R_values = np.array([1.0e-25, 1.0e-21, 1.0e-17])

    Gamma_fit_list  = []
    Gamma_th_list   = []
    Gamma_grav_list = []
    R2_list         = []

    # store traces for plotting
    qubit_traces = []  # list of (R, t_vals, sx_expect, fit_curve)

    print(f"rho = {rho_true}")
    print(f"Gamma_env_qubit = {Gamma_env_qubit:.3e}")
    print(f"Curvature points: {R_values}\n")

    for R in R_values:
        Gamma_grav = gamma_grav_from_R(R)
        # Model C total rate
        Gamma_tot = (Gamma_env_qubit + Gamma_grav
                     + 2.0 * rho_true * np.sqrt(Gamma_env_qubit * Gamma_grav))
        Gamma_th = 2.0 * Gamma_tot   # coherence decay for σ_z dephasing

        # Lindblad operator with Γ_tot
        L = np.sqrt(Gamma_tot) * sz
        c_ops = [L]

        # times
        Gamma_char = max(Gamma_th, 1e-6)
        t_char = 1.0 / Gamma_char
        t_max  = 6.0 * t_char
        t_vals = np.linspace(0.0, t_max, 400)

        result = qt.mesolve(H, rho0, t_vals, c_ops=c_ops, e_ops=[sx])
        sx_expect = np.array(result.expect[0])

        Gamma_fit, R2 = fit_exp_decay(t_vals, sx_expect)

        # simple fit curve with unit amplitude
        fit_curve = np.exp(-Gamma_fit * t_vals) * np.real(sx_expect[0])

        qubit_traces.append((R, t_vals, sx_expect, fit_curve))

        Gamma_fit_list.append(Gamma_fit)
        Gamma_th_list.append(Gamma_th)
        Gamma_grav_list.append(Gamma_grav)
        R2_list.append(R2)

        rel_err = np.abs(Gamma_fit - Gamma_th)/max(Gamma_th, 1e-12)*100.0

        print(f"R = {R:.2e}")
        print(f"  Γ_grav(R)        = {Gamma_grav:.3e}")
        print(f"  Γ_tot (Lindblad) = {Gamma_tot:.3e}")
        print(f"  Γ_fit (from <σx>)= {Gamma_fit:.3e}")
        print(f"  Γ_theory (2Γ_tot)= {Gamma_th:.3e}")
        print(f"  Rel. error       = {rel_err:.2f}%")
        print(f"  R^2 fit          = {R2:.4f}\n")

    Gamma_fit_arr  = np.array(Gamma_fit_list)
    Gamma_th_arr   = np.array(Gamma_th_list)
    Gamma_grav_arr = np.array(Gamma_grav_list)
    R2_arr         = np.array(R2_list)

    rel_errs    = np.abs(Gamma_fit_arr - Gamma_th_arr)/np.maximum(Gamma_th_arr, 1e-12)
    max_rel_err = float(np.max(rel_errs))
    mean_rel_err = float(np.mean(rel_errs))

    # curvature scaling slope of Γ_grav vs (m0**2 + c_R*abs(R))
    X = np.log(m0**2 + c_R * np.abs(R_values))
    Y = np.log(Gamma_grav_arr)
    slope, intercept = np.polyfit(X, Y, 1)
    scaling_exponent = float(slope)

    # tolerances
    tol_geom  = 5e-2   # 5% max error
    tol_slope = 0.1    # around -1.5

    qubit_math_pass = bool(max_rel_err < tol_geom and np.all(R2_arr > 0.99))
    qubit_curv_pass = bool(abs(scaling_exponent + 1.5) < tol_slope)

    print("=== SUMMARY (QUBIT) ===")
    print(f"Max relative error (math)  = {max_rel_err*100:.2f}%")
    print(f"Mean relative error (math) = {mean_rel_err*100:.2f}%")
    print(f"Scaling exponent Γ_grav vs R = {scaling_exponent:.3f} (expected -1.5)\n")
    print(f"Model_C_qubit_math_test_pass     = {qubit_math_pass}")
    print(f"Model_C_qubit_curv_scaling_pass  = {qubit_curv_pass}")
    print("\n" + "="*72 + "\n")

    # --- PLOT qubit coherence decays + fits ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for R, t_vals, sx_expect, fit_curve in qubit_traces:
        ax.plot(t_vals, np.real(sx_expect),
                label=f"R={R:.0e} data")
        ax.plot(t_vals, fit_curve, '--',
                label=f"R={R:.0e} fit")
    ax.set_xlabel("t")
    ax.set_ylabel("Re⟨σ_x⟩")
    ax.set_title("Qubit coherence decay (Model C)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    return qubit_math_pass, qubit_curv_pass


# =========================================================
# PART 2: OSCILLATOR / CAT-STATE TEST
# =========================================================

def run_oscillator_tests():
    print("=== PART 2: MODEL C OSCILLATOR / CAT TEST ===\n")

    # choose Γ_env small enough that Γ_grav scaling is not buried
    Gamma_env_osc = 1.0e-5
    alpha = 4.0        # cat separation
    N     = 40         # Fock cutoff

    a    = qt.destroy(N)
    adag = a.dag()
    n_op = adag * a               # number operator
    x_op = (a + adag)/np.sqrt(2)  # position-like operator

    # cat state |ψ_cat> ∝ |α> + |−α>
    psi_plus  = qt.coherent(N, alpha)
    psi_minus = qt.coherent(N, -alpha)
    psi_cat   = (psi_plus + psi_minus).unit()
    rho0      = psi_cat * psi_cat.dag()

    # parity operator: (-1)**n = exp(iπ n)
    parity = (1j * np.pi * n_op).expm()

    H = qt.qzero(N)

    R_values = np.array([1.0e-25, 1.0e-21, 1.0e-17])

    Gamma_cat_list  = []
    Gamma_tot_list  = []
    Gamma_grav_list = []
    R2_list         = []

    osc_traces = []  # (R, t_vals, parity_expect, fit_curve)

    print(f"rho = {rho_true}")
    print(f"Gamma_env_osc = {Gamma_env_osc:.3e}")
    print("Note: Γ_tot = Γ_grav (environment omitted here to test curvature scaling).")
    print(f"Curvature points: {R_values}")
    print(f"alpha = {alpha:.1f}, N = {N}\n")

    for R in R_values:
        Gamma_grav = gamma_grav_from_R(R)
        Gamma_tot  = Gamma_grav   # for this test we drop Γ_env

        L = np.sqrt(Gamma_tot) * x_op
        c_ops = [L]

        # decoherence timescale for cat ~ 1/(4 alpha**2 Γ_tot)
        Gamma_cat_th = 4.0 * alpha**2 * Gamma_tot

        Gamma_char = Gamma_cat_th
        t_char = 1.0 / max(Gamma_char, 1e-8)
        t_max  = 4.0 * t_char
        t_vals = np.linspace(0.0, t_max, 350)

        result = qt.mesolve(H, rho0, t_vals, c_ops=c_ops, e_ops=[parity])
        parity_expect = np.array(result.expect[0])

        Gamma_cat_fit, R2 = fit_exp_decay(
            t_vals, parity_expect, min_points=30, cutoff_frac=0.8
        )

        fit_curve = np.exp(-Gamma_cat_fit * t_vals) * np.real(parity_expect[0])
        osc_traces.append((R, t_vals, parity_expect, fit_curve))

        Gamma_cat_list.append(Gamma_cat_fit)
        Gamma_tot_list.append(Gamma_tot)
        Gamma_grav_list.append(Gamma_grav)
        R2_list.append(R2)

        rel_err = np.abs(Gamma_cat_fit - Gamma_cat_th)/max(Gamma_cat_th, 1e-12)*100.0

        print(f"R = {R:.2e}")
        print(f"  Γ_grav(R)        = {Gamma_grav:.3e}")
        print(f"  Γ_tot(R)         = {Gamma_tot:.3e}")
        print(f"  Γ_cat (fit)      = {Gamma_cat_fit:.3e}")
        print(f"  Γ_cat (theory)   = {Gamma_cat_th:.3e}")
        print(f"  R^2 (exp fit)    = {R2:.4f}")
        print(f"  Rel. error       = {rel_err:.2f}%\n")

    Gamma_cat_arr  = np.array(Gamma_cat_list)
    Gamma_tot_arr  = np.array(Gamma_tot_list)
    Gamma_grav_arr = np.array(Gamma_grav_list)
    R2_arr         = np.array(R2_list)

    # scaling Γ_cat vs Γ_tot
    X_tot = np.log(Gamma_tot_arr)
    Y_cat = np.log(Gamma_cat_arr)
    slope_tot, intercept_tot = np.polyfit(X_tot, Y_cat, 1)

    # curvature scaling Γ_cat vs (m0**2 + c_R*abs(R))
    X_curv = np.log(m0**2 + c_R * np.abs(R_values))
    Y_cat2 = np.log(Gamma_cat_arr)
    slope_curv, intercept_curv = np.polyfit(X_curv, Y_cat2, 1)

    tol_tot_slope  = 0.25   # around 1.0
    tol_curv_slope = 0.40   # around -1.5
    min_R2         = 0.20

    osc_tot_scaling_pass = bool(
        abs(slope_tot - 1.0) < tol_tot_slope and np.all(R2_arr > min_R2)
    )
    osc_curv_scaling_pass = bool(
        abs(slope_curv + 1.5) < tol_curv_slope and np.all(R2_arr > min_R2)
    )

    print("=== SUMMARY (OSCILLATOR) ===")
    print(f"Slope log Γ_cat vs log Γ_tot    = {slope_tot:.3f} (expected ~1)")
    print(f"Slope log Γ_cat vs log(m0**2+..) = {slope_curv:.3f} (expected ~-1.5)")
    print(f"Min R^2 (exp fits)              = {np.min(R2_arr):.4f}\n")
    print("Logical results:")
    print(f"Model_C_osc_tot_scaling_pass   = {osc_tot_scaling_pass}")
    print(f"Model_C_osc_curv_scaling_pass  = {osc_curv_scaling_pass}")
    print("\n" + "="*72 + "\n")

    # --- PLOT cat-state parity decay + fits ---
    fig, ax = plt.subplots(figsize=(6, 4))
    for R, t_vals, parity_expect, fit_curve in osc_traces:
        ax.plot(t_vals, np.real(parity_expect),
                label=f"R={R:.0e} data")
        ax.plot(t_vals, fit_curve, '--',
                label=f"R={R:.0e} fit")
    ax.set_xlabel("t")
    ax.set_ylabel("Re⟨Parity⟩")
    ax.set_title("Cat-state parity decay (Model C)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    return osc_tot_scaling_pass, osc_curv_scaling_pass


# =========================================================
# PART 3: REALISTIC NOISY GLOBAL CURVATURE INFERENCE (GRID)
# =========================================================

def run_global_curvature_test_realistic():
    print("=== PART 3: REALISTIC NOISY GLOBAL CURVATURE INFERENCE (grid) ===\n")

    # Known environmental rate and curvature points
    Gamma_env = 5.0e-3
    R_values  = np.array([5.0e-24, 1.0e-23, 5.0e-23,
                          1.0e-22, 5.0e-22, 1.0e-21, 5.0e-21])

    # 3% measurement error on each Γ_tot (realistic)
    sigma_rel = 0.03

    print(f"Fixed Gamma_env = {Gamma_env:.2e}")
    print(f"True rho        = {rho_true:.3f}")
    print(f"Measurement uncertainty = {sigma_rel:.1%} on each Γ_tot")
    print(f"Curvature points R = {R_values}\n")

    # Synthetic "experimental" data
    Gamma_grav_true = gamma_grav_from_R(R_values)
    Gamma_tot_true  = (Gamma_env
                       + Gamma_grav_true
                       + 2.0 * rho_true * np.sqrt(Gamma_env * Gamma_grav_true))

    np.random.seed(123)
    sigma_abs = sigma_rel * Gamma_tot_true
    Gamma_tot_obs = Gamma_tot_true + np.random.normal(0, sigma_abs)

    # --- Parameter grid over (log10 c_R, log10 Gamma0, rho) ---
    log10_c_R_grid    = np.linspace(21.5, 22.5, 21)
    log10_Gamma0_grid = np.linspace(-2.3, -1.7, 19)
    rho_grid          = np.linspace(0.3, 0.9, 25)

    chi2_best = np.inf
    best = None
    results = []

    for log10_cR in log10_c_R_grid:
        cR = 10**log10_cR
        m_ref2 = m0**2 + cR * np.abs(R0)
        for log10_G0 in log10_Gamma0_grid:
            G0 = 10**log10_G0
            m_eff2 = m0**2 + cR * np.abs(R_values)
            Ggrav = G0 * (m_ref2 / m_eff2)**1.5
            for rho in rho_grid:
                model = Gamma_env + Ggrav + 2.0 * rho * np.sqrt(Gamma_env * Ggrav)
                chi2 = np.sum(((Gamma_tot_obs - model)/sigma_abs)**2)
                results.append((log10_cR, log10_G0, rho, chi2))
                if chi2 < chi2_best:
                    chi2_best = chi2
                    best = (log10_cR, log10_G0, rho)

    log10_cR_best, log10_G0_best, rho_best = best

    print("Best-fit (grid) parameters:")
    print(f"  log10(c_R)    = {log10_cR_best:.3f}")
    print(f"  log10(Gamma0) = {log10_G0_best:.3f}")
    print(f"  rho           = {rho_best:.3f}")
    print(f"  chi2_min      = {chi2_best:.2f}\n")

    # Best-fit model + Γ_grav
    cR_best = 10**log10_cR_best
    G0_best = 10**log10_G0_best
    m_eff2_best = m0**2 + cR_best * np.abs(R_values)
    m_ref2_best = m0**2 + cR_best * np.abs(R0)
    Gamma_grav_best = G0_best * (m_ref2_best / m_eff2_best)**1.5
    Gamma_tot_best  = (Gamma_env
                       + Gamma_grav_best
                       + 2.0 * rho_best * np.sqrt(Gamma_env * Gamma_grav_best))

    # --- Near-best region for approximate "posterior" ---
    delta_chi2_1sigma = 3.5
    near = [r for r in results if r[3] - chi2_best <= delta_chi2_1sigma]

    print(f"Near-best sample size (Δχ² ≤ {delta_chi2_1sigma:.1f}): {len(near)}")

    slopes = []
    rhos   = []

    for log10_cR, log10_G0, rho, _ in near:
        cR = 10**log10_cR
        G0 = 10**log10_G0
        m_eff2 = m0**2 + cR * np.abs(R_values)
        m_ref2 = m0**2 + cR * np.abs(R0)
        Ggrav = G0 * (m_ref2 / m_eff2)**1.5

        X = np.log(m0**2 + cR * np.abs(R_values))
        Y = np.log(Ggrav)
        slope, _ = np.polyfit(X, Y, 1)

        slopes.append(slope)
        rhos.append(rho)

    slopes = np.array(slopes)
    rhos   = np.array(rhos)

    slope_med = float(np.median(slopes))
    slope_lo, slope_hi = np.percentile(slopes, [16, 84])
    rho_med = float(np.median(rhos))
    rho_lo, rho_hi = np.percentile(rhos, [16, 84])

    slope_true = -1.5
    rho_ok   = (rho_true >= rho_lo) and (rho_true <= rho_hi)
    slope_ok = (slope_true >= slope_lo) and (slope_true <= slope_hi)
    within_band = abs(slope_med + 1.5) < 0.25   # ±0.25

    global_realistic_pass = bool(rho_ok and slope_ok and within_band)

    print("\nPosterior-ish summaries from grid:")
    print(f"  rho_true = {rho_true:.3f}")
    print(f"  rho_med  = {rho_med:.3f} [{rho_lo:.3f}, {rho_hi:.3f}]")
    print(f"  slope_true = -1.500")
    print(f"  slope_med  = {slope_med:.3f} [{slope_lo:.3f}, {slope_hi:.3f}]")
    print(f"  rho in interval?   {rho_ok}")
    print(f"  slope in interval? {slope_ok}")
    print(f"  |slope_med + 1.5| < 0.25 ? {within_band}\n")
    print(f"Model_C_global_realistic_pass = {global_realistic_pass}")
    print("\n" + "="*72 + "\n")

    # --- PLOT Γ_tot data vs best-fit model (log–log) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(R_values, Gamma_tot_obs, yerr=sigma_abs,
                fmt='o', label="data")
    ax.plot(R_values, Gamma_tot_best, '-', label="best-fit model")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("R")
    ax.set_ylabel("Γ_tot")
    ax.set_title("Global curvature test: Γ_tot vs R")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    return global_realistic_pass


# =========================================================
# PART 4: MULTI-MODEL AIC COMPARISON (WITH PLOTS)
# =========================================================

def run_multimodel_aic_test():
    print("=== PART 4: MULTI-MODEL COMPARISON (AIC / χ²) ===\n")

    # True model used to generate synthetic ΔΓ data
    true_model_name = "Model_C"   # change to test bias

    R_true = 1.0e-23
    rho    = rho_true
    Genv_vals   = np.logspace(-4, -1, 15)
    sigma_noise = 1e-4

    # helper model forms
    def model_C_delta(Genv, Ggrav, rho):
        return Ggrav + 2.0 * rho * np.sqrt(Genv * Ggrav)

    def model_linear_delta(Genv, Ggrav):
        return Ggrav * np.ones_like(Genv)

    def model_env_nl_delta(Genv, a):
        return a * np.sqrt(Genv)

    Ggrav_true = gamma_grav_from_R(R_true)

    if true_model_name == "Model_C":
        delta_true = model_C_delta(Genv_vals, Ggrav_true, rho)
    elif true_model_name == "Linear_grav":
        delta_true = model_linear_delta(Genv_vals, Ggrav_true)
    elif true_model_name == "Env_nonlinear":
        a_true = 2.0 * np.sqrt(Ggrav_true)
        delta_true = model_env_nl_delta(Genv_vals, a_true)
    else:
        raise ValueError("Unknown true_model_name.")

    delta_obs = delta_true + np.random.normal(0.0, sigma_noise, size=delta_true.shape)

    # fit Model C: ΔΓ = B + C sqrt(Genv)
    u = np.sqrt(Genv_vals)
    A = np.vstack([np.ones_like(u), u]).T
    params, _, _, _ = np.linalg.lstsq(A, delta_obs, rcond=None)
    B_fit, C_fit = params
    Ggrav_fit_C  = max(B_fit, 0.0)
    rho_fit_C    = C_fit/(2.0*np.sqrt(Ggrav_fit_C)) if Ggrav_fit_C > 0 else 0.0
    delta_fit_C  = model_C_delta(Genv_vals, Ggrav_fit_C, rho_fit_C)

    # fit linear grav
    Ggrav_fit_L = np.mean(delta_obs)
    delta_fit_L = model_linear_delta(Genv_vals, Ggrav_fit_L)

    # fit env-nonlinear
    a_fit_NL     = np.dot(u, delta_obs)/np.dot(u, u)
    delta_fit_NL = model_env_nl_delta(Genv_vals, a_fit_NL)

    def chi2(y_obs, y_fit, sigma):
        return np.sum(((y_obs - y_fit)/sigma)**2)

    def aic(chi2_val, k):
        return 2*k + chi2_val

    chi2_C  = chi2(delta_obs, delta_fit_C,  sigma_noise)
    chi2_L  = chi2(delta_obs, delta_fit_L,  sigma_noise)
    chi2_NL = chi2(delta_obs, delta_fit_NL, sigma_noise)

    aic_C  = aic(chi2_C,  k=2)
    aic_L  = aic(chi2_L,  k=1)
    aic_NL = aic(chi2_NL, k=1)

    models    = ["Model_C", "Linear_grav", "Env_nonlinear"]
    chi2_vals = np.array([chi2_C, chi2_L, chi2_NL])
    aic_vals  = np.array([aic_C,  aic_L,  aic_NL])

    idx_best_chi2 = np.argmin(chi2_vals)
    idx_best_aic  = np.argmin(aic_vals)

    best_model_chi2 = models[idx_best_chi2]
    best_model_aic  = models[idx_best_aic]

    Model_C_pref_chi2 = bool(best_model_chi2 == "Model_C")
    Model_C_pref_aic  = bool(best_model_aic  == "Model_C")

    print(f"True generating model: {true_model_name}\n")

    print("Chi-square values:")
    for name, val in zip(models, chi2_vals):
        print(f"  {name:15s}  χ² = {val:.2f}")
    print("\nAIC values (lower is better):")
    for name, val in zip(models, aic_vals):
        print(f"  {name:15s}  AIC = {val:.2f}")
    print()
    print(f"Best by χ²   : {best_model_chi2}")
    print(f"Best by AIC  : {best_model_aic}\n")
    print("Logical flags (no hard-wired passes):")
    print(f"Model_C_pref_chi2 = {Model_C_pref_chi2}")
    print(f"Model_C_pref_aic  = {Model_C_pref_aic}")
    print()
    print("Fitted parameters:")
    print(f"  Model C:       Ggrav_fit = {Ggrav_fit_C:.3e}, rho_fit = {rho_fit_C:.3f}")
    print(f"  Linear grav:   Ggrav_fit = {Ggrav_fit_L:.3e}")
    print(f"  Env-nonlinear: a_fit     = {a_fit_NL:.3e}")
    print("\n" + "="*72 + "\n")

    # --- PLOTS ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    axes[0].bar(models, chi2_vals)
    axes[0].set_title("Model comparison: χ²")
    axes[0].set_ylabel("χ²")

    axes[1].bar(models, aic_vals)
    axes[1].set_title("Model comparison: AIC")
    axes[1].set_ylabel("AIC")

    plt.tight_layout()
    plt.show()

    return Model_C_pref_chi2, Model_C_pref_aic


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    qubit_math_pass, qubit_curv_pass = run_qubit_tests()
    osc_tot_pass, osc_curv_pass     = run_oscillator_tests()
    global_realistic_pass           = run_global_curvature_test_realistic()
    pref_chi2, pref_aic             = run_multimodel_aic_test()

    print("=== OVERALL FLAGS ===")
    print(f"Model_C_qubit_math_test_pass     = {qubit_math_pass}")
    print(f"Model_C_qubit_curv_scaling_pass  = {qubit_curv_pass}")
    print(f"Model_C_osc_tot_scaling_pass     = {osc_tot_pass}")
    print(f"Model_C_osc_curv_scaling_pass    = {osc_curv_pass}")
    print(f"Model_C_global_realistic_pass    = {global_realistic_pass}")
    print(f"Model_C_pref_chi2                = {pref_chi2}")
    print(f"Model_C_pref_aic                 = {pref_aic}")
