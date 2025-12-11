# Model C: Curvature–Suppressed Correlation Lengths as a Source of Geometry–Dependent Decoherence

This repository contains the full research package for the paper:

**“Model C: Curvature–Suppressed Correlation Lengths as a Falsifiable Source of Geometry–Dependent Decoherence”**  
*Richard Taylor (2025)*

Model C introduces a curvature-dressed correlation length  
\[
m_{\rm eff}^2(R)=m_0^2 + c_R |R|,
\]
which generates a gravitational decoherence rate  
\[
\Gamma_{\rm grav}(R) \propto (m_0^2 + c_R |R|)^{-3/2}.
\]

Coupling this channel to environmental decoherence produces the geometric-mean deformation:
\[
\Gamma_{\rm tot}=\Gamma_{\rm env}+\Gamma_{\rm grav}
+2\rho\sqrt{\Gamma_{\rm env}\Gamma_{\rm grav}},
\qquad |\rho|\le1.
\]

This repository provides:
- The **full LaTeX manuscript**  
- All **figures, bibliography**, and supplementary materials  
- A complete **4-stage numerical verification suite** (QuTiP + emcee)  
- AIC model comparison against competing models  
- A reproducibility pipeline

---

## 📦 Repository Contents# modelC-decoherence
Test suite and paper for Model C: curvature-suppressed decoherence.
