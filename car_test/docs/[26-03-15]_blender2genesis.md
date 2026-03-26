$$
J = \sum_{h=1}^{H} \left[
\underbrace{w_{v} |v_h - v_{\text{ref},h}| + w_{\kappa} |\kappa_h - \kappa_{\text{ref},h}|}_{\text{State Tracking}} +
\underbrace{w_{\text{cte}} |\text{CTE}_h| + w_{\text{he}} |\Delta \psi_h|}_{\text{Path Following Error}} +
\underbrace{w_{a} |a_h - a_{\text{ref},h}|}_{\text{Dynamics Consistency}} +
\underbrace{w_{\Delta u} |u_h - u_{h-1}| + w_{\text{ff}} |u_h - u_{\text{ref},h}|}_{\text{Control Smoothness \& Bias}}
\right]
$$
