# pyTQG
Thermal Quasi Geostrophic model implementation using pseudo-spectral method.

Equations solved : 
For one layer **Warneford et al 2014** : 
```math
\begin{align}
\partial_t q+\mathrm{J}(\psi, q) = \frac{1}{\mathrm{Bu}}\mathrm{J}(\psi, \theta) \\
\partial_t \theta+\mathrm{J}(\psi, \theta) =0 \\
q = (\partial_x^2+\partial_y^2)\psi - \frac{\psi - \theta}{\mathrm{Bu}} + \beta y
\end{align}
```
and 2 coupled layer **From Vic et al 2024**, **a corriger** : 
```math
\begin{align}
\partial_t q_o +\mathrm{J}(\psi_o, q_o) = \frac{1}{R_{d, o}^2}\mathrm{J}(\psi_o, \theta_o) + C_{w, a} \triangle_\mathrm{h} (\psi_o - \psi_a) \, | & \, \partial_t q_a+\mathrm{J}(\psi_a, q_a) = \frac{1}{R_{d, a}^2}\mathrm{J}(\psi_a, \theta_a) + C_{w, a} \triangle_\mathrm{h} (\psi_a - \psi_o)\\
	\partial_t \theta_o + \mathrm{J}(\psi_o, \theta_o) = -C_t(\theta_a -\theta_o) \, | &\, \partial_t \theta_a + \mathrm{J}(\psi_a, \theta_a) = -C_t(\theta_o -\theta_a) \\
	q_o = (\partial_x^2+\partial_y^2)\psi_o -\frac{\psi_o - \theta_o}{R_{d, o}^2} + \beta y \,|&\, q_a = (\partial_x^2+\partial_y^2)\psi_a -\frac{\psi_a - \theta_a}{R_{d, a}^2} + \beta y
\end{align}
```

Three configuration will be implemented :
- Biperiodic
- Zonal Channel (have priority)
- Closed basin

For the moment, the code is in construction. But all the parts of the solver are implemented on Python Notebooks, on `/Proto_notebooks/` directory.
