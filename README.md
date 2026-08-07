# pyTQG
Implémentation d'un modèle Thermal Quasi Geostrophic par méthodes pseudo-spectrales, dans le cadre de mon stage de M2 au LOPS

Equations résolues : 
Pour une couche **Warneford et al 2014** : 
```math
\begin{align}
\partial_t q+\mathrm{J}(\psi, q) = \frac{1}{\mathrm{Bu}}\mathrm{J}(\psi, \theta) \\
\partial_t \theta+\mathrm{J}(\psi, \theta) =0 \\
q = (\partial_x^2+\partial_y^2)\psi - \frac{\psi - \theta}{\mathrm{Bu}} + \beta y
\end{align}
```
Et 2 couches couplées **Vic et al 2024**, **a corriger** : 
```math
\begin{align}
\partial_t q_o +\mathrm{J}(\psi_o, q_o) = \frac{1}{R_{d, o}^2}\mathrm{J}(\psi_o, \theta_o) + C_{w, a} \triangle_\mathrm{h} (\psi_o - \psi_a) \, | & \, \partial_t q_a+\mathrm{J}(\psi_a, q_a) = \frac{1}{R_{d, a}^2}\mathrm{J}(\psi_a, \theta_a) + C_{w, a} \triangle_\mathrm{h} (\psi_a - \psi_o)\\
	\partial_t \theta_o + \mathrm{J}(\psi_o, \theta_o) = -C_t(\theta_a -\theta_o) \, | &\, \partial_t \theta_a + \mathrm{J}(\psi_a, \theta_a) = -C_t(\theta_o -\theta_a) \\
	q_o = (\partial_x^2+\partial_y^2)\psi_o -\frac{\psi_o - \theta_o}{R_{d, o}^2} + \beta y \,|&\, q_a = (\partial_x^2+\partial_y^2)\psi_a -\frac{\psi_a - \theta_a}{R_{d, a}^2} + \beta y
\end{align}
```

Trois configurations sont envisagées :
- Bipériodique
- Canal zonal (périodique selon x, fermé selon y)
- Bassin fermé
Seules les deux premières configurations sont implémentées.
Le code est en construction. Le prototypage se fait dans le répertoire `/Proto_notebooks/`.


Le calcul des dérivées se fait soit dans la base de Fourier (si périodicité), avec les matrices de différenciation de Chebyshev (méthode de collocation). 
Les schémas d'intégration temporels (Euler, Heun, LeapFrog avec filtre d'Asselin, RK3, RK4) sont tous explicites, ce qui peut poser des problèmes de stabilité 
lorsqu'on utilise des matrices de différenciation de Chebyshev. A ce stade, aucun dé-aliasing n'est envisagé, de meme que l'utilisation de dissipation numérique... 