# 2dNavierStokes

Numerical code for 2d Navier Stokes Equations

$$\frac{\partial \omega}{\pasrtial t} = -(\mathbf{v}\cdot \namba)\omega - \mu (-\Delta)^p \omega - \alpha\omega  + F$$

**Parameters:** Hyperviscosity $\nu$, linear friction $\alpha$.

**Boundary Conditions:** Periodic boundary conditions.

**Spatial Discretization:** Pseudo-spectral method with full dealiasing with the 3/2-rule.

**Temporal Discretization:** 4th-order exponential time differencing Runge-Kutta method.

**Data Output:** vorticity, energy/enstrophy dissipation rates, energy spectra.

**C++ Libraries** FFTW, armadillo
