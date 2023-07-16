Computational solutions for the Donaldson geometric flow
========================================================

This is a research project to study Donaldson's geometric flow on the space of
symplectic structures by means of computational solutions. In particular, I use
PINN's (physically informed neural networks) to approximate geometric quantities
arising in the flow. The hope is to gain a better intuition on the existence
of critical points, singularities and the structure of the space of symplectic
forms.

An overview of Donaldson's geometric flow on the space of symplectic structures
can be found [here](https://content.intlpress.com/journal/JSG/article/4226/info).

The reasoning for the local coordinate expression of the flow is contained in
`local_coordinates.tex`.

2d solutions
------------

There is a 2-dimensional analog to the Donaldson flow on the space of volume
forms of a closed surface. Its equation is given by 
$$\partial_t u = d^*d\frac{1}{u}$$ 
This equation is closely related to the heat flow. In particular, there is a
maximum-principle that guarantees convergence to a constant solution. To solve
it, a finite-difference method is used to convert the problem to an ODE problem.
Then the Rosenbrock23 solver is used to solve the ODE.

The following are a few solutions on the 2-torus.

https://github.com/drsk0/computational_donaldson_flow/assets/827698/f118badd-c95b-4e2e-9e52-08299cd18c69

https://github.com/drsk0/computational_donaldson_flow/assets/827698/8b34ccda-77aa-4a17-94b5-2de298c721ae

https://github.com/drsk0/computational_donaldson_flow/assets/827698/bd956db9-2dfb-438e-9436-090c48eb38fc

A few insights:
  - The maximum principle can be clearly seen at work.
  - Due to the inversion $\frac{1}{u}$ the convergence is much faster where $u$ is small, and can become extremely slow when $u$ is big.
  - Hence, the flow favours points where $u$ is big, while points where $u$ is small quickly make $u$ grow.

Critical points
---------------

Looking for critical points of the flow means solving a (highly) non-linear elliptic PDE,
$$\sum_{i = 1}^3 J\_i X_{K_i} = 0,$$
where $X_{K_i}$ is defined by the equations
$$dK_i = \rho(X_i, \cdot), \qquad K\_i = \frac{\rho\wedge \omega_i}{{\rm dvol}_\rho}.$$

We are using NeuralPDE.jl to define a loss function on the jet-bundle of
the four-torus corresponding to the critical point equation. The symplectic
structure is then approximated by a fully connected neural network with
several hidden layers. To make the optimization more flexible, we also treat
the Hamiltonian vector fields $X_i$ as free variables and couple them to the
symplectic structure via the constraint
$$dK_i = \rho(X_i, \cdot).$$
Using quasi-random point samples, we get an optimization problem that can be
solved with stochastic gradient decent and the ADAM optimizer.

Unsurprisingly, this finds the lowest critical point of the problem with
constant $K_i$ functions. To find higher critical points, additional constraints
need to be imposed.
