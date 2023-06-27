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

