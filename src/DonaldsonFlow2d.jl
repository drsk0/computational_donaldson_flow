using ModelingToolkit
using MethodOfLines
using OrdinaryDiffEq
using DomainSets

@parameters x y t
@variables u(..) K(..)

Dt = Differential(t)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

△(u) = -Dxx(u) - Dyy(u)

x_min = y_min = 0.0
x_max = y_max = 2π
t_min = 0.0

eq = [
    Dt(u(t, x, y)) ~ △(K(t, x, y)), 
    u(t, x, y) ~ 1 / K(t, x, y)
]

domain(t_max = 10.0) = [
    t ∈ Interval(0.0, t_max), 
    x ∈ Interval(0.0, x_max), 
    y ∈ Interval(0.0, y_max)
]

bcs(u₀) = [
    u(0, x, y) ~ u₀(x, y),
    K(0, x, y) ~ 1 / u₀(x, y),
    u(t, x_min, y) ~ u(t, x_max, y),
    u(t, x, y_min) ~ u(t, x, y_max)
]

pde(u₀, t_max) = 
    @named pdesys = PDESystem(eq, bcs(u₀), domain(t_max), [t, x, y], [u(t, x, y), K(t, x, y)])

function solve2d(;N = 32, method = Rosenbrock23(), t_max = 10.0, u₀)
    discretization = MOLFiniteDifference([x => N, y => N], t)
    pdesys = pde(u₀, t_max)
    prob = discretize(pdesys, discretization)
    @time sol = solve(prob, method, saveat=0.1) 
    return sol
end
