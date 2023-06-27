using ModelingToolkit
using MethodOfLines
using OrdinaryDiffEq
using DomainSets
using Makie
using CairoMakie

@parameters x y t
@variables u(..) 

Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)

x_min = y_min = 0.0
x_max = y_max = 2π
t_min = 0.0

eq = [
    Dt(u(t, x, y)) ~ Dx(1/u(t,x,y)^2 * Dx(u(t,x,y))) + Dy(1/u(t,x,y)^2 * Dy(u(t,x,y))), 
]

domain(t_max = 10.0) = [
    t ∈ Interval(0.0, t_max), 
    x ∈ Interval(0.0, x_max), 
    y ∈ Interval(0.0, y_max)
]

bcs(u₀) = [
    u(0, x, y) ~ u₀(x, y),
    u(t, x_min, y) ~ u(t, x_max, y),
    u(t, x, y_min) ~ u(t, x, y_max)
]

pde(u₀, t_max) = 
    @named pdesys = PDESystem(eq, bcs(u₀), domain(t_max), [t, x, y], [u(t, x, y)])

function solve2d(;N = 32, method = Rosenbrock23(), t_max = 10.0, u₀)
    discretization = MOLFiniteDifference([x => N, y => N], t)
    pdesys = pde(u₀, t_max)
    prob = discretize(pdesys, discretization)
    @time sol = solve(prob, method, saveat=0.1) 
    return sol
end

function createAnimation(sol, fpOut::String)
    solu = sol.u[u(t,x,y)]
    minColor = minimum(solu[1,2:end, 2:end])
    maxColor = maximum(solu[1,2:end, 2:end])
    dimT,dimX,dimY = size(solu)

    ti = Observable(1)

    buf = lift(ti) do ti
        solu[ti,2:end,2:end]
    end

    scene, axis, hm =  Makie.heatmap(buf, padding = (0,0), colorrange = (minColor, maxColor))
    Colorbar(scene[:, end+1], hm)

    record(scene, fpOut, 1:dimT) do t
        ti[] = t
    end
end
