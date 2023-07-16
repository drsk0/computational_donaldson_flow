using NeuralPDE: DomainSets
using NeuralPDE, Lux, CUDA, Random, ComponentArrays, Optimization, OptimizationOptimisers, Integrals
using LinearAlgebra
import ModelingToolkit: Interval

@parameters x0 x1 x2 x3
@variables ρ01(..) ρ02(..) ρ03(..) ρ12(..) ρ13(..) ρ23(..)
# @variables K1(..) K2(..) K3(..) 
@variables X11(..) X12(..) X13(..) X14(..) X21(..) X22(..) X23(..) X24(..) X31(..) X32(..) X33(..) X34(..)

# the 4-torus
domain = [
    x0 ∈ Interval(0.0, 1.0),
    x1 ∈ Interval(0.0, 1.0),
    x2 ∈ Interval(0.0, 1.0),
    x3 ∈ Interval(0.0, 1.0),
]

volM = 1.0

∂₀ = Differential(x0)
∂₁ = Differential(x1)
∂₂ = Differential(x2)
∂₃ = Differential(x3)

d₀(f) = [
    ∂₀(f),
    ∂₁(f),
    ∂₂(f),
    ∂₃(f),
]

d₁(λ) = [
    # commented are the signed permutations of the indeces

    #(0,1) + (1,0)
    ∂₀(λ[2]) - ∂₁(λ[1]),
    #(0,2) + (2,0)
    ∂₀(λ[3]) - ∂₂(λ[1]),
    #(0,3) + (3,0)
    ∂₀(λ[4]) - ∂₃(λ[1]),
    #(1,2) + (2,1)
    ∂₁(λ[3]) - ∂₂(λ[2]),
    #(1,3) + (3,1)
    ∂₁(λ[4]) - ∂₃(λ[2]),
    #(2,3) + (3,2)
    ∂₂(λ[4]) - ∂₃(λ[3]),
]

d₂(ρ) = [
    # commented are the signed permutations of the indeces

    #(0,1,2) + (2,0,1) + (1,2,0) - (1,0,2) - (2,1,0) - (0,2,1)
    2 * ∂₀(ρ[4]) - 2 * ∂₁(ρ[2]) + 2 * ∂₂(ρ[1]),
    #(0,1,3) + (3,0,1) + (1,3,0) - (1,0,3) - (0,3,1) - (3,1,0)
    2 * ∂₀(ρ[5]) - 2 * ∂₁(ρ[3]) + 2 * ∂₃(ρ[1]),
    #(0,2,3) + (3,0,2) + (2,3,0) - (2,0,3) - (0,3,2) - (3,2,0)
    2 * ∂₀(ρ[6]) - 2 * ∂₂(ρ[3]) + 2 * ∂₃(ρ[2]),
    #(1,2,3) + (3,1,2) + (2,3,1) - (2,1,3) - (1,3,2) - (3,2,1)
    2 * ∂₁(ρ[6]) - 2 * ∂₂(ρ[5]) + 2 * ∂₃(ρ[4]),
]

J₁ = [
    0 -1 0 0
    1 0 0 0
    0 0 0 -1
    0 0 1 0
] .|> Float64

J₂ = [
    0 0 -1 0
    0 0 0 1
    1 0 0 0
    0 -1 0 0
] .|> Float64

J₃ = [
    0 0 0 -1
    0 0 -1 0
    0 1 0 0
    1 0 0 0
] .|> Float64

J = [
    J₁,
    J₂,
    J₃,
]

u(ρ) = ρ[1] * ρ[6] - ρ[2] * ρ[5] + ρ[3] * ρ[4]

K₁(ρ) = 2(ρ[1] + ρ[6]) / u(ρ)
K₂(ρ) = 2(ρ[2] - ρ[5]) / u(ρ)
K₃(ρ) = 2(ρ[3] + ρ[4]) / u(ρ)

K(ρ) = [
    K₁(ρ),
    K₂(ρ),
    K₃(ρ),
]

# g(A⋅, ⋅) = ρ(⋅, ⋅)
A(ρ) = [
    0.0 ρ[1] ρ[2] ρ[3]
    -ρ[1] 0.0 ρ[4] ρ[5]
    -ρ[2] -ρ[4] 0.0 ρ[6]
    -ρ[3] -ρ[5] -ρ[6] 0.0
]

# integrate over the full torus with the standard volume element.
# f needs to have the signature T^4 -> Parameters -> Real.
function ∫_M(f)
    prob = IntegralProblem{false}(f, zeros(4), ones(4))
    sol = solve(prob, HCubatureJL(); reltol=1e-3, abstol=1e-3)
    sol.u
end

# energy
E(ρ) = ∫_M((xs, _) -> (K(ρ(xs))[1]^2 + K(ρ(xs))[2]^2 + K(ρ(xs))[3]^2) * u(ρ(xs)))

# The gradient vector field. At a cricitical point, this vanishes. In particular,
# there is no need to take the exterior derivative to search for a critical point.
ΣJᵢXᵢ(X) = sum([J₁ * X[1], J₂ * X[2], J₃ * X[3]])

# gradient operator
gradE(ρ, X) = d₁(A(ρ) * ΣJᵢXᵢ(X))


# equations for dρ = 0.
eqClosed(ρ) = d₂(ρ)[:] .~ 0

# equations for u > 0.
eqNonDegenerate(ρ, ϵᵤ) = [
    u(ρ) ≳ ϵᵤ
]

ι(X) = function (ρ)
    A(ρ) * X
end

# This is equivalent to ρ(X, ⋅) = dKᵢ, i.e. X is the Hamiltonian vector field of F.
eqHamilton(ρ, X, F) = ι(X)(ρ)[:] .~ d₀(F(ρ))[:]

# equations for ΣJᵢXᵢ = 0.
eqCritPoint(X) = ΣJᵢXᵢ(X)[:] .~ 0

# equations for higher energy.
eqEnergy(ρ) = [
    E(ρ) ≳ 2 * volM + 1
]

eqNonVanishing(X, ϵₓ) = norm(X, 1) ≳ ϵₓ

eqs = vcat(
    eqClosed([ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)]),
    # eqNonDegenerate([ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)], 0.1),
    eqHamilton([ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)],
        [X11(x0, x1, x2, x3), X12(x0, x1, x2, x3), X13(x0, x1, x2, x3), X14(x0, x1, x2, x3)], K₁),
    eqHamilton([ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)],
        [X21(x0, x1, x2, x3), X22(x0, x1, x2, x3), X23(x0, x1, x2, x3), X24(x0, x1, x2, x3)], K₂),
    eqHamilton([ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)],
        [X31(x0, x1, x2, x3), X32(x0, x1, x2, x3), X33(x0, x1, x2, x3), X34(x0, x1, x2, x3)], K₃),
    eqCritPoint([[X11(x0, x1, x2, x3), X12(x0, x1, x2, x3), X13(x0, x1, x2, x3), X14(x0, x1, x2, x3)],
        [X21(x0, x1, x2, x3), X22(x0, x1, x2, x3), X23(x0, x1, x2, x3), X24(x0, x1, x2, x3)],
        [X31(x0, x1, x2, x3), X32(x0, x1, x2, x3), X33(x0, x1, x2, x3), X34(x0, x1, x2, x3)],
    eqNonVanishing([X11(x0, x1, x2, x3), X12(x0, x1, x2, x3), X13(x0, x1, x2, x3), X14(x0, x1, x2, x3)], 0.1),
    eqNonVanishing([X21(x0, x1, x2, x3), X22(x0, x1, x2, x3), X23(x0, x1, x2, x3), X24(x0, x1, x2, x3)], 0.1),
    eqNonVanishing([X31(x0, x1, x2, x3), X32(x0, x1, x2, x3), X33(x0, x1, x2, x3), X34(x0, x1, x2, x3)], 0.1),
    ])
)

# periodic boundary conditions for the 4-torus
bcs = [
    ρ01(0.0, x1, x2, x3) ~ ρ01(1.0, x1, x2, x3),
    ρ01(x0, 0.0, x2, x3) ~ ρ01(x0, 1.0, x2, x3),
    ρ01(x0, x1, 0.0, x3) ~ ρ01(x0, x1, 1.0, x3),
    ρ01(x0, x1, x2, 0.0) ~ ρ01(x0, x1, x2, 1.0),
    ρ02(0.0, x1, x2, x3) ~ ρ02(1.0, x1, x2, x3),
    ρ02(x0, 0.0, x2, x3) ~ ρ02(x0, 1.0, x2, x3),
    ρ02(x0, x1, 0.0, x3) ~ ρ02(x0, x1, 1.0, x3),
    ρ02(x0, x1, x2, 0.0) ~ ρ02(x0, x1, x2, 1.0),
    ρ03(0.0, x1, x2, x3) ~ ρ03(1.0, x1, x2, x3),
    ρ03(x0, 0.0, x2, x3) ~ ρ03(x0, 1.0, x2, x3),
    ρ03(x0, x1, 0.0, x3) ~ ρ03(x0, x1, 1.0, x3),
    ρ03(x0, x1, x2, 0.0) ~ ρ03(x0, x1, x2, 1.0),
    ρ12(0.0, x1, x2, x3) ~ ρ12(1.0, x1, x2, x3),
    ρ12(x0, 0.0, x2, x3) ~ ρ12(x0, 1.0, x2, x3),
    ρ12(x0, x1, 0.0, x3) ~ ρ12(x0, x1, 1.0, x3),
    ρ12(x0, x1, x2, 0.0) ~ ρ12(x0, x1, x2, 1.0),
    ρ13(0.0, x1, x2, x3) ~ ρ13(1.0, x1, x2, x3),
    ρ13(x0, 0.0, x2, x3) ~ ρ13(x0, 1.0, x2, x3),
    ρ13(x0, x1, 0.0, x3) ~ ρ13(x0, x1, 1.0, x3),
    ρ13(x0, x1, x2, 0.0) ~ ρ13(x0, x1, x2, 1.0),
    ρ23(0.0, x1, x2, x3) ~ ρ23(1.0, x1, x2, x3),
    ρ23(x0, 0.0, x2, x3) ~ ρ23(x0, 1.0, x2, x3),
    ρ23(x0, x1, 0.0, x3) ~ ρ23(x0, x1, 1.0, x3),
    ρ23(x0, x1, x2, 0.0) ~ ρ23(x0, x1, x2, 1.0),
]

ixToSym = Dict(
    1 => :ρ01,
    2 => :ρ02,
    3 => :ρ03,
    4 => :ρ12,
    5 => :ρ13,
    6 => :ρ23,
    7 => :X11,
    8 => :X12,
    9 => :X13,
    10 => :X14,
    11 => :X21,
    12 => :X22,
    13 => :X23,
    14 => :X24,
    15 => :X31,
    16 => :X32,
    17 => :X33,
    18 => :X34,
)


input_ = length(domain)
n = 16
chains1 = NamedTuple((ixToSym[ix], Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1))) for ix in 1:6)
chains4 = NamedTuple((ixToSym[ix], Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1))) for ix in 7:18)
chains = merge(chains1, chains4)
chains0 = collect(chains)

strategy = QuasiRandomTraining(1000)
ps = map(c -> Lux.setup(Random.default_rng(), c)[1], chains) |> ComponentArray .|> Float64 |> gpu
discretization = PhysicsInformedNN(chains0, strategy, init_params=ps)
@named pdesystem = PDESystem(eqs, bcs, domain, [x0, x1, x2, x3],
    [ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3),
        X11(x0, x1, x2, x3), X12(x0, x1, x2, x3), X13(x0, x1, x2, x3), X14(x0, x1, x2, x3),
        X21(x0, x1, x2, x3), X22(x0, x1, x2, x3), X23(x0, x1, x2, x3), X24(x0, x1, x2, x3),
        X31(x0, x1, x2, x3), X32(x0, x1, x2, x3), X33(x0, x1, x2, x3), X34(x0, x1, x2, x3)]
)

prob = discretize(pdesystem, discretization)
sym_prob = symbolic_discretize(pdesystem, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

callback(ϵ::Float64) = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return l < ϵ
end

run(maxiters::Int=1; ϵ::Float64) = Optimization.solve(prob, Adam(0.01); callback=callback(ϵ), maxiters=maxiters)

function solToCoordFunction(sol::ComponentVector, sym_prob, sym::Symbol)
    weights = sol[sym]
    f(xs) = sym_prob.phi[sym_prob.dict_depvars[sym]](xs, weights)[1]

    return f
end

function ρ(sol::ComponentVector, sym_prob)
    f(xs) = [solToCoordFunction(sol, sym_prob, sym)(xs) for sym in [:ρ01, :ρ02, :ρ03, :ρ12, :ρ13, :ρ23]]
end

function X(sol::ComponentVector, sym_prob)
    X1(xs) = [solToCoordFunction(sol, sym_prob, sym)(xs) for sym in [:X11, :X12, :X13, :X14]]
    X2(xs) = [solToCoordFunction(sol, sym_prob, sym)(xs) for sym in [:X21, :X22, :X23, :X24]]
    X3(xs) = [solToCoordFunction(sol, sym_prob, sym)(xs) for sym in [:X31, :X32, :X33, :X34]]

    X(xs) = [X1(xs), X2(xs), X3(xs)]
    return X
end
