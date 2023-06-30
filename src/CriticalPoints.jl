using NeuralPDE: DomainSets
using NeuralPDE, Lux, CUDA, Random, ComponentArrays, Optimization, OptimizationOptimisers, Integrals
using LinearAlgebra
import ModelingToolkit: Interval

@parameters x0 x1 x2 x3
@variables ρ01(..) ρ02(..) ρ03(..) ρ12(..) ρ13(..) ρ23(..) K1(..) K2(..) K3(..)

# the 4-torus
domain = [
    x0 ∈ Interval(0, 2π),
    x1 ∈ Interval(0, 2π),
    x2 ∈ Interval(0, 2π),
    x3 ∈ Interval(0, 2π),
]

volM = (2π)^4

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
    2*∂₀(ρ[4]) - 2*∂₁(ρ[2]) + 2*∂₂(ρ[1]),
    #(0,1,3) + (3,0,1) + (1,3,0) - (1,0,3) - (0,3,1) - (3,1,0)
    2*∂₀(ρ[5]) - 2*∂₁(ρ[3]) + 2*∂₃(ρ[1]),
    #(0,2,3) + (3,0,2) + (2,3,0) - (2,0,3) - (0,3,2) - (3,2,0)
    2*∂₀(ρ[6]) - 2* ∂₂(ρ[3]) + 2*∂₃(ρ[2]),
    #(1,2,3) + (3,1,2) + (2,3,1) - (2,1,3) - (1,3,2) - (3,2,1)
    2*∂₁(ρ[6]) - 2*∂₂(ρ[5]) + 2*∂₃(ρ[4]),
]

J₁ = [
    0 -1 0 0;
    1 0 0 0;
    0 0 0 -1;
    00 0 1 0;
] .|> Float64

J₂ = [
    0 0 -1 0;
    0 0 0 1;
    1 0 0 0;
    0 -1 0 0;
] .|> Float64

J₃ = [
    0 0 0 -1;
    0 0 -1 0;
    0 1 0 0;
    1 0 0 0;
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
    0.0 ρ[1] ρ[2] ρ[3];
    -ρ[1] 0.0 ρ[4] ρ[5];
    -ρ[2] -ρ[4] 0.0 ρ[6];
    -ρ[3] -ρ[5] -ρ[6] 0.0;
]

# integrate over the full torus with the standard volume element.
function ∫_M(f) 
    IntegralProblem(f, zeros(4), 2π *ones(4))
    sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
    sol.u
end

# energy
E(ρ) = ∫_M((K(ρ)[1]^2 + K(ρ)[2]^2 + K(ρ)[3]^2)*u(ρ))

# The gradient vector field. At a cricitical point, this vanishes. In particular,
# there is no need to take the exterior derivative to search for a critical point.
#
# TODO: Make sure we use the best algorithm for solve the linear problem AXᵢ = dKᵢ.
# This is equivalent to ρ(Xᵢ, ⋅) = dKᵢ, is the Hamiltonian vector field of Kᵢ.
function ΣJᵢXᵢ(ρ, K) 
    B = A(ρ)
    J₁ * B \ d₀(K[1]) + J₂ * B \ d₀(K[2]) + J₃ * B \ d₀(K[3])
end
# @register_symbolic(ΣJᵢXᵢ(ρ, K))

# gradient operator
gradE(ρ) = d₁(A(ρ)*ΣJᵢXᵢ(ρ))


# equations for dρ = 0.
eqClosed(ρ) = [
    norm(d₂(ρ)) ~ 0
]

# equations for u > 0.
ϵ = 0.1
eqNonDegenerate(ρ) = [
    u(ρ) ≳ ϵ,
]

# equations for ΣJᵢXᵢ = 0.
eqCritPoint(ρ, K) = [
    K[1] ~ K₁(ρ)   
    K[2] ~ K₂(ρ)  
    K[3] ~ K₃(ρ)  
    norm(ΣJᵢXᵢ(ρ, K)) ~ 0
]

# equations for higher energy.
eqEnergy(ρ) = [
    E(ρ) ≳ 2 * volM + 1
]

eqs = vcat(
    eqClosed([ρ01(x0,x1,x2,x3),ρ02(x0,x1,x2,x3),ρ03(x0,x1,x2,x3),ρ12(x0,x1,x2,x3),ρ13(x0,x1,x2,x3),ρ23(x0,x1,x2,x3)]),
    eqNonDegenerate([ρ01(x0,x1,x2,x3),ρ02(x0,x1,x2,x3),ρ03(x0,x1,x2,x3),ρ12(x0,x1,x2,x3),ρ13(x0,x1,x2,x3),ρ23(x0,x1,x2,x3)]),
    eqCritPoint([ρ01(x0,x1,x2,x3),ρ02(x0,x1,x2,x3),ρ03(x0,x1,x2,x3),ρ12(x0,x1,x2,x3),ρ13(x0,x1,x2,x3),ρ23(x0,x1,x2,x3)], [K1(x0,x1,x2,x3), K2(x0,x1,x2,x3), K3(x0,x1,x2,x3)]),
)

# periodic boundary conditions for the 4-torus
bcs = [
    ρ01(0.0, x1, x2, x3) ~ ρ01(2π, x1, x2, x3),
    ρ01(x0, 0.0, x2, x3) ~ ρ01(x0, 2π, x2, x3),
    ρ01(x0, x1, 0.0, x3) ~ ρ01(x0, x1, 2π, x3),
    ρ01(x0, x1, x2, 0.0) ~ ρ01(x0, x1, x2, 2π), 
    ρ02(0.0, x1, x2, x3) ~ ρ02(2π, x1, x2, x3),
    ρ02(x0, 0.0, x2, x3) ~ ρ02(x0, 2π, x2, x3),
    ρ02(x0, x1, 0.0, x3) ~ ρ02(x0, x1, 2π, x3),
    ρ02(x0, x1, x2, 0.0) ~ ρ02(x0, x1, x2, 2π), 
    ρ03(0.0, x1, x2, x3) ~ ρ03(2π, x1, x2, x3),
    ρ03(x0, 0.0, x2, x3) ~ ρ03(x0, 2π, x2, x3),
    ρ03(x0, x1, 0.0, x3) ~ ρ03(x0, x1, 2π, x3),
    ρ03(x0, x1, x2, 0.0) ~ ρ03(x0, x1, x2, 2π), 
    ρ12(0.0, x1, x2, x3) ~ ρ12(2π, x1, x2, x3),
    ρ12(x0, 0.0, x2, x3) ~ ρ12(x0, 2π, x2, x3),
    ρ12(x0, x1, 0.0, x3) ~ ρ12(x0, x1, 2π, x3),
    ρ12(x0, x1, x2, 0.0) ~ ρ12(x0, x1, x2, 2π), 
    ρ13(0.0, x1, x2, x3) ~ ρ13(2π, x1, x2, x3),
    ρ13(x0, 0.0, x2, x3) ~ ρ13(x0, 2π, x2, x3),
    ρ13(x0, x1, 0.0, x3) ~ ρ13(x0, x1, 2π, x3),
    ρ13(x0, x1, x2, 0.0) ~ ρ13(x0, x1, x2, 2π), 
    ρ23(0.0, x1, x2, x3) ~ ρ23(2π, x1, x2, x3),
    ρ23(x0, 0.0, x2, x3) ~ ρ23(x0, 2π, x2, x3),
    ρ23(x0, x1, 0.0, x3) ~ ρ23(x0, x1, 2π, x3),
    ρ23(x0, x1, x2, 0.0) ~ ρ23(x0, x1, x2, 2π),
]


input_ = length(domain)
n = 1
chains = [Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1)) for _ in 1:9]

strategy = QuasiRandomTraining(10)
ps = [Lux.setup(Random.default_rng(), c)[1] |> ComponentArray |> gpu .|> Float64 for c in chains]
discretization = PhysicsInformedNN(chains, strategy, init_params = ps)
@named pdesystem = PDESystem(eqs, bcs, domain, [x0, x1, x2, x3], [ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3), K1(x0, x1, x2, x3), K2(x0, x1, x2, x3), K3(x0, x1, x2, x3)])
prob = discretize(pdesystem, discretization)
sym_prob = symbolic_discretize(pdesystem, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

callback = function (p, l)
    println("loss: ", l)
    println("pde_losses: ", map(l_ -> l_(p), pde_inner_loss_functions))
    println("bcs_losses: ", map(l_ -> l_(p), bcs_inner_loss_functions))
    return false
end


run(maxiters::Int = 1) = Optimization.solve(prob, Adam(0.01); callback=callback, maxiters=maxiters)