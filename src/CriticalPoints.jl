using NeuralPDE: DomainSets
using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters x0 x1 x2 x3
@variables ρ01(..) ρ02(..) ρ03(..) ρ12(..) ρ13(..) ρ23(..)

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
]

J₂ = [
    0 0 -1 0;
    0 0 0 1;
    1 0 0 0;
    0 -1 0 0;
]

J₃ = [
    0 0 0 -1;
    0 0 -1 0;
    0 1 0 0;
    1 0 0 0;
]

J = [
    J₁,
    J₂,
    J₃,
]

ρ(x0,x1,x2,x3) = [
    #(0,1)
    ρ01(x0,x1,x2,x3),
    #(0,2)
    ρ02(x0,x1,x2,x3),
    #(0,3)
    ρ03(x0,x1,x2,x3),
    #(1,2)
    ρ12(x0,x1,x2,x3),
    #(1,3)
    ρ13(x0,x1,x2,x3),
    #(2,3)
    ρ23(x0,x1,x2,x3),
]

u(x0, x1, x2, x3) = ρ01(x0, x1, x2, x3) * ρ23(x0, x1, x2, x3) - ρ02(x0, x1, x2, x3) * ρ13(x0, x1, x2, x3) + ρ03(x0, x1, x2, x3) * ρ12(x0, x1, x2, x3)

K₁(x0,x1,x2,x3) = 2(ρ01(x0, x1, x2, x3) + ρ23(x0, x1, x2, x3)) / u(x0, x1, x2, x3)
K₂(x0,x1,x2,x3) = 2(ρ02(x0, x1, x2, x3) - ρ13(x0, x1, x2, x3)) / u(x0, x1, x2, x3)
K₃(x0,x1,x2,x3) = 2(ρ03(x0, x1, x2, x3) + ρ12(x0, x1, x2, x3)) / u(x0, x1, x2, x3)

K(x0, x1, x2, x3) = [
    K₁(x0,x1,x2,x3),
    K₂(x0,x1,x2,x3),
    K₃(x0,x1,x2,x3),
]

# g(A⋅, ⋅) = ρ(⋅, ⋅)
A(x0, x1, x2, x3) = [
    0.0 ρ01(x0, x1, x2, x3) ρ02(x0, x1, x2, x3) ρ03(x0, x1, x2, x3);
    -ρ01(x0, x1, x2, x3) 0.0 ρ12(x0, x1, x2, x3) ρ13(x0, x1, x2, x3);
    -ρ02(x0, x1, x2, x3) -ρ12(x0, x1, x2, x3) 0.0 ρ23(x0, x1, x2, x3);
    -ρ03(x0, x1, x2, x3) -ρ13(x0, x1, x2, x3) -ρ23(x0, x1, x2, x3) 0.0;
]

# integrate over the full torus with the standard volume element.
∫_M = Integral((x0,x1,x2,x3) in DomainSets.ProductDomain(DomainSets.ClosedInterval(0,2π), 
                                                            DomainSets.ClosedInterval(0,2π), 
                                                            DomainSets.ClosedInterval(0,2π), 
                                                            DomainSets.ClosedInterval(0,2π)))

# energy
# TODO: What quadrature will this be using? Is this under my control?
E = ∫_M((K(x0,x1,x2,x3)[1]^2 + K(x0,x1,x2,x3)[2]^2 + K(x0,x1,x2,x3)[3]^2)*u(x0,x1,x2,x3))

# fundamental class a[M]
# TODO: What quadrature will this be using? Is this under my control?
aM = ∫_M(u(x0,x1,x2,x3))

# The gradient vector field. At a cricitical point, this vanishes. In particular,
# there is no need to take the exterior derivative to search for a critical point.
#
# TODO: Make sure we use the best algorithm for solve the linear problem AXᵢ = dKᵢ.
# This is equivalent to ρ(Xᵢ, ⋅) = dKᵢ, is the Hamiltonian vector field of Kᵢ.
ΣJᵢXᵢ(x0,x1,x2,x3) =
    J₁ * A(x0, x1, x2, x3) \ d₀(K₁(x0,x1,x2,x3))
    + J₂ * A(x0, x1, x2, x3) \ d₀(K₂(x0,x1,x2,x3))
    + J₃ * A(x0, x1, x2, x3) \ d₀(K₃(x0,x1,x2,x3))

# ∑ = sum
# gradE1(x0, x1, x2, x3) = A(x0,x1,x2,x3) * ∑([J[i] * A(x0,x1,x2,x3) \ d₀(K(x0,x1,x2,x3)[i]) for i in [1,2,3]])

# gradient operator
gradE(x0,x1,x2,x3) = d₁(A(x0,x1,x2,x3)*ΣJᵢXᵢ(x0,x1,x2,x3))


# equations for dρ = 0.
eqClosed = [
    d₂(ρ(x0,x1,x2,x3))[1] ~ 0,
    d₂(ρ(x0,x1,x2,x3))[2] ~ 0,
    d₂(ρ(x0,x1,x2,x3))[3] ~ 0,
    d₂(ρ(x0,x1,x2,x3))[4] ~ 0,
]

# equations for u > 0.
ϵ = 0.1
eqNonDegenerate = [
    u(x0,x1,x2,x3) ≳ ϵ,
]

# equations for ΣJᵢXᵢ = 0.
eqCritPoint = [
    ΣJᵢXᵢ(x0,x1,x2,x3)[1] ~ 0,
    ΣJᵢXᵢ(x0,x1,x2,x3)[2] ~ 0,
    ΣJᵢXᵢ(x0,x1,x2,x3)[3] ~ 0,
    ΣJᵢXᵢ(x0,x1,x2,x3)[4] ~ 0,
]

# equations for higher energy.
eqEnergy = [
    E ≳ 2 * volM + 1
]

eqs = vcat(
    eqClosed,
    eqNonDegenerate,
    eqCritPoint,
# TODO: energy and cohomology conditions could also be part of the boundary conditions.
    # eqEnergy,
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
n = 15
chain = [Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1)) for _ in 1:6]

strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, strategy)
@named pdesystem = PDESystem(eqs, bcs, domain, [x0, x1, x2, x3], [ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)])
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


run(maxiters::Int = 1) = Optimization.solve(prob, BFGS(); callback=callback, maxiters=maxiters)