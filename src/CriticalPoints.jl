using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters x0 x1 x2 x3
@variables ρ01(..) ρ02(..) ρ03(..) ρ12(..) ρ13(..) ρ23(..)

∂₀ = Differential(x0)
∂₁ = Differential(x1)
∂₂ = Differential(x2)
∂₃ = Differential(x3)

d(ρ) = [
    ∂₀(ρ[2]),
    ∂₀(ρ[3]),
    ∂₀(ρ[4]),
    ∂₁(ρ[3]),
    ∂₁(ρ[4]),
    ∂₂(ρ[4]),
]


J₁ = [
    0 -1 0 0
    1 0 0 0
    0 0 0 -1
    0 0 1 0
]

J₂ = [
    0 0 -1 0
    0 0 0 1
    1 0 0 0
    0 -1 0 0
]

J₃ = [
    0 0 0 -1
    0 0 -1 0
    0 1 0 0
    1 0 0 0
]

u(x0, x1, x2, x3) = ρ01(x0, x1, x2, x3) * ρ23(x0, x1, x2, x3) - ρ02(x0, x1, x2, x3) * ρ13(x0, x1, x2, x3) + ρ03(x0, x1, x2, x3) * ρ12(x0, x1, x2, x3)

K(x0, x1, x2, x3) = [
    2(ρ01(x0, x1, x2, x3) + ρ23(x0, x1, x2, x3)) / u(x0, x1, x2, x3),
    2(ρ02(x0, x1, x2, x3) - ρ13(x0, x1, x2, x3)) / u(x0, x1, x2, x3),
    2(ρ03(x0, x1, x2, x3) + ρ12(x0, x1, x2, x3)) / u(x0, x1, x2, x3)
]

A(x0, x1, x2, x3) = [
    0 ρ01(x0, x1, x2, x3) ρ02(x0, x1, x2, x3) ρ03(x0, x1, x2, x3)
    -ρ01(x0, x1, x2, x3) 0 ρ12(x0, x1, x2, x3) ρ13(x0, x1, x2, x3)
    -ρ02(x0, x1, x2, x3) -ρ12(x0, x1, x2, x3) 0 ρ23(x0, x1, x2, x3)
    -ρ03(x0, x1, x2, x3) -ρ13(x0, x1, x2, x3) -ρ23(x0, x1, x2, x3) 0
]

▿E(x0, x1, x2, x3) = d(A(x0, x1, x2, x3) * J₁ * A(x0, x1, x2, x3) \ [∂₀(K(x0, x1, x2, x3)[1]), ∂₁(K(x0, x1, x2, x3)[1]), ∂₂(K(x0, x1, x2, x3)[1]), ∂₃(K(x0, x1, x2, x3)[1])]
                       + A(x0, x1, x2, x3) * J₂ * A(x0, x1, x2, x3) \ [∂₀(K(x0, x1, x2, x3)[2]), ∂₁(K(x0, x1, x2, x3)[2]), ∂₂(K(x0, x1, x2, x3)[2]), ∂₃(K(x0, x1, x2, x3)[2])]
                       + A(x0, x1, x2, x3) * J₃ * A(x0, x1, x2, x3) \ [∂₀(K(x0, x1, x2, x3)[3]), ∂₁(K(x0, x1, x2, x3)[3]), ∂₂(K(x0, x1, x2, x3)[3]), ∂₃(K(x0, x1, x2, x3)[3])]
)

#critical point equation
eqs = [▿E(x0, x1, x2, x3) ~ zeros(6)]

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
    ρ23(x0, x1, x2, 0.0) ~ ρ23(x0, x1, x2, 2π)
]

# # the 4-torus
domains = [
    x0 ∈ Interval(0, 2π),
    x1 ∈ Interval(0, 2π),
    x2 ∈ Interval(0, 2π),
    x3 ∈ Interval(0, 2π),
]

input_ = length(domains)
n = 15
chain = [Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1)) for _ in 1:6]

strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chain, strategy)
@named pdesystem = PDESystem(eqs, bcs, domains, [x0, x1, x2, x3], [ρ01(x0, x1, x2, x3), ρ02(x0, x1, x2, x3), ρ03(x0, x1, x2, x3), ρ12(x0, x1, x2, x3), ρ13(x0, x1, x2, x3), ρ23(x0, x1, x2, x3)])
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


run() = Optimization.solve(prob, BFGS(); callback=callback, maxiters=5000)