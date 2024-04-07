using NeuralPDE: DomainSets
using NeuralPDE, Lux, Random, ComponentArrays, Optimization, OptimizationOptimisers, Integrals
# using LuxCUDA, CUDA
using LinearAlgebra
using Distributed
using SharedArrays
using JLD2
using Base.Threads
import ModelingToolkit: Interval

@parameters x0 x1
@variables f0(..) f1(..) f2(..) f3(..)

# Differential of an embedding of a 2-torus.
D(f) = Symbolics.jacobian(f, [x0,x1])

domain = [
    x0 ∈ Interval(0.0, 1.0),
    x1 ∈ Interval(0.0, 1.0),
]

A(ρ) = [
    0.0 ρ[1] ρ[2] ρ[3]
    -ρ[1] 0.0 ρ[4] ρ[5]
    -ρ[2] -ρ[4] 0.0 ρ[6]
    -ρ[3] -ρ[5] -ρ[6] 0.0
]

# periodic boundary conditions for an embedded 2-torus
bcs = [
    f0(0.0, x1) ~ f0(1.0, x1),
    f0(x0, 0.0) ~ f0(x0, 1.0),
    f1(0.0, x1) ~ f1(1.0, x1),
    f1(x0, 0.0) ~ f1(x0, 1.0),
    f2(0.0, x1) ~ f2(1.0, x1),
    f2(x0, 0.0) ~ f2(x0, 1.0),
    f3(0.0, x1) ~ f3(1.0, x1),
    f3(x0, 0.0) ~ f3(x0, 1.0),
]

# The pullback of a symplectic form under an embedding of the 2-torus.
function pullback(f, ρ)
    pb = transpose(D(f)) * A(ρ(f)) * D(f)
end

# equation for a Lagrangian embedding for a given ρ.
eqLagragian(f, ρ) = det(pullback(f, ρ)) ~ 0.0

eqs(ρ) = [
    let f = [f0(x0, x1), f1(x0, x1), f2(x0, x1), f3(x0, x1)]
        eqLagragian(f, ρ)
    end
]

energies = [
    let f = [f0(x0, x1), f1(x0, x1), f2(x0, x1), f3(x0, x1)]
    1 / norm(D(f))
    end
]

ixToSym = Dict(
    1 => :f0,
    2 => :f1,
    3 => :f2,
    4 => :f3,
)

pdesystem(ρ) = PDESystem(name=:pdesystem, eqs(ρ), bcs, domain, [x0, x1],
    [f0(x0, x1), f1(x0, x1), f2(x0, x1), f3(x0, x1)])

strategy = QuadratureTraining() #QuasiRandomTraining(1000)
input_ = length(domain)
n = 16
chains = NamedTuple((ixToSym[ix], Lux.Chain(Dense(input_, n, Lux.σ), Dense(n, n, Lux.σ), Dense(n, 1))) for ix in 1:4)

discretization = PhysicsInformedNN(collect(chains), strategy; additional_symb_loss=energies)
prob(ρ) = discretize(pdesystem(ρ), discretization)
sym_prob(ρ) = symbolic_discretize(pdesystem(ρ), discretization)
pde_inner_loss_functions(ρ) = sym_prob(ρ).loss_functions.pde_loss_functions
bcs_inner_loss_functions(ρ) = sym_prob(ρ).loss_functions.bc_loss_functions
asl_inner_loss_functions(ρ) = sym_prob(ρ).loss_functions.asl_loss_functions

# run on one process. use @everywhere run(...)
function runL(ρ; ϵ::Float64=2e-3, maxiters::Int=1, fp::String="solution")
    ps = map(c -> Lux.setup(Random.default_rng(), c)[1], chains) |> ComponentArray .|> Float64 # |> gpu
    prob1 = remake(prob(ρ); u0=ComponentVector(depvar=ps))

    callback(ϵ::Float64) = function (f, l)
        println("loss: ", l)
        pde_losses = map(l_ -> l_(f), pde_inner_loss_functions(ρ))
        println("pde_losses: ", pde_losses)
        println("bcs_losses: ", map(l_ -> l_(f), bcs_inner_loss_functions(ρ)))
        println("asl_losses: ", map(l_ -> l_(f), asl_inner_loss_functions(ρ)))
        return sum(pde_losses) < ϵ || (l > 10e4)
    end
    sol = Optimization.solve(prob1, Adam(0.01); callback=callback(ϵ), maxiters=maxiters)
    depvar = sol.u.depvar |> cpu
    JLD2.save_object("sol$(Distributed.myid()).jld2", depvar)
end
