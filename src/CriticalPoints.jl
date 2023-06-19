using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters x0 x1 x2 x3
@variables ρ01(..) ρ02(..) ρ03(..) ρ12(..) ρ13(..) ρ23(..)
@variables u(..)
@variables K[1:3](..)
@variables A[1:4, 1:4](..)
@variables AInvPartialK[1:3](..)
@variables S[1:3, 1:4](..) STot[1:4](..)

∂₀ = Differential(x0)
∂₁= Differential(x1)
∂₂ = Differential(x2)
∂₃= Differential(x3)

J₁ = [
    0 -1 0 0;
    1 0 0 0;
    0 0 0 -1;
    0 0 1 0
]

J₂ = [
    0 0 -1 0;
    0 0 0 1;
    1 0 0 0;
    0 -1 0 0
]

J₃ = [
    0 0 0 -1;
    0 0 -1 0;
    0 1 0 0;
    1 0 0 0
]

A0(ρ01, ρ02, ρ03, ρ12, ρ13, ρ23) = [
      0   ρ01     ρ02     ρ03;
   -ρ01     0     ρ12     ρ13;
   -ρ02  -ρ12       0     ρ23;
   -ρ03  -ρ13    -ρ23       0;
]

#critical point equation
eq = [ 
    u(x0,x1,x2,x3) ~ ρ01(x0,x1,x2,x3) * ρ23(x0,x1,x2,x3) + ρ02(x0,x1,x2,x3)*ρ31(x0,x1,x2,x3) + ρ03(x0,x1,x2,x3)*ρ12(x0,x1,x2,x3),
    K[1](x0,x1,x2,x3) ~ 2(ρ01(x0,x1,x2,x3) + ρ23(x0,x1,x2,x3)) / u(x0,x1,x2,x3),
    K[2](x0,x1,x2,x3) ~ 2(ρ02(x0,x1,x2,x3) - ρ31(x0,x1,x2,x3)) / u(x0,x1,x2,x3),
    K[3](x0,x1,x2,x3) ~ 2(ρ03(x0,x1,x2,x3) + ρ12(x0,x1,x2,x3)) / u(x0,x1,x2,x3),
    A(x0, x1, x2, x3) ~ A0(ρ01(x0,x1,x2,x3), ρ02(x0,x1,x2,x3), ρ03(x0,x1,x2,x3), ρ12(x0,x1,x2,x3), ρ13(x0,x1,x2,x3), ρ23(x0,x1,x2,x3)),
    A(x0, x1, x2, x3) * AInvPartialK[1](x0, x1, x2, x3) ~ [∂₀(K[1](x0,x1,x2,x3)), ∂₁(K[1](x0,x1,x2,x3)), ∂₂(K[1](x0,x1,x2,x3)), ∂₃(K[1](x0,x1,x2,x3))],
    A(x0, x1, x2, x3) * AInvPartialK[2](x0, x1, x2, x3) ~ [∂₀(K[2](x0,x1,x2,x3)), ∂₁(K[2](x0,x1,x2,x3)), ∂₂(K[2](x0,x1,x2,x3)), ∂₃(K[2](x0,x1,x2,x3))],
    A(x0, x1, x2, x3) * AInvPartialK[3](x0, x1, x2, x3) ~ [∂₀(K[3](x0,x1,x2,x3)), ∂₁(K[3](x0,x1,x2,x3)), ∂₂(K[3](x0,x1,x2,x3)), ∂₃(K[3](x0,x1,x2,x3))],
    S[1](x0,x1,x2,x3) ~ A(x0, x1, x2, x3) * J₁ * AInvPartialK[1](x0, x1, x2, x3),
    S[2](x0,x1,x2,x3) ~ A(x0, x1, x2, x3) * J₂ * AInvPartialK[2](x0, x1, x2, x3),
    S[3](x0,x1,x2,x3) ~ A(x0, x1, x2, x3) * J₃ * AInvPartialK[3](x0, x1, x2, x3),
    # alternatively solve the inverse problem directly.
    # S[1](x0, x1, x2, x3) ~ A(x0, x1, x2, x3) * J₁ * [∂₀(K[1](x0,x1,x2,x3)), ∂₁(K[1](x0,x1,x2,x3)), ∂₂(K[1](x0,x1,x2,x3)), ∂₃(K[1](x0,x1,x2,x3))] / A(x0,x1,x2,x3),
    # S[2](x0, x1, x2, x3) ~ A(x0, x1, x2, x3) * J₂ * [∂₀(K[2](x0,x1,x2,x3)), ∂₁(K[2](x0,x1,x2,x3)), ∂₂(K[2](x0,x1,x2,x3)), ∂₃(K[2](x0,x1,x2,x3))] / A(x0,x1,x2,x3),
    # S[3](x0, x1, x2, x3) ~ A(x0, x1, x2, x3) * J₃ * [∂₀(K[3](x0,x1,x2,x3)), ∂₁(K[3](x0,x1,x2,x3)), ∂₂(K[3](x0,x1,x2,x3)), ∂₃(K[3](x0,x1,x2,x3))] / A(x0,x1,x2,x3),
    STot(x0,x1,x2,x3) ~ sum(S[:](x0,x1,x2,x3)),
    0.0 ~ ∂₀(STot[2](x0,x1,x2,x3)),
    0.0 ~ ∂₀(STot[3](x0,x1,x2,x3)),
    0.0 ~ ∂₀(STot[4](x0,x1,x2,x3)),
    0.0 ~ ∂₁(STot[3](x0,x1,x2,x3)),
    0.0 ~ ∂₁(STot[4](x0,x1,x2,x3)),
    0.0 ~ ∂₂(STot[4](x0,x1,x2,x3)),
]
