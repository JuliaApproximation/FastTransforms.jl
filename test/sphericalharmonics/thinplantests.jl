using FastTransforms, Compat
using Compat.Test

import FastTransforms: normalizecolumns!, maxcolnorm

@testset "Thin plan" begin
    m, n = 724, 600

    A = sphrandn(Float64, m, n);
    normalizecolumns!(A);
    B = zero(A);
    C = zero(A);
    D = zero(A);
    E = zero(A);

    @time TP = ThinSphericalHarmonicPlan(A; sketch = :none);
    @time SP = SlowSphericalHarmonicPlan(A);

    @time A_mul_B!(B, SP, A);
    @time A_mul_B!(C, TP, A);
    @time At_mul_B!(D, SP, B);
    @time At_mul_B!(E, TP, C);

    @test maxcolnorm(B-C) < 10eps()*max(m,n)
    @test maxcolnorm(A-D) < 10eps()*max(m,n)
    @test maxcolnorm(A-E) < 10eps()*max(m,n)
    @test maxcolnorm(D-E) < 10eps()*max(m,n)

    println("The forward difference between slow and thin plans: ", maxcolnorm(B-C))
    println("The backward difference between slow plan and original: ", maxcolnorm(A-D))
    println("The backward difference between thin plan and original: ", maxcolnorm(A-E))
    println("The backward difference between slow and thin plans: ", maxcolnorm(D-E))
end
