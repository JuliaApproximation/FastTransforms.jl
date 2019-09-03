using FastTransforms, Test

import FastTransforms: normalizecolumns!, maxcolnorm

@testset "Fast plan" begin
    m, n = 362, 300

    A = sphrandn(Float64, m, n);
    normalizecolumns!(A);
    B = zero(A);
    C = zero(A);
    D = zero(A);
    E = zero(A);

    @time FP = FastSphericalHarmonicPlan(A; sketch = :none);
    @time SP = SlowSphericalHarmonicPlan(A);

    @time mul!(B, SP, A);
    @time mul!(C, FP, A);
    @time mul!(D, transpose(SP), B);
    @time mul!(E, transpose(FP), C);

    @test maxcolnorm(B-C) < 10eps()*max(m,n)
    @test maxcolnorm(A-D) < 10eps()*max(m,n)
    @test maxcolnorm(A-E) < 10eps()*max(m,n)
    @test maxcolnorm(D-E) < 10eps()*max(m,n)

    println("The forward difference between slow and fast plans: ", maxcolnorm(B-C))
    println("The backward difference between slow plan and original: ", maxcolnorm(A-D))
    println("The backward difference between fast plan and original: ", maxcolnorm(A-E))
    println("The backward difference between slow and fast plans: ", maxcolnorm(D-E))
end
