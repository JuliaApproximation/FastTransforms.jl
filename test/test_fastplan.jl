import FastTransforms: allranks

n = 511

A = sphrandn(Float64, n+1, n+1);
normalizecolumns!(A);
B = zero(A);
C = zero(A);
D = zero(A);
E = zero(A);

@time FP = FastSphericalHarmonicPlan(A);
@time SP = SlowSphericalHarmonicPlan(A);

@time A_mul_B!(B, SP, A);
@time A_mul_B!(C, FP, A);
@time At_mul_B!(D, SP, B);
@time At_mul_B!(E, FP, C);

println("The forward difference between slow and fast plans: ", maxcolnorm(B-C))
println("The backward difference between slow plan and original: ", maxcolnorm(A-D))
println("The backward difference between fast plan and original: ", maxcolnorm(A-E))
println("The backward difference between slow and fast plans: ", maxcolnorm(D-E))
