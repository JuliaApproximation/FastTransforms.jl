import FastTransforms: allranks, normalizecolumns!, maxcolnorm

m, n = VERSION < v"0.6.0-" ? (362, 300) : (724, 600)

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

println("The forward difference between slow and thin plans: ", maxcolnorm(B-C))
println("The backward difference between slow plan and original: ", maxcolnorm(A-D))
println("The backward difference between thin plan and original: ", maxcolnorm(A-E))
println("The backward difference between slow and thin plans: ", maxcolnorm(D-E))
