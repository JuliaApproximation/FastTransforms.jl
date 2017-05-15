using FastTransforms, Base.Test

srand(0)

include("sphericalharmonictestfunctions.jl")

println("Testing slow plan")
include("test_slowplan.jl")
println("Testing fast plan")
include("test_fastplan.jl")
println("Testing thin plan")
include("test_thinplan.jl")

println("Testing API")

n = 511
A = sphrandn(Float64, n+1, n+1);
normalizecolumns!(A);

B = sph2fourier(A; sketch = :none)
C = fourier2sph(B; sketch = :none)
println("The backward difference between slow plan and original: ", maxcolnorm(A-C))

P = plan_sph2fourier(A; sketch = :none)
B = P*A
C = P\B

println("The backward difference between slow plan and original: ", maxcolnorm(A-C))

n = 1023
A = sphrandn(Float64, n+1, n+1);
normalizecolumns!(A);

B = sph2fourier(A; sketch = :none)
C = fourier2sph(B; sketch = :none)
println("The backward difference between slow plan and original: ", maxcolnorm(A-C))

P = plan_sph2fourier(A; sketch = :none)
B = P*A
C = P\B

println("The backward difference between thin plan and original: ", maxcolnorm(A-C))
