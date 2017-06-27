using FastTransforms, Base.Test

import FastTransforms: normalizecolumns!, maxcolnorm

@testset "Spherical harmonic API" begin
    n = VERSION < v"0.6.0-" ? 256 : 512
    A = sphrandn(Float64, n, n);
    normalizecolumns!(A);

    B = sph2fourier(A)
    C = fourier2sph(B)
    println("The backward difference between slow plan and original: ", maxcolnorm(A-C))

    P = plan_sph2fourier(A)
    B = P*A
    C = P\B

    println("The backward difference between slow plan and original: ", maxcolnorm(A-C))

    if VERSION ≥ v"0.6.0-"
        n = 1024
        A = sphrandn(Float64, n, n);
        normalizecolumns!(A);

        B = sph2fourier(A; sketch = :none)
        C = fourier2sph(B; sketch = :none)
        println("The backward difference between thin plan and original: ", maxcolnorm(A-C))

        P = plan_sph2fourier(A; sketch = :none)
        B = P*A
        C = P\B

        println("The backward difference between thin plan and original: ", maxcolnorm(A-C))
    end
end
