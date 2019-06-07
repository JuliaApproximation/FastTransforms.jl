using FastTransforms, Compat
using Compat.Test

import FastTransforms: normalizecolumns!, maxcolnorm

@testset "Spherical harmonic API" begin
    let n = 512
        A = sphrandn(Float64, n, n);
        normalizecolumns!(A);

        B = sph2fourier(A)
        C = fourier2sph(B)
        println("The backward difference between slow plan and original: ", maxcolnorm(A-C))

        P = plan_sph2fourier(A)
        B = P*A
        C = P\B

        println("The backward difference between slow plan and original: ", maxcolnorm(A-C))


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
