# FastTransforms.jl

[![Build Status](https://travis-ci.org/MikaelSlevinsky/FastTransforms.jl.svg?branch=master)](https://travis-ci.org/MikaelSlevinsky/FastTransforms.jl) [![Documentation Status](https://readthedocs.org/projects/fasttransformsjl/badge/?version=latest)](http://fasttransformsjl.readthedocs.org/en/latest/?badge=latest)

The aim of this package is to provide a new class of fast transforms
based on the use of asymptotic formulae to relate the transforms to a small
number of fast Fourier transforms. This new class of fast transforms does not
require large pre-computation for fast execution, and they are designed
to work on expansions of functions with any degree of regularity.

The Chebyshev—Jacobi transform and its inverse are implemented. This
allows the fast conversion of Chebyshev expansion coefficients to Jacobi expansion
coefficients and back.
```julia
julia> Pkg.add("FastTransforms")

julia> using FastTransforms

julia> c = rand(10001);

julia> @time norm(icjt(cjt(c,0.1,-0.2),0.1,-0.2)-c,Inf)
  0.435853 seconds (507 allocations: 5.366 MB)
1.4830359162942841e-12

julia> p1 = plan_cjt(c,0.1,-0.2);

julia> p2 = plan_icjt(c,0.1,-0.2);

julia> @time norm(p2*(p1*c)-c,Inf)
  0.396803 seconds (101 allocations: 473.281 KB)
1.4830359162942841e-12

```

The design and implementation is analogous to FFTW: there is a type `ChebyshevJacobiPlan`
that stores pre-planned optimized DCT-I and DST-I plans, recurrence coefficients,
and temporary arrays to allow the execution of either the `cjt` or the `icjt` allocation-free.
This type is constructed with either `plan_cjt` or `plan_icjt`. Composition of transforms
allows the Jacobi—Jacobi transform, computed via `jjt`. The remainder in Hahn's asymptotic expansion
is valid for the half-open square `(α,β) ∈ (-1/2,1/2]^2`. Therefore, the fast transform works best
when the parameters are inside. If the parameters `(α,β)` are not exceptionally beyond the square,
then increment/decrement operators are used with linear complexity (and linear conditioning) in the degree.

# References:

   1.	N. Hale and A. Townsend. <a href="http://dx.doi.org/10.1137/130932223">A fast, simple, and stable Chebyshev—Legendre transform using and asymptotic formula</a>, SIAM J. Sci. Comput., 36:A148—A167, 2014.

   2.	R. M. Slevinsky. <a href="http://arxiv.org/abs/1602.02618">On the use of Hahn's asymptotic formula and stabilized recurrence for a fast, simple, and stable Chebyshev—Jacobi transform</a>, arXiv:1602.02618, 2016.

   3.	A. Townsend, M. Webb, and S. Olver. <a href="http://arxiv.org/abs/1604.07486">Fast polynomial transforms based on Toeplitz and Hankel matrices</a>, arXiv:1604.07486, 2016.
