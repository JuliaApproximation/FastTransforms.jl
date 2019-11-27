# FastTransforms.jl

[![Travis](https://travis-ci.org/JuliaApproximation/FastTransforms.jl.svg?branch=master)](https://travis-ci.org/JuliaApproximation/FastTransforms.jl) [![AppVeyor](https://ci.appveyor.com/api/projects/status/oba9qush15q3x8pb/branch/master?svg=true)](https://ci.appveyor.com/project/MikaelSlevinsky/fasttransforms-jl/branch/master) [![codecov](https://codecov.io/gh/JuliaApproximation/FastTransforms.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/FastTransforms.jl) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaApproximation.github.io/FastTransforms.jl/stable) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaApproximation.github.io/FastTransforms.jl/latest)

`FastTransforms.jl` allows the user to conveniently work with orthogonal polynomials with degrees well into the millions.

This package provides a Julia wrapper for the [C library](https://github.com/MikaelSlevinsky/FastTransforms) of the same name. Additionally, all three types of nonuniform fast Fourier transforms are available, as well as the Padua transform.

## Installation

Installation, which uses [BinaryBuilder](https://github.com/JuliaPackaging/BinaryBuilder.jl) for Intel processors, may be as straightforward as:

```julia
pkg> add FastTransforms

julia> using FastTransforms, LinearAlgebra

```

## Fast orthogonal polynomial transforms

The 26 orthogonal polynomial transforms are listed in `FastTransforms.kind2string.(0:25)`. Univariate transforms may be planned with the standard normalization or with orthonormalization. For multivariate transforms, the standard normalization may be too severe for floating-point computations, so it is omitted. Here are two examples:

### The Chebyshev--Legendre transform

```julia
julia> c = rand(8192);

julia> leg2cheb(c);

julia> cheb2leg(c);

julia> norm(cheb2leg(leg2cheb(c; normcheb=true); normcheb=true)-c)/norm(c)
1.1866591414786334e-14

```

The implementation separates pre-computation into an `FTPlan`. This type is constructed with either `plan_leg2cheb` or `plan_cheb2leg`. Let's see how much faster it is if we pre-compute.

```julia
julia> p1 = plan_leg2cheb(c);

julia> p2 = plan_cheb2leg(c);

julia> @time leg2cheb(c);
  0.433938 seconds (9 allocations: 64.641 KiB)

julia> @time p1*c;
  0.005713 seconds (8 allocations: 64.594 KiB)

julia> @time cheb2leg(c);
  0.423865 seconds (9 allocations: 64.641 KiB)

julia> @time p2*c;
  0.005829 seconds (8 allocations: 64.594 KiB)

```

Furthermore, for orthogonal polynomial connection problems that are degree-preserving, we should expect to be able to apply the transforms in-place:

```julia
julia> lmul!(p1, c);

julia> lmul!(p2, c);

julia> ldiv!(p1, c);

julia> ldiv!(p2, c);

```

### The spherical harmonic transform

Let `F` be an array of spherical harmonic expansion coefficients with columns arranged by increasing order in absolute value, alternating between negative and positive orders. Then `sph2fourier` converts the representation into a bivariate Fourier series, and `fourier2sph` converts it back. Once in a bivariate Fourier series on the sphere, `plan_sph_synthesis` converts the coefficients to function samples on an equiangular grid that does not sample the poles, and `plan_sph_analysis` converts them back.

```julia
julia> F = sphrandn(Float64, 1024, 2047); # convenience method

julia> P = plan_sph2fourier(F);

julia> PS = plan_sph_synthesis(F);

julia> PA = plan_sph_analysis(F);

julia> @time G = PS*(P*F);
  0.090767 seconds (12 allocations: 31.985 MiB, 1.46% gc time)

julia> @time H = P\(PA*G);
  0.092726 seconds (12 allocations: 31.985 MiB, 1.63% gc time)

julia> norm(F-H)/norm(F)
2.1541073345177038e-15

```

Due to the structure of the spherical harmonic connection problem, these transforms may also be performed in-place with `lmul!` and `ldiv!`.

## Nonuniform fast Fourier transforms

The NUFFTs are implemented thanks to [Alex Townsend](https://github.com/ajt60gaibb):
 - `nufft1` assumes uniform samples and noninteger frequencies;
 - `nufft2` assumes nonuniform samples and integer frequencies;
 - `nufft3 ( = nufft)` assumes nonuniform samples and noninteger frequencies;
 - `inufft1` inverts an `nufft1`; and,
 - `inufft2` inverts an `nufft2`.

Here is an example:

```julia
julia> n = 10^4;

julia> c = complex(rand(n));

julia> ω = collect(0:n-1) + rand(n);

julia> nufft1(c, ω, eps());

julia> p1 = plan_nufft1(ω, eps());

julia> @time p1*c;
  0.002383 seconds (6 allocations: 156.484 KiB)

julia> x = (collect(0:n-1) + 3rand(n))/n;

julia> nufft2(c, x, eps());

julia> p2 = plan_nufft2(x, eps());

julia> @time p2*c;
  0.001478 seconds (6 allocations: 156.484 KiB)

julia> nufft3(c, x, ω, eps());

julia> p3 = plan_nufft3(x, ω, eps());

julia> @time p3*c;
  0.058999 seconds (6 allocations: 156.484 KiB)

```

## The Padua Transform

The Padua transform and its inverse are implemented thanks to [Michael Clarke](https://github.com/MikeAClarke). These are optimized methods designed for computing the bivariate Chebyshev coefficients by interpolating a bivariate function at the Padua points on `[-1,1]^2`.

```julia
julia> n = 200;

julia> N = div((n+1)*(n+2), 2);

julia> v = rand(N); # The length of v is the number of Padua points

julia> @time norm(ipaduatransform(paduatransform(v)) - v)/norm(v)
  0.007373 seconds (543 allocations: 1.733 MiB)
3.925164683252905e-16

```

# References:

   [1]  B. Alpert and V. Rokhlin. <a href="http://dx.doi.org/10.1137/0912009">A fast algorithm for the evaluation of Legendre expansions</a>, *SIAM J. Sci. Stat. Comput.*, **12**:158—179, 1991.

   [2]  N. Hale and A. Townsend. <a href="http://dx.doi.org/10.1137/130932223">A fast, simple, and stable Chebyshev—Legendre transform using an asymptotic formula</a>, *SIAM J. Sci. Comput.*, **36**:A148—A167, 2014.

   [3]  J. Keiner. <a href="http://dx.doi.org/10.1137/070703065">Computing with expansions in Gegenbauer polynomials</a>, *SIAM J. Sci. Comput.*, **31**:2151—2171, 2009.

   [4]  D. Ruiz—Antolín and A. Townsend. <a href="https://arxiv.org/abs/1701.04492">A nonuniform fast Fourier transform based on low rank approximation</a>, arXiv:1701.04492, 2017.

   [5]  R. M. Slevinsky. <a href="https://doi.org/10.1093/imanum/drw070">On the use of Hahn's asymptotic formula and stabilized recurrence for a fast, simple, and stable Chebyshev—Jacobi transform</a>, *IMA J. Numer. Anal.*, **38**:102—124, 2018.

   [6]  R. M. Slevinsky. <a href="https://doi.org/10.1016/j.acha.2017.11.001">Fast and backward stable transforms between spherical harmonic expansions and bivariate Fourier series</a>, *Appl. Comput. Harmon. Anal.*, **47**:585—606, 2019.

   [7]  R. M. Slevinsky, <a href="https://arxiv.org/abs/1711.07866">Conquering the pre-computation in two-dimensional harmonic polynomial transforms</a>, arXiv:1711.07866, 2017.

   [8]  A. Townsend, M. Webb, and S. Olver. <a href="https://doi.org/10.1090/mcom/3277">Fast polynomial transforms based on Toeplitz and Hankel matrices</a>, in press at *Math. Comp.*, 2017.
