# FastTransforms.jl

[![Travis](https://travis-ci.org/MikaelSlevinsky/FastTransforms.jl.svg?branch=master)](https://travis-ci.org/MikaelSlevinsky/FastTransforms.jl) [![AppVeyor](https://ci.appveyor.com/api/projects/status/oba9qush15q3x8pb/branch/master?svg=true)](https://ci.appveyor.com/project/MikaelSlevinsky/fasttransforms-jl/branch/master) [![codecov](https://codecov.io/gh/MikaelSlevinsky/FastTransforms.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MikaelSlevinsky/FastTransforms.jl) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://MikaelSlevinsky.github.io/FastTransforms.jl/stable) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://MikaelSlevinsky.github.io/FastTransforms.jl/latest)

The aim of this package is to provide fast orthogonal polynomial transforms that are designed for expansions of functions with any degree of regularity. There are multiple approaches to the classical connection problem, though the user does not need to know the specifics.

One approach is based on the use of asymptotic formulae to relate the transforms to a small number of fast Fourier transforms. Another approach is based on a Toeplitz-dot-Hankel decomposition of the matrix of connection coefficients. Alternatively, the matrix of connection coefficients may be decomposed hierarchically à la Fast Multipole Method.

## The Chebyshev—Legendre Transform

The Chebyshev—Legendre transform allows the fast conversion of Chebyshev expansion coefficients to Legendre expansion coefficients and back.

```julia
julia> Pkg.add("FastTransforms")

julia> using FastTransforms

julia> c = rand(10001);

julia> leg2cheb(c);

julia> cheb2leg(c);

julia> norm(cheb2leg(leg2cheb(c))-c)
5.564168202018823e-13
```

The implementation separates pre-computation into a type of plan. This type is constructed with either `plan_leg2cheb` or `plan_cheb2leg`. Let's see how much faster it is if we pre-compute.

```julia
julia> p1 = plan_leg2cheb(c);

julia> p2 = plan_cheb2leg(c);

julia> @time leg2cheb(c);
  0.082615 seconds (11.94 k allocations: 31.214 MiB, 6.75% gc time)

julia> @time p1*c;
  0.004297 seconds (6 allocations: 78.422 KiB)

julia> @time cheb2leg(c);
  0.110388 seconds (11.94 k allocations: 31.214 MiB, 8.16% gc time)

julia> @time p2*c;
  0.004500 seconds (6 allocations: 78.422 KiB)
```

## The Chebyshev—Jacobi Transform

The Chebyshev—Jacobi transform allows the fast conversion of Chebyshev expansion coefficients to Jacobi expansion coefficients and back.

```julia
julia> c = rand(10001);

julia> @time norm(icjt(cjt(c,0.1,-0.2),0.1,-0.2)-c,Inf)
  0.258390 seconds (431 allocations: 6.278 MB)
1.4830359162942841e-12

julia> p1 = plan_cjt(c,0.1,-0.2);

julia> p2 = plan_icjt(c,0.1,-0.2);

julia> @time norm(p2*(p1*c)-c,Inf)
  0.244842 seconds (17 allocations: 469.344 KB)
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

julia> N = div((n+1)*(n+2),2);

julia> v = rand(N); # The length of v is the number of Padua points

julia> @time norm(ipaduatransform(paduatransform(v))-v)
  0.006571 seconds (846 allocations: 1.746 MiB)
3.123637691861415e-14

```

## The Spherical Harmonic Transform

Let `F` be a matrix of spherical harmonic expansion coefficients arranged by increasing order in absolute value, alternating between negative and positive orders. Then `sph2fourier` converts the representation into a bivariate Fourier series, and `fourier2sph` converts it back.
```julia
julia> F = sphrandn(Float64, 256, 256);

julia> G = sph2fourier(F);

julia> H = fourier2sph(G);

julia> norm(F-H)
4.950645831278297e-14

julia> F = sphrandn(Float64, 1024, 1024);

julia> G = sph2fourier(F; sketch = :none);
Pre-computing...100%|███████████████████████████████████████████| Time: 0:00:04

julia> H = fourier2sph(G; sketch = :none);
Pre-computing...100%|███████████████████████████████████████████| Time: 0:00:04

julia> norm(F-H)
1.1510623098225283e-12

```

As with other fast transforms, `plan_sph2fourier` saves effort by caching the pre-computation. Be warned that for dimensions larger than `1,000`, this is no small feat!

# References:

   [1]  B. Alpert and V. Rokhlin. <a href="http://dx.doi.org/10.1137/0912009">A fast algorithm for the evaluation of Legendre expansions</a>, *SIAM J. Sci. Stat. Comput.*, **12**:158—179, 1991.

   [2]  N. Hale and A. Townsend. <a href="http://dx.doi.org/10.1137/130932223">A fast, simple, and stable Chebyshev—Legendre transform using an asymptotic formula</a>, *SIAM J. Sci. Comput.*, **36**:A148—A167, 2014.

   [3]  J. Keiner. <a href="http://dx.doi.org/10.1137/070703065">Computing with expansions in Gegenbauer polynomials</a>, *SIAM J. Sci. Comput.*, **31**:2151—2171, 2009.

   [4]  D. Ruiz—Antolín and A. Townsend. <a href="https://arxiv.org/abs/1701.04492">A nonuniform fast Fourier transform based on low rank approximation</a>, arXiv:1701.04492, 2017.

   [5]  R. M. Slevinsky. <a href="https://doi.org/10.1093/imanum/drw070">On the use of Hahn's asymptotic formula and stabilized recurrence for a fast, simple, and stable Chebyshev—Jacobi transform</a>, in press at *IMA J. Numer. Anal.*, 2017.

   [6]  R. M. Slevinsky. <a href="https://arxiv.org/abs/1705.05448">Fast and backward stable transforms between spherical harmonic expansions and bivariate Fourier series</a>, arXiv:1705.05448, 2017.

   [7]  A. Townsend, M. Webb, and S. Olver. <a href="https://doi.org/10.1090/mcom/3277">Fast polynomial transforms based on Toeplitz and Hankel matrices</a>, in press at *Math. Comp.*, 2017.
