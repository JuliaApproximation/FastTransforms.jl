# Harmonic polynomial transforms

The transforms on this page convert an expansion in orthonormal polynomials on a
domain, such as spherical harmonics on the sphere or Zernike polynomials on the
disk, to an expansion in a tensor-product basis of Chebyshev polynomials and
Fourier modes, and back. They wrap the C library
[FastTransforms](https://github.com/MikaelSlevinsky/FastTransforms), whose
[documentation](https://MikaelSlevinsky.github.io/FastTransforms) is the
reference for the definitions of the polynomials.

This page describes how the arrays of coefficients are laid out in Julia, and
what the transforms require of them.

## Summary

| Domain | Forward | Inverse | Size of the array | Maximum degree |
|:-------|:--------|:--------|:------------------|:---------------|
| Sphere | [`sph2fourier`](@ref) | [`fourier2sph`](@ref) | `n × (2n-1)` | `n-1` |
| Sphere, with spin | [`plan_spinsph2fourier`](@ref) | | `n × (2n-1)`, complex | `n-1` |
| Disk | [`disk2cxf`](@ref) | [`cxf2disk`](@ref) | `n × (4n-3)` | `2n-2` |
| Annulus | [`ann2cxf`](@ref) | [`cxf2ann`](@ref) | `n × (4n-3)` | `2n-2` |
| Rectangularized disk | [`rectdisk2cheb`](@ref) | [`cheb2rectdisk`](@ref) | `n × n` | `n-1` |
| Triangle | [`tri2cheb`](@ref) | [`cheb2tri`](@ref) | `n × n` | `n-1` |
| Tetrahedron | [`tet2cheb`](@ref) | [`cheb2tet`](@ref) | `n × n × n` | `n-1` |

Three things hold for all of them, and each is explained below:

1. The size `n` is the number of rows of the array. It is not the degree.
2. Only some of the entries of the input store a coefficient.
3. The other entries must be zero. They are not ignored.

## The size of the arrays

Every transform is determined by the number of rows `n` of the array. This is
the `n` in the documentation of the C library, where it is also called `N`. It
counts coefficients, so the smallest problem is `n = 1`, a `1 × 1` array that
stores the coefficient of the constant:

```julia
julia> disk2cxf(fill(1.0, 1, 1))
1×1 Matrix{Float64}:
 1.4142135623730951
```

There is no `n = 0` case: the number of columns, `4n-3` on the disk, would be
negative.

The number of columns follows from `n` as in the table above. On the sphere, the
disk, and the annulus, the columns are ordered by the order `m` of the Fourier
mode, `m = 0, -1, 1, -2, 2, …`, and so the number of columns is odd.

## Plans

A transform consists of a precomputation that depends only on `n` and the
parameters of the polynomials, and an execution. The functions in the table do
both. To reuse the precomputation, create a plan:

```julia
using FastTransforms, LinearAlgebra

F = diskrandn(Float64, 10, 37)   # n = 10, 4n-3 = 37
P = plan_disk2cxf(F, 0, 0)       # or plan_disk2cxf(Float64, 10, 0, 0)

G = P*F       # the forward transform, the same as disk2cxf(F, 0, 0)
P\G ≈ F       # the inverse transform, the same as cxf2disk(G, 0, 0)

lmul!(P, F)   # the forward transform, overwriting F
ldiv!(P, F)   # the inverse transform, overwriting F
```

`P*F` and `P\G` copy their argument, and `lmul!` and `ldiv!` work in place.

## Where the coefficients are stored

The polynomials of degree at most a given number do not fill a rectangular array,
and so the input of a forward transform is only partially used. For example,
there is one Zernike polynomial of degree `0`, two of degree `1`, and three of
degree `2`. With `n = 2`, [`disk2cxf`](@ref) acts on the six coefficients
``f_\ell^m`` of degree ``\ell \le 2`` and order ``m``, stored in a `2 × 5` array as:

```math
F = \begin{pmatrix}
f_0^0 & f_1^{-1} & f_1^1 & f_2^{-2} & f_2^2\\
f_2^0 & 0 & 0 & 0 & 0
\end{pmatrix}.
```

The functions `diskones`, `sphones`, `triones`, and so on return an array with a
one in every entry that stores a coefficient, which shows the pattern:

```julia
julia> diskones(Float64, 3, 9)
3×9 Matrix{Float64}:
 1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

julia> sphones(Float64, 3, 5)
3×5 Matrix{Float64}:
 1.0  1.0  1.0  1.0  1.0
 1.0  1.0  1.0  0.0  0.0
 1.0  0.0  0.0  0.0  0.0

julia> triones(Float64, 3, 3)
3×3 Matrix{Float64}:
 1.0  1.0  1.0
 1.0  1.0  0.0
 1.0  0.0  0.0
```

The rule for each transform is given in [The transforms](@ref) below.

## The other entries must be zero

The entries of the input that do not store a coefficient are not ignored. A
transform works in place on the whole array, and the output generally has more
nonzero entries than the input:

```julia
julia> F = [1.0 2.0 3.0 4.0 5.0
            6.0 0.0 0.0 0.0 0.0];

julia> G = disk2cxf(F)
2×5 Matrix{Float64}:
  1.41421  4.0  6.0  4.89898  6.12372
 14.6969   0.0  0.0  4.89898  6.12372

julia> cxf2disk(G) ≈ F
true
```

To compute the output, the transform reads every entry of the input, and it
assumes that the entries which do not store a coefficient are zero. If they are
not, then the output is wrong. In particular, they cannot be used to mark unused
storage:

```julia
julia> F[2, 2:end] .= NaN;

julia> disk2cxf(F)
2×5 Matrix{Float64}:
  1.41421  NaN  NaN  NaN  NaN
 14.6969   NaN  NaN  NaN  NaN
```

The reason is that a column of the array is converted by a sequence of Givens
rotations, each acting on two entries of the column, followed by a
multiplication of the whole column by a triangular matrix. Both steps combine
the entries that store coefficients with those that do not.

To create an array with the right pattern of zeros, start from the functions
that end in `zeros`, `ones`, `rand`, or `randn`:

```@docs
sphzeros
diskzeros
trizeros
tetzeros
spinsphzeros
```

## The transforms

In all of the following, the first index of a coefficient, usually ``\ell``, is
the degree of the polynomial, and a tilde over a univariate polynomial means
that it is orthonormal. On the sphere, the disk, and the annulus, column `1`
stores the order ``m = 0``, and columns `2j` and `2j+1` store the orders
``m = -j`` and ``m = j``.

### Sphere

The input stores the coefficients ``f_\ell^m`` of real orthonormal spherical
harmonics, for ``0 \le \ell \le n-1`` and ``-\ell \le m \le \ell``:

```math
Y_\ell^m(\theta,\varphi) = (-1)^{|m|}\sqrt{(\ell+\tfrac{1}{2})\frac{(\ell-|m|)!}{(\ell+|m|)!}} P_\ell^{|m|}(\cos\theta) \sqrt{\frac{2-\delta_{m,0}}{2\pi}} \left\{\begin{array}{ccc} \cos(m\varphi) & {\rm for} & m \ge 0,\\ \sin(|m|\varphi) & {\rm for} & m < 0.\end{array}\right.
```

In the column of order ``m``, row `i` stores ``f_{|m|+i-1}^m``, for
`i = 1:n-|m|`. The output stores the coefficients ``g_\ell^m`` of the bivariate
Fourier series:

```math
\sum_{\ell=0}^{n-1}\sum_{m=1-n}^{n-1} g_\ell^m \left\{\begin{array}{ccc} \cos(\ell\theta) & {\rm for} & m~{\rm even},\\ \sin((\ell+1)\theta) & {\rm for} & m~{\rm odd},\end{array}\right\} \sqrt{\frac{2-\delta_{m,0}}{2\pi}}\left\{\begin{array}{ccc} \cos(m\varphi) & {\rm for} & m \ge 0,\\ \sin(|m|\varphi) & {\rm for} & m < 0,\end{array}\right.
```

with ``g_\ell^m`` in row `ℓ+1` of the column of order ``m``.

```@docs
sph2fourier
fourier2sph
plan_sph2fourier
```

### Sphere, with spin

The input stores the complex coefficients ``f_\ell^m`` of spin-weighted
spherical harmonics with integer spin ``s``, for
``\ell_0 \le \ell \le n-1`` and ``-\ell \le m \le \ell``, where
``\ell_0 = \max\{|m|,|s|\}``. In the column of order ``m``, row `i` stores
``f_{\ell_0+i-1}^m``, for `i = 1:n-ℓ₀`. The output stores the coefficients
``g_\ell^m`` of the bivariate Fourier series:

```math
\sum_{\ell=0}^{n-1}\sum_{m=1-n}^{n-1} g_\ell^m \frac{e^{{\rm i} m\varphi}}{\sqrt{2\pi}} \left\{\begin{array}{ccc} \cos(\ell\theta) & {\rm for} & m+s~{\rm even},\\ \sin((\ell+1)\theta) & {\rm for} & m+s~{\rm odd},\end{array}\right.
```

with ``g_\ell^m`` in row `ℓ+1` of the column of order ``m``.

```@docs
plan_spinsph2fourier
```

### Disk

The input stores the coefficients ``f_\ell^m`` of real orthonormal generalized
Zernike polynomials, for ``0 \le \ell \le 2n-2`` and
``m = -\ell, -\ell+2, \ldots, \ell``:

```math
Z_{\ell,m}^{(\alpha,\beta)}(r,\theta) = 2^{\frac{|m|+\alpha+\beta+2}{2}} r^{|m|}\tilde{P}_{\frac{\ell-|m|}{2}}^{(\beta,|m|+\alpha)}(2r^2-1) \sqrt{\frac{2-\delta_{m,0}}{2\pi}} \left\{\begin{array}{ccc} \cos(m\theta) & {\rm for} & m \ge 0,\\ \sin(|m|\theta) & {\rm for} & m < 0.\end{array}\right.
```

They are orthonormal with respect to the weight
``r^{2\alpha+1}(1-r^2)^\beta{\rm\,d}r{\rm\,d}\theta`` on the unit disk, so that
``\alpha = \beta = 0`` are the Zernike polynomials. In the column of order
``m``, row `i` stores ``f_{|m|+2i-2}^m``, for `i = 1:n-⌈|m|/2⌉`. The output
stores the coefficients ``g_{2\ell}^m`` of the Chebyshev×Fourier series:

```math
\sum_{\ell=0}^{n-1}\sum_{m=2-2n}^{2n-2} g_{2\ell}^m \left\{\begin{array}{ccc} T_{2\ell}(r) & {\rm for} & m~{\rm even},\\ T_{2\ell+1}(r) & {\rm for} & m~{\rm odd},\end{array}\right\} \sqrt{\frac{2-\delta_{m,0}}{2\pi}}\left\{\begin{array}{ccc} \cos(m\theta) & {\rm for} & m \ge 0,\\ \sin(|m|\theta) & {\rm for} & m < 0,\end{array}\right.
```

with ``g_{2\ell}^m`` in row `ℓ+1` of the column of order ``m``.

```@docs
disk2cxf
cxf2disk
plan_disk2cxf
```

### Annulus

The input stores the coefficients of real orthonormal annulus polynomials
``Z_{\ell,m}^{\rho,(\alpha,\beta,\gamma)}(r,\theta)``, in the same layout as on
the disk. They are orthonormal with respect to the weight
``r^{2\gamma+1}(r^2-\rho^2)^\alpha(1-r^2)^\beta{\rm\,d}r{\rm\,d}\theta`` on the
annulus ``\rho < r < 1``. The output stores the coefficients ``g_{2\ell}^m`` of
the Chebyshev×Fourier series:

```math
\sum_{\ell=0}^{n-1}\sum_{m=2-2n}^{2n-2} g_{2\ell}^m \left\{\begin{array}{ccc} T_{\ell}\left(\frac{2r^2-1-\rho^2}{1-\rho^2}\right) & {\rm for} & m~{\rm even},\\ \sqrt{\frac{2}{1-\rho^2}}rT_{\ell}\left(\frac{2r^2-1-\rho^2}{1-\rho^2}\right) & {\rm for} & m~{\rm odd},\end{array}\right\} \sqrt{\frac{2-\delta_{m,0}}{2\pi}}\left\{\begin{array}{ccc} \cos(m\theta) & {\rm for} & m \ge 0,\\ \sin(|m|\theta) & {\rm for} & m < 0,\end{array}\right.
```

with ``g_{2\ell}^m`` in row `ℓ+1` of the column of order ``m``.

```@docs
ann2cxf
cxf2ann
plan_ann2cxf
```

### Rectangularized disk

The input stores the coefficients ``f_\ell^m`` of real orthonormal Dunkl–Xu
polynomials, for ``0 \le m \le \ell \le n-1``:

```math
\tilde{P}_{\ell,m}^{(\beta)}(x,y) = (1-x^2)^{\frac{m}{2}}\tilde{P}_{\ell-m}^{(m+\beta+\frac{1}{2},m+\beta+\frac{1}{2})}(x) \tilde{P}_m^{(\beta,\beta)}\left(\frac{y}{\sqrt{1-x^2}}\right).
```

They are orthonormal with respect to the weight ``(1-x^2-y^2)^\beta`` on the
unit disk. Column `m+1` stores the order ``m``, and its row `i` stores
``f_{m+i-1}^m``, for `i = 1:n-m`. The output stores the coefficients
``g_\ell^m`` of the partially weighted Chebyshev series:

```math
\sum_{\ell=0}^{n-1}\sum_{m=0}^{n-1} g_\ell^m \left\{\begin{array}{ccc} T_\ell(x) & {\rm for} & m~{\rm even},\\ \sqrt{1-x^2}U_\ell(x) & {\rm for} & m~{\rm odd},\end{array}\right\} T_m\left(\frac{y}{\sqrt{1-x^2}}\right),
```

with ``g_\ell^m`` in row `ℓ+1` and column `m+1`.

```@docs
rectdisk2cheb
cheb2rectdisk
plan_rectdisk2cheb
```

### Triangle

The input stores the coefficients ``f_\ell^m`` of Proriol polynomials on the
triangle with vertices ``(0,0)``, ``(1,0)``, and ``(0,1)``, for
``0 \le m \le \ell \le n-1``:

```math
\tilde{P}_{\ell,m}^{(\alpha,\beta,\gamma)}(x,y) = (2-2x)^m \tilde{P}_{\ell-m}^{(2m+\beta+\gamma+1,\alpha)}(2x-1) \tilde{P}_m^{(\gamma,\beta)}\left(\frac{2y}{1-x}-1\right).
```

They are orthogonal with respect to the weight
``x^\alpha y^\beta(1-x-y)^\gamma``. Column `m+1` stores the order ``m``, and its
row `i` stores ``f_{m+i-1}^m``, for `i = 1:n-m`. The output stores the
coefficients ``g_\ell^m`` of the bivariate Chebyshev series:

```math
\sum_{\ell=0}^{n-1}\sum_{m=0}^{n-1} g_\ell^m T_\ell(2x-1) T_m\left(\frac{2y}{1-x}-1\right),
```

with ``g_\ell^m`` in row `ℓ+1` and column `m+1`.

```@docs
tri2cheb
cheb2tri
plan_tri2cheb
```

### Tetrahedron

The input stores the coefficients ``f_{k,\ell}^m`` of Proriol polynomials on the
tetrahedron with vertices ``(0,0,0)``, ``(1,0,0)``, ``(0,1,0)``, and
``(0,0,1)``, for ``\ell, m \ge 0`` and ``\ell + m \le k \le n-1``, where ``k``
is the degree:

```math
\tilde{P}_{k,\ell,m}^{(\alpha,\beta,\gamma,\delta)}(x,y,z) = (2-2x)^{\ell+m} \tilde{P}_{k-\ell-m}^{(2\ell+2m+\beta+\gamma+\delta+2,\alpha)}(2x-1) \left(2-\frac{2y}{1-x}\right)^m \tilde{P}_\ell^{(2m+\gamma+\delta+1,\beta)}\left(\frac{2y}{1-x}-1\right) \tilde{P}_m^{(\delta,\gamma)}\left(\frac{2z}{1-x-y}-1\right).
```

They are orthogonal with respect to the weight
``x^\alpha y^\beta z^\gamma(1-x-y-z)^\delta``. The entry `F[i, ℓ+1, m+1]` stores
``f_{\ell+m+i-1,\ell}^m``, for `i = 1:n-ℓ-m`. The output stores the coefficients
``g_{k,\ell}^m`` of the trivariate Chebyshev series:

```math
\sum_{k=0}^{n-1}\sum_{\ell=0}^{n-1}\sum_{m=0}^{n-1} g_{k,\ell}^m T_k(2x-1) T_\ell\left(\frac{2y}{1-x}-1\right) T_m\left(\frac{2z}{1-x-y}-1\right),
```

with ``g_{k,\ell}^m`` in the entry `G[k+1, ℓ+1, m+1]`.

```@docs
tet2cheb
cheb2tet
plan_tet2cheb
```
