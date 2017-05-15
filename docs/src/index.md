# FastTransforms.jl Documentation

## Introduction

In numerical analysis, it is customary to expand a function in a basis:
```math
f(x) = \sum_{\ell = 0}^{n} f_{\ell} \phi_{\ell}(x).
```
Orthogonal polynomials are examples of convenient bases for sufficiently smooth functions. To perform some operation, it may be necessary to convert our representation to one in a new basis, say, ``\{\psi_m(x)\}_{m\ge0}``.

If each function ``\phi_{\ell}`` is expanded in the basis ``\psi_m``, such as:
```math
\phi_{\ell}(x) \sim \sum_{m=0}^{\infty} c_{m,\ell}\psi_{m}(x),
```
then our original function has the alternative representation:
```math
f(x) \sim \sum_{m = 0}^{\infty} g_m \psi_m(x),
```
where ``g_m`` are defined by the sum:
```math
g_m = \sum_{\ell = 0}^{n} c_{m,\ell} f_{\ell}.
```

This is the classical connection problem. In many cases of interest, the both representations are finite and we seek a fast method (faster than ``\mathcal{O}(n^2)``) to transform the coefficients ``f_{\ell}`` to ``g_m``. These are the fast transforms.

## Fast Transforms

```@docs
leg2cheb
```

```@docs
cheb2leg
```

```@docs
plan_leg2cheb
```

```@docs
plan_cheb2leg
```

```@docs
cjt
```

```@docs
icjt
```

```@docs
jjt
```

```@docs
plan_cjt
```

```@docs
plan_icjt
```

```@docs
paduatransform
```

```@docs
ipaduatransform
```

```@docs
plan_paduatransform!
```

```@docs
plan_ipaduatransform!
```

```@docs
sph2fourier
```

```@docs
fourier2sph
```

```@docs
plan_sph2fourier
```

## Other Exported Methods

```@docs
gaunt
```

```@docs
paduapoints
```

## Internal Methods

```@docs
FastTransforms.half
```

```@docs
FastTransforms.two
```

```@docs
FastTransforms.δ
```

```@docs
FastTransforms.Λ
```

```@docs
FastTransforms.pochhammer
```

```@docs
FastTransforms.stirlingseries
```

```@docs
FastTransforms.clenshawcurtis
```

```@docs
FastTransforms.clenshawcurtisweights
```

```@docs
FastTransforms.fejer1
```

```@docs
FastTransforms.fejer2
```

```@docs
FastTransforms.fejerweights1
```

```@docs
FastTransforms.fejerweights2
```

```@docs
FastTransforms.chebyshevjacobimoments1
```

```@docs
FastTransforms.chebyshevjacobimoments2
```

```@docs
FastTransforms.incrementα!
```

```@docs
FastTransforms.incrementβ!
```

```@docs
FastTransforms.incrementαβ!
```

```@docs
FastTransforms.decrementα!
```

```@docs
FastTransforms.decrementβ!
```

```@docs
FastTransforms.decrementαβ!
```
