# FastTransforms.jl Documentation

## Introduction

In numerical analysis, it is customary to expand a function in a basis:
```math
f(x) \sim \sum_{\ell=0}^{\infty} f_{\ell} \phi_{\ell}(x).
```
It may be necessary to transform our representation to one in a new basis, say, ``\{\psi_m(x)\}_{m\ge0}``:
```math
f(x) \sim \sum_{m=0}^{\infty} g_m \psi_m(x).
```
In many cases of interest, both representations are of finite length ``n`` and we seek a fast method (faster than ``\mathcal{O}(n^2)``) to transform the original coefficients ``f_{\ell}`` to the new coefficients ``g_m``.

A similar problem arises when we wish to evaluate ``f`` at a set of points ``\{x_m\}_{m=0}^n``. We wish to transform coefficients of ``f`` to values at the set of points in fewer than ``\mathcal{O}(n^2)`` operations.

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
nufft1
```

```@docs
nufft2
```

```@docs
nufft3
```

```@docs
plan_nufft1
```

```@docs
plan_nufft2
```

```@docs
plan_nufft3
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

```@docs
sphevaluate
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
FastTransforms.lambertw
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
