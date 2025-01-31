# FastTransforms.jl Documentation

## Introduction

[`FastTransforms.jl`](https://github.com/JuliaApproximation/FastTransforms.jl) allows the user to conveniently work with orthogonal polynomials with degrees well into the millions.

This package provides a Julia wrapper for the [C library](https://github.com/MikaelSlevinsky/FastTransforms) of the same name. Additionally, all three types of nonuniform fast Fourier transforms available, as well as the Padua transform.

## Fast orthogonal polynomial transforms

For this documentation, please see the documentation for [FastTransforms](https://github.com/MikaelSlevinsky/FastTransforms). Most transforms have separate forward and inverse plans. In some instances, however, the inverse is in the sense of least-squares, and therefore only the forward transform is planned.

### Modified orthogonal polynomials via fast Cholesky factorization of the Gram matrix

```@docs
GramMatrix
ChebyshevGramMatrix
```

## Nonuniform fast Fourier transforms

```@docs
nufft1
nufft2
nufft3
inufft1
inufft2
paduatransform
ipaduatransform
```

## Other Exported Methods

```@docs
gaunt
paduapoints
sphevaluate
```

## Internal Methods

### Miscellaneous Special Functions

```@docs
FastTransforms.half
FastTransforms.two
FastTransforms.δ
FastTransforms.Λ
FastTransforms.lambertw
FastTransforms.pochhammer
FastTransforms.stirlingseries
```

### Modified Chebyshev Moment-Based Quadrature

```@docs
FastTransforms.clenshawcurtisnodes
FastTransforms.clenshawcurtisweights
FastTransforms.fejernodes1
FastTransforms.fejerweights1
FastTransforms.fejernodes2
FastTransforms.fejerweights2
FastTransforms.chebyshevmoments1
FastTransforms.chebyshevjacobimoments1
FastTransforms.chebyshevlogmoments1
FastTransforms.chebyshevmoments2
FastTransforms.chebyshevjacobimoments2
FastTransforms.chebyshevlogmoments2
```

### Elliptic Submodule

```@docs
FastTransforms.Elliptic
```
