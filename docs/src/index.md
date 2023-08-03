# FastTransforms.jl Documentation

## Introduction

[`FastTransforms.jl`](https://github.com/JuliaApproximation/FastTransforms.jl) allows the user to conveniently work with orthogonal polynomials with degrees well into the millions.

This package provides a Julia wrapper for the [C library](https://github.com/MikaelSlevinsky/FastTransforms) of the same name. Additionally, all three types of nonuniform fast Fourier transforms available, as well as the Padua transform.

## Fast orthogonal polynomial transforms

For this documentation, please see the documentation for [FastTransforms](https://github.com/MikaelSlevinsky/FastTransforms). Most transforms have separate forward and inverse plans. In some instances, however, the inverse is in the sense of least-squares, and therefore only the forward transform is planned.

## Nonuniform fast Fourier transforms

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
inufft1
```

```@docs
inufft2
```

```@docs
paduatransform
```

```@docs
ipaduatransform
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

### Miscellaneous Special Functions

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

### Modified Chebyshev Moment-Based Quadrature

```@docs
FastTransforms.clenshawcurtisnodes
```

```@docs
FastTransforms.clenshawcurtisweights
```

```@docs
FastTransforms.fejernodes1
```

```@docs
FastTransforms.fejerweights1
```

```@docs
FastTransforms.fejernodes2
```

```@docs
FastTransforms.fejerweights2
```

```@docs
FastTransforms.chebyshevmoments1
```

```@docs
FastTransforms.chebyshevjacobimoments1
```

```@docs
FastTransforms.chebyshevlogmoments1
```

```@docs
FastTransforms.chebyshevmoments2
```

```@docs
FastTransforms.chebyshevjacobimoments2
```

```@docs
FastTransforms.chebyshevlogmoments2
```

### Elliptic

```@docs
FastTransforms.Elliptic
```
