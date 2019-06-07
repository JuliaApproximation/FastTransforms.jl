# FastTransforms.jl Documentation

## Introduction

`FastTransforms.jl` allows the user to conveniently work with orthogonal polynomials with degrees well into the millions.

Transforms include conversion between Jacobi polynomial expansions, with Chebyshev, Legendre, and ultraspherical polynomial transforms as special cases. For the signal processor, all three types of nonuniform fast Fourier transforms available. As well, spherical harmonic transforms and transforms between orthogonal polynomials on the triangle allow for the efficient simulation of partial differential equations of evolution.

Algorithms include methods based on asymptotic formulae to relate the transforms to a small number of fast Fourier transforms, matrix factorizations based on the Hadamard product, hierarchical matrix decompositions à la Fast Multipole Method, and the butterfly algorithm.

## Fast Transforms

```@docs
leg2cheb
```

```@docs
cheb2leg
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

```@docs
sph2fourier
```

```@docs
fourier2sph
```

## Planning The Fast Transforms

Most transforms have separate forward and inverse plans. In some instances, however, the inverse is in the sense of least-squares, and therefore only the forward transform is planned.

```@docs
plan_leg2cheb
```

```@docs
plan_cheb2leg
```

```@docs
plan_cjt
```

```@docs
plan_icjt
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
plan_inufft1
```

```@docs
plan_inufft2
```

```@docs
plan_paduatransform!
```

```@docs
plan_ipaduatransform!
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

### Jacobi Polynomial Increment and Decrement Operators

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
