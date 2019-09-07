module FastTransforms

using AbstractFFTs, DSP, FastGaussQuadrature, FFTW, Libdl, LinearAlgebra, SpecialFunctions, ToeplitzMatrices

import Base: unsafe_convert, eltype, ndims, adjoint, transpose, show, *, \,
             inv, size, view

import Base.GMP: Limb
import Base.MPFR: BigFloat, _BigFloat

import AbstractFFTs: Plan

import DSP: conv

import FastGaussQuadrature: unweightedgausshermite

import FFTW: dct, dct!, idct, idct!,
             plan_fft!, plan_ifft!, plan_dct!, plan_idct!,
             plan_fft, plan_ifft, plan_rfft, plan_irfft, plan_dct, plan_idct,
			 plan_bfft, plan_bfft!, plan_brfft, fftwNumber

import LinearAlgebra: mul!, lmul!, ldiv!

export leg2cheb, cheb2leg, ultra2ultra, jac2jac,
       lag2lag, jac2ultra, ultra2jac, jac2cheb,
       cheb2jac, ultra2cheb, cheb2ultra,
       sph2fourier, sphv2fourier, disk2cxf, tri2cheb,
       fourier2sph, fourier2sphv, cxf2disk, cheb2tri

export plan_leg2cheb, plan_cheb2leg, plan_ultra2ultra, plan_jac2jac,
       plan_lag2lag, plan_jac2ultra, plan_ultra2jac, plan_jac2cheb,
       plan_cheb2jac, plan_ultra2cheb, plan_cheb2ultra,
       plan_sph2fourier, plan_sph_synthesis, plan_sph_analysis,
       plan_sphv2fourier, plan_sphv_synthesis, plan_sphv_analysis,
       plan_disk2cxf, plan_disk_synthesis, plan_disk_analysis,
       plan_tri2cheb, plan_tri_synthesis, plan_tri_analysis

include("libfasttransforms.jl")

export plan_nufft, plan_nufft1, plan_nufft2, plan_nufft3, plan_inufft1, plan_inufft2
export nufft, nufft1, nufft2, nufft3, inufft1, inufft2

include("nufft.jl")
include("inufft.jl")

export paduatransform, ipaduatransform, paduatransform!, ipaduatransform!,
       paduapoints, plan_paduatransform!, plan_ipaduatransform!

include("PaduaTransform.jl")

export plan_chebyshevtransform, plan_ichebyshevtransform, plan_chebyshevtransform!, plan_ichebyshevtransform!,
            chebyshevtransform, ichebyshevtransform, chebyshevpoints,
            plan_chebyshevutransform, plan_ichebyshevutransform, plan_chebyshevutransform!, plan_ichebyshevutransform!,
            chebyshevutransform, ichebyshevutransform

include("chebyshevtransform.jl")

export plan_clenshawcurtis, clenshawcurtisnodes, clenshawcurtisweights
export plan_fejer1, fejernodes1, fejerweights1,
       plan_fejer2, fejernodes2, fejerweights2

include("clenshawcurtis.jl")
include("fejer.jl")

export hermitepoints, weightedhermitetransform, iweightedhermitetransform

include("hermite.jl")

include("fftBigFloat.jl")

export gaunt

include("gaunt.jl")

export sphones, sphzeros, sphrand, sphrandn, sphevaluate,
       sphvones, sphvzeros, sphvrand, sphvrandn,
       diskones, diskzeros, diskrand, diskrandn,
       triones, trizeros, trirand, trirandn, trievaluate

lgamma(x) = logabsgamma(x)[1]

include("specialfunctions.jl")

end # module
