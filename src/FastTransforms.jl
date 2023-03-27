module FastTransforms

using FastGaussQuadrature, FillArrays, LinearAlgebra,
      Reexport, SpecialFunctions, ToeplitzMatrices

@reexport using AbstractFFTs
@reexport using FFTW
@reexport using GenericFFT

import Base: convert, unsafe_convert, eltype, ndims, adjoint, transpose, show,
             *, \, inv, length, size, view, getindex

import Base.GMP: Limb

import AbstractFFTs: Plan, ScaledPlan,
                     fft, ifft, bfft, fft!, ifft!, bfft!, rfft, irfft, brfft,
                     plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!,
                     plan_bfft!, plan_rfft, plan_irfft, plan_brfft,
                     fftshift, ifftshift, rfft_output_size, brfft_output_size,
                     normalization

import FFTW: dct, dct!, idct, idct!, plan_dct!, plan_idct!,
             plan_dct, plan_idct, fftwNumber

import FastGaussQuadrature: unweightedgausshermite

import FillArrays: AbstractFill, getindex_value

import LinearAlgebra: mul!, lmul!, ldiv!

import GenericFFT: interlace # imported in downstream packages

export leg2cheb, cheb2leg, ultra2ultra, jac2jac,
       lag2lag, jac2ultra, ultra2jac, jac2cheb,
       cheb2jac, ultra2cheb, cheb2ultra, associatedjac2jac,
       modifiedjac2jac, modifiedlag2lag, modifiedherm2herm,
       sph2fourier, sphv2fourier, disk2cxf, ann2cxf, rectdisk2cheb,
       tri2cheb, tet2cheb,fourier2sph, fourier2sphv, cxf2disk, cxf2ann,
       cheb2rectdisk, cheb2tri, cheb2tet

export plan_leg2cheb, plan_cheb2leg, plan_ultra2ultra, plan_jac2jac,
       plan_lag2lag, plan_jac2ultra, plan_ultra2jac, plan_jac2cheb,
       plan_cheb2jac, plan_ultra2cheb, plan_cheb2ultra, plan_associatedjac2jac,
       plan_modifiedjac2jac, plan_modifiedlag2lag, plan_modifiedherm2herm,
       plan_sph2fourier, plan_sph_synthesis, plan_sph_analysis,
       plan_sphv2fourier, plan_sphv_synthesis, plan_sphv_analysis,
       plan_disk2cxf, plan_disk_synthesis, plan_disk_analysis,
       plan_ann2cxf, plan_annulus_synthesis, plan_annulus_analysis,
       plan_rectdisk2cheb, plan_rectdisk_synthesis, plan_rectdisk_analysis,
       plan_tri2cheb, plan_tri_synthesis, plan_tri_analysis,
       plan_tet2cheb, plan_tet_synthesis, plan_tet_analysis,
       plan_spinsph2fourier, plan_spinsph_synthesis, plan_spinsph_analysis

include("clenshaw.jl")

include("libfasttransforms.jl")
include("elliptic.jl")

export nufft, nufft1, nufft2, nufft3, inufft1, inufft2

export plan_nufft, plan_nufft1, plan_nufft2, plan_nufft3,
       plan_inufft1, plan_inufft2

include("nufft.jl")
include("inufft.jl")

export paduatransform, ipaduatransform, paduatransform!, ipaduatransform!,
       paduapoints

export plan_paduatransform!, plan_ipaduatransform!

include("PaduaTransform.jl")

export chebyshevtransform, ichebyshevtransform,
       chebyshevtransform!, ichebyshevtransform!,
       chebyshevutransform, ichebyshevutransform,
       chebyshevutransform!, ichebyshevutransform!, chebyshevpoints

export plan_chebyshevtransform, plan_ichebyshevtransform,
       plan_chebyshevtransform!, plan_ichebyshevtransform!,
       plan_chebyshevutransform, plan_ichebyshevutransform,
       plan_chebyshevutransform!, plan_ichebyshevutransform!

include("chebyshevtransform.jl")

export clenshawcurtisnodes, clenshawcurtisweights, fejernodes1, fejerweights1,
       fejernodes2, fejerweights2

export plan_clenshawcurtis, plan_fejer1, plan_fejer2

include("clenshawcurtis.jl")
include("fejer.jl")

export weightedhermitetransform, iweightedhermitetransform

include("hermite.jl")

export gaunt

include("gaunt.jl")

export sphones, sphzeros, sphrand, sphrandn, sphevaluate,
       sphvones, sphvzeros, sphvrand, sphvrandn,
       diskones, diskzeros, diskrand, diskrandn,
       rectdiskones, rectdiskzeros, rectdiskrand, rectdiskrandn,
       triones, trizeros, trirand, trirandn, trievaluate,
       tetones, tetzeros, tetrand, tetrandn,
       spinsphones, spinsphzeros, spinsphrand, spinsphrandn

include("specialfunctions.jl")

include("toeplitzplans.jl")
include("toeplitzhankel.jl")

# following use libfasttransforms by default
for f in (:jac2jac,
    :lag2lag, :jac2ultra, :ultra2jac, :jac2cheb,
    :cheb2jac, :ultra2cheb, :cheb2ultra, :associatedjac2jac,
    :modifiedjac2jac, :modifiedlag2lag, :modifiedherm2herm,
    :sph2fourier, :sphv2fourier, :disk2cxf, :ann2cxf,
    :rectdisk2cheb, :tri2cheb, :tet2cheb)
    lib_f = Symbol("lib_", f)
    @eval $f(x::AbstractArray, y...; z...) = $lib_f(x, y...; z...)
end

# following use Toeplitz-Hankel to avoid expensive plans
for f in (:leg2cheb, :cheb2leg, :ultra2ultra)
    th_f = Symbol("th_", f)
    lib_f = Symbol("lib_", f)
    @eval begin
        $f(x::AbstractArray, y...; z...) = $th_f(x, y...; z...)
        # $f(x::AbstractArray, y...; z...) = $lib_f(x, y...; z...)
    end
end


end # module
