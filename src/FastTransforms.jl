__precompile__()
module FastTransforms

using ToeplitzMatrices, HierarchicalMatrices, LowRankApprox, ProgressMeter, Compat,
        AbstractFFTs, SpecialFunctions

if VERSION < v"0.7-"
    using Base.FFTW
    import Base.FFTW: r2rFFTWPlan, unsafe_execute!, fftwSingle, fftwDouble, fftwNumber
    import Base.FFTW: libfftw, libfftwf, PlanPtr, r2rFFTWPlan
else
    using FFTW
    import FFTW: r2rFFTWPlan, unsafe_execute!, fftwSingle, fftwDouble, fftwNumber
    import FFTW: libfftw, libfftwf, PlanPtr, r2rFFTWPlan
end

import Base: *, \, size, view
import Base: getindex, setindex!, Factorization, length
import Base.LinAlg: BlasFloat, BlasInt
import HierarchicalMatrices: HierarchicalMatrix, unsafe_broadcasttimes!
import HierarchicalMatrices: A_mul_B!, At_mul_B!, Ac_mul_B!
import HierarchicalMatrices: ThreadSafeVector, threadsafezeros
import LowRankApprox: ColPerm
import AbstractFFTs: Plan


export cjt, icjt, jjt, plan_cjt, plan_icjt
export leg2cheb, cheb2leg, leg2chebu, ultra2ultra, jac2jac
export normleg2cheb, cheb2normleg, normleg12cheb2, cheb22normleg1
export plan_leg2cheb, plan_cheb2leg
export plan_normleg2cheb, plan_cheb2normleg
export plan_normleg12cheb2, plan_cheb22normleg1

export gaunt

export nufft, nufft1, nufft2, nufft3, inufft1, inufft2
export plan_nufft, plan_nufft1, plan_nufft2, plan_nufft3, plan_inufft1, plan_inufft2

export paduatransform, ipaduatransform, paduatransform!, ipaduatransform!, paduapoints
export plan_paduatransform!, plan_ipaduatransform!

export SlowSphericalHarmonicPlan, FastSphericalHarmonicPlan, ThinSphericalHarmonicPlan
export sph2fourier, fourier2sph, plan_sph2fourier
export sphones, sphzeros, sphrand, sphrandn, sphevaluate

export SlowTriangularHarmonicPlan
export tri2cheb, cheb2tri, plan_tri2cheb
export triones, trizeros, trirand, trirandn, trievaluate

# Other module methods and constants:
#export ChebyshevJacobiPlan, jac2cheb, cheb2jac
#export sqrtpi, pochhammer, stirlingseries, stirlingremainder, Aratio, Cratio, Anαβ
#export Cnmαβ, Cnαβ, Cnmλ, Cnλ, Λ, absf, findmindices!
#export clenshawcurtis, clenshawcurtis_plan, clenshawcurtisweights
#export fejer1, fejer_plan1, fejerweights1
#export fejer2, fejer_plan2, fejerweights2
#export RecurrencePlan, forward_recurrence!, backward_recurrence

include("stepthreading.jl")
include("fftBigFloat.jl")
include("specialfunctions.jl")
include("clenshawcurtis.jl")
include("fejer.jl")
include("recurrence.jl")
include("PaduaTransform.jl")

abstract type FastTransformPlan{D,T} end

include("ChebyshevJacobiPlan.jl")
include("jac2cheb.jl")
include("cheb2jac.jl")

include("ChebyshevUltrasphericalPlan.jl")
include("ultra2cheb.jl")
include("cheb2ultra.jl")
include("nufft.jl")
include("inufft.jl")

include("cjt.jl")

include("toeplitzhankel.jl")

#leg2cheb(x...)=th_leg2cheb(x...)
#cheb2leg(x...)=th_cheb2leg(x...)
leg2chebu(x...)=th_leg2chebu(x...)
ultra2ultra(x...)=th_ultra2ultra(x...)
jac2jac(x...)=th_jac2jac(x...)

include("hierarchical.jl")
include("SphericalHarmonics/SphericalHarmonics.jl")
include("TriangularHarmonics/TriangularHarmonics.jl")

include("gaunt.jl")


include("precompile.jl")
_precompile_()

end # module
