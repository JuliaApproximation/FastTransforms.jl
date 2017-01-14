__precompile__()
module FastTransforms

using Base, ToeplitzMatrices, Compat

import Base: *
import Compat: view

export cjt, icjt, jjt, plan_cjt, plan_icjt
export leg2cheb, cheb2leg, leg2chebu, ultra2ultra, jac2jac
export gaunt
export paduatransform, ipaduatransform, paduatransform!, ipaduatransform!, paduapoints
export plan_paduatransform!, plan_ipaduatransform!

# Other module methods and constants:
#export ChebyshevJacobiPlan, jac2cheb, cheb2jac
#export sqrtpi, pochhammer, stirlingseries, stirlingremainder, Aratio, Cratio, Anαβ
#export Cnmαβ, Cnαβ, Cnmλ, Cnλ, Λ, absf, findmindices!
#export clenshawcurtis, clenshawcurtis_plan, clenshawcurtisweights
#export fejer1, fejer_plan1, fejerweights1
#export fejer2, fejer_plan2, fejerweights2
#export RecurrencePlan, forward_recurrence!, backward_recurrence

include("fftBigFloat.jl")
include("specialfunctions.jl")
include("clenshawcurtis.jl")
include("fejer.jl")
include("recurrence.jl")
include("PaduaTransform.jl")

abstract FastTransformPlan{D,T}

include("ChebyshevJacobiPlan.jl")
include("jac2cheb.jl")
include("cheb2jac.jl")

include("ChebyshevUltrasphericalPlan.jl")
include("ultra2cheb.jl")
include("cheb2ultra.jl")

include("cjt.jl")

include("toeplitzhankel.jl")

leg2cheb(x...)=th_leg2cheb(x...)
cheb2leg(x...)=th_cheb2leg(x...)
leg2chebu(x...)=th_leg2chebu(x...)
ultra2ultra(x...)=th_ultra2ultra(x...)
jac2jac(x...)=th_jac2jac(x...)

include("gaunt.jl")


include("precompile.jl")
_precompile_()
 
end # module
