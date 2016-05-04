#__precompile__()
module FastTransforms

using Base, ToeplitzMatrices

import Base: *

export cjt, icjt, jjt, plan_cjt, plan_icjt
export leg2cheb,leg2chebu
export gaunt

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

abstract FastTransformPlan{D,T}

include("ChebyshevJacobiPlan.jl")
include("jac2cheb.jl")
include("cheb2jac.jl")

include("ChebyshevUltrasphericalPlan.jl")
include("ultra2cheb.jl")
include("cheb2ultra.jl")

include("cjt.jl")

include("gaunt.jl")


include("toeplitzhankel.jl")

leg2cheb(x...)=th_leg2cheb(x...)
leg2chebu(x...)=th_leg2chebu(x...)




end # module
