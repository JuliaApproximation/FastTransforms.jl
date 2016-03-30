__precompile__()
module FastTransforms

using Base

import Base: *

export cjt, icjt, jjt, plan_cjt, plan_icjt
export gaunt

# Other module methods and constants:
#export ChebyshevJacobiPlan, jac2cheb, cheb2jac
#export sqrtpi, pochhammer, stirlingseries, stirlingremainder, Aratio, Cratio, Anαβ
#export Cnmαβ, Cnαβ, Cnmλ, Cnλ, Cx, absf, findmindices!
#export clenshawcurtis, clenshawcurtis_plan, clenshawcurtisweights
#export fejer1, fejer_plan1, fejerweights1
#export fejer2, fejer_plan2, fejerweights2
#export RecurrencePlan, forward_recurrence!, backward_recurrence

include("specialfunctions.jl")
include("clenshawcurtis.jl")
include("fejer.jl")
include("recurrence.jl")

include("ChebyshevJacobiPlan.jl")
include("jac2cheb.jl")
include("cheb2jac.jl")

include("cjt.jl")

include("gaunt.jl")

end # module
