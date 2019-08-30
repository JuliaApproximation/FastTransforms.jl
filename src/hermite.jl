

hermitepoints(n) = unweightedgausshermite( n )[1]


# exp(-x^2/2) H_n(x) / sqrt(π*prod(1:n))
struct ForwardWeightedHermitePlan <: FastTransformPlan{FORWARD,Float64}
    Vtw::Matrix{Float64} # vandermonde
end

struct BackwardWeightedHermitePlan <: FastTransformPlan{BACKWARD,Float64}
    V::Matrix{Float64} # vandermonde
end


function _weightedhermite_vandermonde(n)
    V = Array{Float64}(undef,n,n)
    x,w = unweightedgausshermite( n )
    for k=1:n
        V[k,:] = FastGaussQuadrature.hermpoly_rec(0:n-1, sqrt(2)*x[k])
    end
    V,w
end
function ForwardWeightedHermitePlan(n::Integer) 
    V,w = _weightedhermite_vandermonde(n)
    ForwardWeightedHermitePlan(V' * Diagonal(w / sqrt(π)))
end
BackwardWeightedHermitePlan(n::Integer) = BackwardWeightedHermitePlan(_weightedhermite_vandermonde(n)[1])


*(P::ForwardWeightedHermitePlan, v::AbstractVector) = P.Vtw*v
*(P::ForwardWeightedHermitePlan, v::AbstractVector{Float64}) = P.Vtw*v
*(P::BackwardWeightedHermitePlan, v::AbstractVector) = P.V*v
*(P::BackwardWeightedHermitePlan, v::AbstractVector{Float64}) = P.V*v

weightedhermitetransform(v) = ForwardWeightedHermitePlan(length(v))*v
iweightedhermitetransform(v) = BackwardWeightedHermitePlan(length(v))*v

