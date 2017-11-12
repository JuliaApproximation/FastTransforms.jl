struct ChebyshevJacobiConstants{D,T}
    α::T
    β::T
    M::Int
    N::Int
    nM₀::Int
    αN::T
    K::Int
end

function ChebyshevJacobiConstants{T}(c::AbstractVector{T},α::T,β::T;M::Int=7,D::Bool=FORWARD)
    N = length(c)-1
    if modαβ(α) == 0.5 && modαβ(β) == 0.5 return ChebyshevJacobiConstants{D,T}(α,β,M,N,0,zero(T),0) end
    if D == FORWARD
        nM₀ = floor(Int,min((eps(T)*2.0^(2M-1)*sqrtpi/absf(modαβ(α),modαβ(β),M,1/2))^(-1/(M+1/2)),N))
        αN = min(one(T)/log(N/nM₀),half(T))
        K = N > 0 ? ceil(Int,log(N/nM₀)/log(1/αN)) : 0
    else#if D == BACKWARD
        nM₀ = floor(Int,min((eps(T)*2.0^(2M-1)*sqrtpi/absf(modαβ(α),modαβ(β),M,1/2))^(-1/(M+1/2)),2N))
        αN = min(one(T)/log(2N/nM₀),half(T))
        K = N > 0 ? ceil(Int,log(2N/nM₀)/log(1/αN)) : 0
    end
    ChebyshevJacobiConstants{D,T}(α,β,M,N,nM₀,αN,K)
end

struct ChebyshevJacobiIndices
    i₁::Vector{Int}
    i₂::Vector{Int}
    j₁::Vector{Int}
    j₂::Vector{Int}
end

function ChebyshevJacobiIndices{D,T}(α::T,β::T,CJC::ChebyshevJacobiConstants{D,T},tempmindices::Vector{T},cfs::Matrix{T},tempcos::Vector{T},tempsin::Vector{T},tempcosβsinα::Vector{T})
    M,N,nM₀,αN,K = getconstants(CJC)

    i₁,i₂ = zeros(Int,K+1),zeros(Int,K+1)
    j₁,j₂ = zeros(Int,K+1),zeros(Int,K+1)

    if D == FORWARD
        i₁[1],i₂[1],j₂[1] = 1,N+1,N+1
        for k=1:K j₁[k] = ceil(Int,αN^k*N) end
    else#if D == BACKWARD
        i₁[1],i₂[1],j₂[1] = 1,2N+1,2N+1
        for k=1:K j₁[k] = ceil(Int,αN^k*2N) end
    end

    for k=1:K
        if j₁[k] < nM₀ break end
        i₁[k+1],i₂[k+1] = findmindices!(tempmindices,cfs,α,β,j₁[k],M,tempcos,tempsin,tempcosβsinα) # 4 allocations from the Beta function
    end
    for k=1:K j₂[k+1] = j₁[k]-1 end

    ChebyshevJacobiIndices(i₁,i₂,j₁,j₂)
end

mutable struct ChebyshevJacobiPlan{D,T,DCT,DST,SA} <: FastTransformPlan{D,T}
    CJC::ChebyshevJacobiConstants{D,T}
    CJI::ChebyshevJacobiIndices
    p₁::DCT
    p₂::DST
    rp::RecurrencePlan{T}
    c₁::Vector{T}
    c₂::SA
    um::Vector{T}
    vm::Vector{T}
    cfs::Matrix{T}
    θ::Vector{T}
    tempcos::Vector{T}
    tempsin::Vector{T}
    tempcosβsinα::Vector{T}
    tempmindices::Vector{T}
    cnαβ::Vector{T}
    cnmαβ::Vector{T}
    w::Vector{T}
    anαβ::Vector{T}
    c_cheb2::Vector{T}
    pr::Vector{T}
    function (::Type{ChebyshevJacobiPlan{D,T,DCT,DST,SA}}){D,T,DCT,DST,SA}(CJC::ChebyshevJacobiConstants{D,T},CJI::ChebyshevJacobiIndices,p₁::DCT,p₂::DST,rp::RecurrencePlan{T},c₁::Vector{T},c₂::SA,um::Vector{T},vm::Vector{T},cfs::Matrix{T},θ::Vector{T},tempcos::Vector{T},tempsin::Vector{T},
                                                tempcosβsinα::Vector{T},tempmindices::Vector{T},cnαβ::Vector{T},cnmαβ::Vector{T})
        P = new{D,T,DCT,DST,SA}()
        P.CJC = CJC
        P.CJI = CJI
        P.p₁ = p₁
        P.p₂ = p₂
        P.rp = rp
        P.c₁ = c₁
        P.c₂ = c₂
        P.um = um
        P.vm = vm
        P.cfs = cfs
        P.θ = θ
        P.tempcos = tempcos
        P.tempsin = tempsin
        P.tempcosβsinα = tempcosβsinα
        P.tempmindices = tempmindices
        P.cnαβ = cnαβ
        P.cnmαβ = cnmαβ
        P
    end
    function (::Type{ChebyshevJacobiPlan{D,T,DCT,DST,SA}}){D,T,DCT,DST,SA}(CJC::ChebyshevJacobiConstants{D,T},CJI::ChebyshevJacobiIndices,p₁::DCT,p₂::DST,rp::RecurrencePlan{T},c₁::Vector{T},c₂::SA,um::Vector{T},vm::Vector{T},cfs::Matrix{T},θ::Vector{T},tempcos::Vector{T},tempsin::Vector{T},tempcosβsinα::Vector{T},tempmindices::Vector{T},cnαβ::Vector{T},cnmαβ::Vector{T},
                                                 w::Vector{T},anαβ::Vector{T},c_cheb2::Vector{T},pr::Vector{T})
        P = ChebyshevJacobiPlan{D,T,DCT,DST,SA}(CJC,CJI,p₁,p₂,rp,c₁,c₂,um,vm,cfs,θ,tempcos,tempsin,tempcosβsinα,tempmindices,cnαβ,cnmαβ)
        P.w = w
        P.anαβ = anαβ
        P.c_cheb2 = c_cheb2
        P.pr = pr
        P
    end
    function (::Type{ChebyshevJacobiPlan{D,T,DCT,DST,SA}}){D,T,DCT,DST,SA}(CJC::ChebyshevJacobiConstants{D,T})
        P = new{D,T,DCT,DST,SA}()
        P.CJC = CJC
        P
    end
end

function ForwardChebyshevJacobiPlan{T}(c_jac::AbstractVector{T},α::T,β::T,M::Int)
    # Initialize constants
    CJC = ChebyshevJacobiConstants(c_jac,α,β;M=M,D=FORWARD)
    if modαβ(α) == 0.5 && modαβ(β) == 0.5 return ChebyshevJacobiPlan{FORWARD,T,Any,Any,Any}(CJC) end
    M,N,nM₀,αN,K = getconstants(CJC)
    α,β = modαβ(α),modαβ(β)

    # Initialize recurrence coefficients
    rp = RecurrencePlan(α,β,N+1)

    # Initialize temporary arrays
    c₁,c₂,um,vm = zero(c_jac),view(zero(c_jac),2:N),zero(c_jac),zero(c_jac)

    # Initialize DCT-I and DST-I plans
    p₁,p₂ = applyTN_plan(c₁),applyUN_plan(c₂)

    # Initialize coefficients of the asymptotic formula
    cfs = init_cfs(α,β,M)

    # Clenshaw-Curtis points
    θ = N > 0 ? T[k/N for k=zero(T):N] : T[0]

    # Initialize sines and cosines
    tempsin = sinpi.(θ./2)
    tempcos = reverse(tempsin)
    tempcosβsinα,tempmindices = zero(c_jac),zero(c_jac)
    @inbounds for i=1:N+1 tempcosβsinα[i] = tempcos[i]^(β+1/2)*tempsin[i]^(α+1/2) end

    # Initialize normalizing constant
    cnαβ = Cnαβ(0:N,α,β)
    cnmαβ = zero(cnαβ)

    # Get indices
    CJI = ChebyshevJacobiIndices(α,β,CJC,tempmindices,cfs,tempcos,tempsin,tempcosβsinα)

    ChebyshevJacobiPlan{FORWARD,T,typeof(p₁),typeof(p₂),typeof(c₂)}(CJC,CJI,p₁,p₂,rp,c₁,c₂,um,vm,cfs,θ,tempcos,tempsin,tempcosβsinα,tempmindices,cnαβ,cnmαβ)
end

function BackwardChebyshevJacobiPlan{T}(c_cheb::AbstractVector{T},α::T,β::T,M::Int)
    # Initialize constants
    CJC = ChebyshevJacobiConstants(c_cheb,α,β;M=M,D=BACKWARD)
    if modαβ(α) == 0.5 && modαβ(β) == 0.5 return ChebyshevJacobiPlan{BACKWARD,T,Any,Any,Any}(CJC) end
    M,N,nM₀,αN,K = getconstants(CJC)
    α,β = modαβ(α),modαβ(β)

    # Array of almost double the size of the coefficients
    c_cheb2 = N > 0 ? zeros(T,2N+1) : T[0]

    # Initialize recurrence coefficients and temporary array
    rp,pr = RecurrencePlan(α,β,2N+1),zero(c_cheb2)

    # Initialize temporary arrays
    c₁,c₂,um,vm = zero(c_cheb2),view(zero(c_cheb2),2:2N),zero(c_cheb2),zero(c_cheb2)

    # Initialize DCT-I and DST-I plans
    p₁,p₂ = applyTN_plan(c₁),applyUN_plan(c₂)

    # Initialize coefficients of the asymptotic formula
    cfs = init_cfs(α,β,M)

    # Clenshaw-Curtis nodes and weights
    θ = N > 0 ? T[k/2N for k=zero(T):2N] : T[0]
    w = N > 0 ? clenshawcurtisweights(2N+1,α,β,p₁) : T[0]

    # Initialize sines and cosines
    tempsin = sinpi.(θ./2)
    tempcos = reverse(tempsin)
    tempcosβsinα,tempmindices = zero(c_cheb2),zero(c_cheb2)
    @inbounds for i=1:2N+1 tempcosβsinα[i] = tempcos[i]^(β+1/2)*tempsin[i]^(α+1/2) end

    # Initialize normalizing constant
    cnαβ = Cnαβ(0:2N,α,β)
    cnmαβ = zero(cnαβ)

    # Initialize orthonormality constants
    anαβ = Anαβ(0:N,α,β)

    # Get indices
    CJI = ChebyshevJacobiIndices(α,β,CJC,tempmindices,cfs,tempcos,tempsin,tempcosβsinα)

    ChebyshevJacobiPlan{BACKWARD,T,typeof(p₁),typeof(p₂),typeof(c₂)}(CJC,CJI,p₁,p₂,rp,c₁,c₂,um,vm,cfs,θ,tempcos,tempsin,tempcosβsinα,tempmindices,cnαβ,cnmαβ,w,anαβ,c_cheb2,pr)
end

getplanαβ(CJC::ChebyshevJacobiConstants) = CJC.α,CJC.β
getplanαβ(plan::ChebyshevJacobiPlan) = getplanαβ(plan.CJC)

getconstants(CJC::ChebyshevJacobiConstants) = CJC.M,CJC.N,CJC.nM₀,CJC.αN,CJC.K
getconstants(plan::ChebyshevJacobiPlan) = getconstants(plan.CJC)

getindices(CJI::ChebyshevJacobiIndices) = CJI.i₁,CJI.i₂,CJI.j₁,CJI.j₂
getindices(plan::ChebyshevJacobiPlan) = getindices(plan.CJI)

getplan(plan::ChebyshevJacobiPlan{FORWARD})  = plan.p₁,plan.p₂,plan.rp,plan.c₁,plan.c₂,plan.um,plan.vm,plan.cfs,plan.θ,plan.tempcos,plan.tempsin,plan.tempcosβsinα,plan.tempmindices,plan.cnαβ,plan.cnmαβ
getplan(plan::ChebyshevJacobiPlan{BACKWARD}) = plan.p₁,plan.p₂,plan.rp,plan.c₁,plan.c₂,plan.um,plan.vm,plan.cfs,plan.θ,plan.tempcos,plan.tempsin,plan.tempcosβsinα,plan.tempmindices,plan.cnαβ,plan.cnmαβ,plan.w,plan.anαβ,plan.c_cheb2,plan.pr
