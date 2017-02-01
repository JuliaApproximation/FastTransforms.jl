immutable ChebyshevUltrasphericalConstants{D,T}
    λ::T
    M::Int
    N::Int
    nM₀::Int
    αN::T
    K::Int
end

function ChebyshevUltrasphericalConstants{T}(c::AbstractVector{T},λ::T;M::Int=7,D::Bool=FORWARD)
    N = length(c)-1
    if modλ(λ) == 0 return ChebyshevUltrasphericalConstants{D,T}(λ,M,N,0,zero(T),0) end
    λm = modλ(λ)
    if D == FORWARD
        nM₀ = min(floor(Int,(eps(T)*sqrtpi*2^(M-λm-1)*factorial(M)/pochhammer(λm,M)/pochhammer(1-λm,M))^(-1/(M+1/2))),N)
        αN = min(one(T)/log(N/nM₀),half(T))
        K = N > 0 ? ceil(Int,log(N/nM₀)/log(1/αN)) : 0
    else#if D == BACKWARD
        nM₀ = min(floor(Int,(eps(T)*sqrtpi*2^(M-λm-1)*factorial(M)/pochhammer(λm,M)/pochhammer(1-λm,M))^(-1/(M+1/2))),2N)
        αN = min(one(T)/log(2N/nM₀),half(T))
        K = N > 0 ? ceil(Int,log(2N/nM₀)/log(1/αN)) : 0
    end
    ChebyshevUltrasphericalConstants{D,T}(λ,M,N,nM₀,αN,K)
end

immutable ChebyshevUltrasphericalIndices
    i₁::Vector{Int}
    i₂::Vector{Int}
    j₁::Vector{Int}
    j₂::Vector{Int}
end

function ChebyshevUltrasphericalIndices{D,T}(λ::T,CUC::ChebyshevUltrasphericalConstants{D,T},tempmindices::Vector{T},tempsin::Vector{T},tempsinλ::Vector{T})
    M,N,nM₀,αN,K = getconstants(CUC)

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
        i₁[k+1],i₂[k+1] = findmindices!(tempmindices,λ,j₁[k],M,tempsin,tempsinλ)
    end
    for k=1:K j₂[k+1] = j₁[k]-1 end

    ChebyshevUltrasphericalIndices(i₁,i₂,j₁,j₂)
end

type ChebyshevUltrasphericalPlan{D,T,DCT,DST,SA} <: FastTransformPlan{D,T}
    CUC::ChebyshevUltrasphericalConstants{D,T}
    CUI::ChebyshevUltrasphericalIndices
    p₁::DCT
    p₂::DST
    rp::RecurrencePlan{T}
    c₁::Vector{T}
    c₂::SA
    um::Vector{T}
    vm::Vector{T}
    θ::Vector{T}
    tempsin::Vector{T}
    tempsin2::Vector{T}
    tempsinλ::Vector{T}
    tempsinλm::Vector{T}
    tempmindices::Vector{T}
    cnλ::Vector{T}
    cnmλ::Vector{T}
    w::Vector{T}
    anλ::Vector{T}
    c_cheb2::Vector{T}
    pr::Vector{T}
    function ChebyshevUltrasphericalPlan(CUC::ChebyshevUltrasphericalConstants{D,T},CUI::ChebyshevUltrasphericalIndices,p₁::DCT,p₂::DST,rp::RecurrencePlan{T},c₁::Vector{T},c₂::SA,um::Vector{T},vm::Vector{T},θ::Vector{T},tempsin::Vector{T},tempsin2::Vector{T},tempsinλ::Vector{T},tempsinλm::Vector{T},tempmindices::Vector{T},cnλ::Vector{T},cnmλ::Vector{T})
        P = new()
        P.CUC = CUC
        P.CUI = CUI
        P.p₁ = p₁
        P.p₂ = p₂
        P.rp = rp
        P.c₁ = c₁
        P.c₂ = c₂
        P.um = um
        P.vm = vm
        P.θ = θ
        P.tempsin = tempsin
        P.tempsin2 = tempsin2
        P.tempsinλ = tempsinλ
        P.tempsinλm = tempsinλm
        P.tempmindices = tempmindices
        P.cnλ = cnλ
        P.cnmλ = cnmλ
        P
    end
    function ChebyshevUltrasphericalPlan(CUC::ChebyshevUltrasphericalConstants{D,T},CUI::ChebyshevUltrasphericalIndices,p₁::DCT,p₂::DST,rp::RecurrencePlan{T},c₁::Vector{T},c₂::SA,um::Vector{T},vm::Vector{T},θ::Vector{T},tempsin::Vector{T},tempsin2::Vector{T},tempsinλ::Vector{T},tempsinλm::Vector{T},tempmindices::Vector{T},cnλ::Vector{T},cnmλ::Vector{T},w::Vector{T},anλ::Vector{T},c_cheb2::Vector{T},pr::Vector{T})
        P = ChebyshevUltrasphericalPlan{D,T,DCT,DST,SA}(CUC,CUI,p₁,p₂,rp,c₁,c₂,um,vm,θ,tempsin,tempsin2,tempsinλ,tempsinλm,tempmindices,cnλ,cnmλ)
        P.w = w
        P.anλ = anλ
        P.c_cheb2 = c_cheb2
        P.pr = pr
        P
    end
    function ChebyshevUltrasphericalPlan(CUC::ChebyshevUltrasphericalConstants{D,T})
        P = new()
        P.CUC = CUC
        P
    end
end

function ForwardChebyshevUltrasphericalPlan{T}(c_ultra::AbstractVector{T},λ::T,M::Int)
    # Initialize constants
    CUC = ChebyshevUltrasphericalConstants(c_ultra,λ;M=M,D=FORWARD)
    if modλ(λ) == 0 return ChebyshevUltrasphericalPlan{FORWARD,T,Any,Any,Any}(CUC) end
    M,N,nM₀,αN,K = getconstants(CUC)
    λ = modλ(λ)

    # Initialize recurrence coefficients
    rp = RecurrencePlan(λ-half(λ),λ-half(λ),N+1)

    # Initialize temporary arrays
    c₁,c₂,um,vm = zero(c_ultra),view(zero(c_ultra),2:N),zero(c_ultra),zero(c_ultra)

    # Initialize DCT-I and DST-I plans
    p₁,p₂ = applyTN_plan(c₁),applyUN_plan(c₂)

    # Clenshaw-Curtis points
    θ = N > 0 ? T[k/N for k=zero(T):N] : T[0]

    # Initialize sines and cosines
    tempsin = sinpi.(θ)
    @inbounds for i = 1:N÷2 tempsin[N+2-i] = tempsin[i] end
    tempsin2 = sinpi.(θ./2)
    tempsinλ,tempmindices = zero(c_ultra),zero(c_ultra)
    @inbounds for i=1:N+1 tempsinλ[i] = tempsin[i]^λ end
    tempsinλm = similar(tempsinλ)

    # Initialize normalizing constant
    cnλ = Cnλ(0:N,λ)
    cnmλ = similar(cnλ)

    # Get indices
    CUI = ChebyshevUltrasphericalIndices(λ,CUC,tempmindices,tempsin,tempsinλ)

    ChebyshevUltrasphericalPlan{FORWARD,T,typeof(p₁),typeof(p₂),typeof(c₂)}(CUC,CUI,p₁,p₂,rp,c₁,c₂,um,vm,θ,tempsin,tempsin2,tempsinλ,tempsinλm,tempmindices,cnλ,cnmλ)
end

function BackwardChebyshevUltrasphericalPlan{T}(c_ultra::AbstractVector{T},λ::T,M::Int)
    # Initialize constants
    CUC = ChebyshevUltrasphericalConstants(c_ultra,λ;M=M,D=BACKWARD)
    if modλ(λ) == 0 return ChebyshevUltrasphericalPlan{BACKWARD,T,Any,Any,Any}(CUC) end
    M,N,nM₀,αN,K = getconstants(CUC)
    λ = modλ(λ)

    # Array of almost double the size of the coefficients
    c_cheb2 = N > 0 ? zeros(T,2N+1) : T[0]

    # Initialize recurrence coefficients
    rp,pr = RecurrencePlan(λ-half(λ),λ-half(λ),2N+1),zero(c_cheb2)

    # Initialize temporary arrays
    c₁,c₂,um,vm = zero(c_cheb2),view(zero(c_cheb2),2:2N),zero(c_cheb2),zero(c_cheb2)

    # Initialize DCT-I and DST-I plans
    p₁,p₂ = applyTN_plan(c₁),applyUN_plan(c₂)

    # Clenshaw-Curtis nodes and weights
    θ = N > 0 ? T[k/2N for k=zero(T):2N] : T[0]
    w = N > 0 ? clenshawcurtisweights(2N+1,λ-half(λ),λ-half(λ),p₁) : T[0]

    # Initialize sines and cosines
    tempsin = sinpi.(θ)
    @inbounds for i = 1:N tempsin[2N+2-i] = tempsin[i] end
    tempsin2 = sinpi.(θ/2)
    tempsinλ,tempmindices = zero(c_cheb2),zero(c_cheb2)
    @inbounds for i=1:2N+1 tempsinλ[i] = tempsin[i]^λ end
    tempsinλm = similar(tempsinλ)

    # Initialize normalizing constant
    cnλ = Cnλ(0:2N,λ)
    cnmλ = similar(cnλ)

    # Initialize orthonormality constants
    anλ = Anαβ(0:N,λ-half(λ),λ-half(λ))

    # Get indices
    CUI = ChebyshevUltrasphericalIndices(λ,CUC,tempmindices,tempsin,tempsinλ)

    ChebyshevUltrasphericalPlan{BACKWARD,T,typeof(p₁),typeof(p₂),typeof(c₂)}(CUC,CUI,p₁,p₂,rp,c₁,c₂,um,vm,θ,tempsin,tempsin2,tempsinλ,tempsinλm,tempmindices,cnλ,cnmλ,w,anλ,c_cheb2,pr)
end

getplanλ(CUC::ChebyshevUltrasphericalConstants) = CUC.λ
getplanλ(plan::ChebyshevUltrasphericalPlan) = getplanλ(plan.CUC)

getconstants(CUC::ChebyshevUltrasphericalConstants) = CUC.M,CUC.N,CUC.nM₀,CUC.αN,CUC.K
getconstants(plan::ChebyshevUltrasphericalPlan) = getconstants(plan.CUC)

getindices(CUI::ChebyshevUltrasphericalIndices) = CUI.i₁,CUI.i₂,CUI.j₁,CUI.j₂
getindices(plan::ChebyshevUltrasphericalPlan) = getindices(plan.CUI)

getplan(plan::ChebyshevUltrasphericalPlan{FORWARD})  = plan.p₁,plan.p₂,plan.rp,plan.c₁,plan.c₂,plan.um,plan.vm,plan.θ,plan.tempsin,plan.tempsin2,plan.tempsinλ,plan.tempsinλm,plan.tempmindices,plan.cnλ,plan.cnmλ
getplan(plan::ChebyshevUltrasphericalPlan{BACKWARD}) = plan.p₁,plan.p₂,plan.rp,plan.c₁,plan.c₂,plan.um,plan.vm,plan.θ,plan.tempsin,plan.tempsin2,plan.tempsinλ,plan.tempsinλm,plan.tempmindices,plan.cnλ,plan.cnmλ,plan.w,plan.anλ,plan.c_cheb2,plan.pr
