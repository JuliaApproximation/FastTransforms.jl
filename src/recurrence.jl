type RecurrencePlan{T}
    A::Vector{T}
    B::Vector{T}
    C::Vector{T}
    rf₀::Vector{T}
    rf₁::Vector{T}
    rb₀::Vector{T}
    rb₁::Vector{T}
end

function RecurrencePlan{T}(α::T,β::T,N::Int)
    A  = T[recA(α,β,k) for k=0:N-1]
    B  = T[recB(α,β,k) for k=0:N-1]
    C  = T[recC(α,β,k) for k=1:N  ]
    rf₀ = T[rf0(α,β,k) for k=0:N-1]
    rf₁ = T[rf1(α,β,k) for k=0:N-1]
    rb₀ = T[rb0(α,β,k) for k=1:N-1]
    rb₁ = T[rb1(α,β,k) for k=1:N-1]
    RecurrencePlan(A,B,C,rf₀,rf₁,rb₀,rb₁)
end

recA{T}(α::T,β::T,k::Int)=k==0&&((α+β==0)||(α+β==-1))?(α+β)/2+1:(2k+α+β+1)*(2k+α+β+2)/(2*(k+1)*(k+α+β+1))
recB{T}(α::T,β::T,k::Int)=k==0&&((α+β==0)||(α+β==-1))?(α-β)/2:(α-β)*(α+β)*(2k+α+β+1)/(2*(k+1)*(k+α+β+1)*(2k+α+β))
recC{T}(α::T,β::T,k::Int)=(k+α)*(k+β)*(2k+α+β+2)/((k+1)*(k+α+β+1)*(2k+α+β))

rf0{T}(α::T,β::T,k::Int) = (k+α+one(T))/(k+one(T))
rf1{T}(α::T,β::T,k::Int) = -(k+β+one(T))/(k+one(T))

rb0{T}(α::T,β::T,k::Int) = (k+one(T))/k*(α+β+k+1)/(k+β)*(α+β+2k)/(α+β+2k+2)
rb1{T}(α::T,β::T,k::Int) = -(k+one(T))/k*(α+β+k+1)/(k+α)*(α+β+2k)/(α+β+2k+2)


# Forward polynomial recurrence. Select modified algorithms for x≈±1.

function forward_recurrence!{T}(p::AbstractVector,N::Int,θ::Number,plan::RecurrencePlan{T})
    if θ ≤ 1/4
        reinsch_f0!(p,N,sinpi(θ/2),plan)
    elseif θ ≥ 3/4
        reinsch_f1!(p,N,cospi(θ/2),plan)
    else#if 1/4 < θ < 3/4
        orthogonal_polynomial_recurrence!(p,N,cospi(θ),plan)
    end
end
forward_recurrence!{T}(p::AbstractVector,θ::Number,plan::RecurrencePlan{T}) = forward_recurrence!(p,length(p),θ,plan)

function forward_recurrence!{T}(p::AbstractVector,N::Int,θ::Number,cpθ::Number,spθ::Number,plan::RecurrencePlan{T})
    if θ ≤ 1/4
        reinsch_f0!(p,N,spθ,plan)
    elseif θ ≥ 3/4
        reinsch_f1!(p,N,cpθ,plan)
    else#if 1/4 < θ < 3/4
        orthogonal_polynomial_recurrence!(p,N,cospi(θ),plan)
    end
end

# Forward recurrence

function orthogonal_polynomial_recurrence!{V}(p::AbstractVector,N::Int,x::Number,plan::RecurrencePlan{V})
    A=plan.A
    B=plan.B
    C=plan.C

    T = promote_type(eltype(p),typeof(x),V)
    pk1, pk = one(T), muladd(A[1],x,B[1])
    p[1],p[2] = pk1, pk

    @inbounds for k = 2:N-1
        Ak, Bk, Ck = A[k], B[k], C[k-1]
        pk, pk1 = muladd(muladd(Ak,x,Bk),pk,-Ck*pk1), pk
        p[k+1] = pk
    end
end

# Modified forward recurrence algorithm for θ near 0
# spθ = sinpi(θ/2)

function reinsch_f0!{V}(p::AbstractVector,N::Int,spθ::Number,plan::RecurrencePlan{V})
    A=plan.A
    C=plan.C
    r=plan.rf₀

    T = promote_type(eltype(p),typeof(spθ),V)
    xm1 = -2spθ^2
    pk, dk = one(T), zero(T)
    dkp1 = A[1]*xm1/r[1]
    pkp1 = (pk+dkp1)*r[1]
    p[1], p[2] = pk, pkp1

    @inbounds for k = 2:N-1
        Ak, Ck, rk = A[k], C[k-1], r[k]
        pk, dk = pkp1, dkp1
        dkp1 = muladd(Ak*xm1,pk,Ck*dk)/rk
        pkp1 = (pk+dkp1)*rk
        p[k+1] = pkp1
    end
end

# Modified forward recurrence for θ near 1
# cpθ = cospi(θ/2)

function reinsch_f1!{V}(p::AbstractVector,N::Int,cpθ::Number,plan::RecurrencePlan{V})
    A=plan.A
    C=plan.C
    r=plan.rf₁

    T = promote_type(eltype(p),typeof(cpθ),V)
    xp1 = 2cpθ^2
    pk, dk = one(T), zero(T)
    dkp1 = A[1]*xp1/r[1]
    pkp1 = (pk+dkp1)*r[1]
    p[1], p[2] = pk, pkp1

    @inbounds for k = 2:N-1
        Ak, Ck, rk = A[k], C[k-1], r[k]
        pk, dk = pkp1, dkp1
        dkp1 = muladd(Ak*xp1,pk,Ck*dk)/rk
        pkp1 = (pk+dkp1)*rk
        p[k+1] = pkp1
    end
end


# Backward polynomial recurrence. Select modified algorithms for x≈±1.

function backward_recurrence{T}(c::AbstractVector,N::Int,θ::Number,plan::RecurrencePlan{T})
    if θ ≤ 1/4
        reinsch_b0(c,N,sinpi(θ/2),plan)
    elseif θ ≥ 3/4
        reinsch_b1(c,N,cospi(θ/2),plan)
    else#if 1/4 < θ < 3/4
        clenshaw(c,N,cospi(θ),plan)
    end
end
backward_recurrence{T}(c::AbstractVector,θ::Number,plan::RecurrencePlan{T}) = backward_recurrence(c,length(c),θ,plan)

function backward_recurrence{T}(c::AbstractVector,N::Int,θ::Number,cpθ::Number,spθ::Number,plan::RecurrencePlan{T})
    if θ ≤ 1/4
        reinsch_b0(c,N,spθ,plan)
    elseif θ ≥ 3/4
        reinsch_b1(c,N,cpθ,plan)
    else#if 1/4 < θ < 3/4
        clenshaw(c,N,cospi(θ),plan)
    end
end

# Clenshaw-Smith algorithm

function clenshaw{V}(c::AbstractVector,N::Int,x::Number,plan::RecurrencePlan{V})
    A=plan.A
    B=plan.B
    C=plan.C

    T = promote_type(eltype(c),typeof(x),V)
    bk, bk1, bk2 = zero(T), zero(T), zero(T)

    @inbounds for k = N:-1:2
        ck, Ak, Bk, Ck = c[k], A[k], B[k], C[k]
        bk = muladd(muladd(Ak,x,Bk),bk1,muladd(-Ck,bk2,ck))
        bk2, bk1, bk = bk1, bk, bk2
    end

    @inbounds ck, Ak, Bk, Ck = c[1], A[1], B[1], C[1]
    muladd(muladd(Ak,x,Bk),bk1,muladd(-Ck,bk2,ck))
end

# Modified Clenshaw-Smith algorithm for θ near 0
# spθ = sinpi(θ/2)

function reinsch_b0{V}(c::AbstractVector,N::Int,spθ::Number,plan::RecurrencePlan{V})
    A=plan.A
    C=plan.C
    r=plan.rb₀

    T = promote_type(eltype(c),typeof(spθ),V)
    xm1 = -2spθ^2
    ukp1, dkp1 = zero(T), zero(T)

    @inbounds for k=N:-1:2
        ck, Ak, Ck, rk = c[k], A[k], C[k], r[k-1]
        dk = muladd(Ak*xm1,ukp1,muladd(Ck,dkp1,ck))*rk
        uk = (dk+ukp1)/rk
        dkp1, ukp1 = dk, uk
    end

    @inbounds ck, Ak, Ck = c[1], A[1], C[1]
    muladd(Ak*xm1,ukp1,muladd(Ck,dkp1,ck))
end

# Modified Clenshaw-Smith algorithm for θ near 1
# cpθ = cospi(θ/2)

function reinsch_b1{V}(c::AbstractVector,N::Int,cpθ::Number,plan::RecurrencePlan{V})
    A=plan.A
    C=plan.C
    r=plan.rb₁

    T = promote_type(eltype(c),typeof(cpθ),V)
    xp1 = 2cpθ^2
    ukp1, dkp1 = zero(T), zero(T)

    @inbounds for k=N:-1:2
        ck, Ak, Ck, rk = c[k], A[k], C[k], r[k-1]
        dk = muladd(Ak*xp1,ukp1,muladd(Ck,dkp1,ck))*rk
        uk = (dk+ukp1)/rk
        dkp1, ukp1 = dk, uk
    end

    @inbounds ck, Ak, Ck = c[1], A[1], C[1]
    muladd(Ak*xp1,ukp1,muladd(Ck,dkp1,ck))
end
