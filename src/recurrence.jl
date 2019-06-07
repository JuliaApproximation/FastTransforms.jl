struct RecurrencePlan{T}
    A::Vector{T}
    B::Vector{T}
    C::Vector{T}
    rf₀::Vector{T}
    rf₁::Vector{T}
    rf₀inv::Vector{T}
    rf₁inv::Vector{T}
    rb₀::Vector{T}
    rb₁::Vector{T}
    rb₀inv::Vector{T}
    rb₁inv::Vector{T}
end

function RecurrencePlan(α::T,β::T,N::Int) where T
    A  = T[recA(α,β,k) for k=0:N-1]
    B  = T[recB(α,β,k) for k=0:N-1]
    C  = T[recC(α,β,k) for k=1:N  ]
    rf₀ = T[rf0(α,β,k) for k=0:N-1]
    rf₁ = T[rf1(α,β,k) for k=0:N-1]
    rf₀inv = T[inv(rf₀[k]) for k=1:N]
    rf₁inv = T[inv(rf₁[k]) for k=1:N]
    rb₀ = T[rb0(α,β,k) for k=1:N-1]
    rb₁ = T[rb1(α,β,k) for k=1:N-1]
    rb₀inv = T[inv(rb₀[k]) for k=1:N-1]
    rb₁inv = T[inv(rb₁[k]) for k=1:N-1]
    RecurrencePlan(A,B,C,rf₀,rf₁,rf₀inv,rf₁inv,rb₀,rb₁,rb₀inv,rb₁inv)
end

recA(α::T,β::T,k::Int) where T = k == 0 && ((α+β==0)||(α+β==-1)) ? (α+β)/2+1 : (2k+α+β+1)*(2k+α+β+2)/(2*(k+1)*(k+α+β+1))
recB(α::T,β::T,k::Int) where T = k==0&&((α+β==0)||(α+β==-1)) ? (α-β)/2 : (α-β)*(α+β)*(2k+α+β+1)/(2*(k+1)*(k+α+β+1)*(2k+α+β))
recC(α::T,β::T,k::Int) where T = (k+α)*(k+β)*(2k+α+β+2)/((k+1)*(k+α+β+1)*(2k+α+β))

rf0(α::T,β::T,k::Int) where T = (k+α+one(T))/(k+one(T))
rf1(α::T,β::T,k::Int) where T = -(k+β+one(T))/(k+one(T))

rb0(α::T,β::T,k::Int) where T = (k+one(T))/k*(α+β+k+1)/(k+β)*(α+β+2k)/(α+β+2k+2)
rb1(α::T,β::T,k::Int) where T = -(k+one(T))/k*(α+β+k+1)/(k+α)*(α+β+2k)/(α+β+2k+2)

# Forward polynomial recurrence. Select modified algorithms for x≈±1.

function forward_recurrence!(p::AbstractVector,N::Int,θ::Number,plan::RecurrencePlan{T}) where T
    if θ ≤ 1/4
        reinsch_f0!(p,N,sinpi(θ/2),plan)
    elseif θ ≥ 3/4
        reinsch_f1!(p,N,cospi(θ/2),plan)
    else#if 1/4 < θ < 3/4
        orthogonal_polynomial_recurrence!(p,N,cospi(θ),plan)
    end
end

forward_recurrence!(p::AbstractVector,θ::Number,plan::RecurrencePlan{T}) where T =
    forward_recurrence!(p,length(p),θ,plan)


function forward_recurrence!(p::AbstractVector,N::Int,θ::Number,cpθ::Number,spθ::Number,plan::RecurrencePlan{T}) where T
    if θ ≤ 1/4
        reinsch_f0!(p,N,spθ,plan)
    elseif θ ≥ 3/4
        reinsch_f1!(p,N,cpθ,plan)
    else#if 1/4 < θ < 3/4
        orthogonal_polynomial_recurrence!(p,N,cospi(θ),plan)
    end
end

# Forward recurrence

function orthogonal_polynomial_recurrence!(p::AbstractVector,N::Int,x::Number,plan::RecurrencePlan{V}) where V
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

function reinsch_f0!(p::AbstractVector,N::Int,spθ::Number,plan::RecurrencePlan{V}) where V
    A=plan.A
    C=plan.C
    r=plan.rf₀
    rinv=plan.rf₀inv

    T = promote_type(eltype(p),typeof(spθ),V)
    xm1 = -2spθ^2
    p[1] = one(T)
    dk = A[1]*xm1*rinv[1]
    p[2] = (p[1]+dk)*r[1]

    @inbounds for k = 2:N-1
        Ak, Ck, rk, rkinv = A[k], C[k-1], r[k], rinv[k]
        temp = muladd(Ak*xm1,p[k],Ck*dk)
        dk, p[k+1] = temp*rkinv, muladd(rk,p[k],temp)
    end
end

# Modified forward recurrence for θ near 1
# cpθ = cospi(θ/2)

function reinsch_f1!(p::AbstractVector,N::Int,cpθ::Number,plan::RecurrencePlan{V}) where V
    A=plan.A
    C=plan.C
    r=plan.rf₁
    rinv=plan.rf₁inv

    T = promote_type(eltype(p),typeof(cpθ),V)
    xp1 = 2cpθ^2
    p[1] = one(T)
    dk = A[1]*xp1*rinv[1]
    p[2] = (p[1]+dk)*r[1]

    @inbounds for k = 2:N-1
        Ak, Ck, rk, rkinv = A[k], C[k-1], r[k], rinv[k]
        temp = muladd(Ak*xp1,p[k],Ck*dk)
        dk, p[k+1] = temp*rkinv, muladd(rk,p[k],temp)
    end
end


# Backward polynomial recurrence. Select modified algorithms for x≈±1.

function backward_recurrence(c::AbstractVector,N::Int,θ::Number,plan::RecurrencePlan{T}) where T
    if θ ≤ 1/4
        reinsch_b0(c,N,sinpi(θ/2),plan)
    elseif θ ≥ 3/4
        reinsch_b1(c,N,cospi(θ/2),plan)
    else#if 1/4 < θ < 3/4
        clenshaw(c,N,cospi(θ),plan)
    end
end

backward_recurrence(c::AbstractVector,θ::Number,plan::RecurrencePlan{T}) where T =
    backward_recurrence(c,length(c),θ,plan)

function backward_recurrence(c::AbstractVector,N::Int,θ::Number,cpθ::Number,spθ::Number,plan::RecurrencePlan{T}) where T
    if θ ≤ 1/4
        reinsch_b0(c,N,spθ,plan)
    elseif θ ≥ 3/4
        reinsch_b1(c,N,cpθ,plan)
    else#if 1/4 < θ < 3/4
        clenshaw(c,N,cospi(θ),plan)
    end
end

# Clenshaw-Smith algorithm

function clenshaw(c::AbstractVector,N::Int,x::Number,plan::RecurrencePlan{V}) where V
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

function reinsch_b0(c::AbstractVector,N::Int,spθ::Number,plan::RecurrencePlan{V}) where V
    A=plan.A
    C=plan.C
    r=plan.rb₀
    rinv=plan.rb₀inv

    T = promote_type(eltype(c),typeof(spθ),V)
    xm1 = -2spθ^2
    uk, dk = zero(T), zero(T)

    @inbounds for k=N:-1:2
        ck, Ak, Ck, rk, rkinv = c[k], A[k], C[k], r[k-1], rinv[k-1]
        temp = muladd(Ak*xm1,uk,muladd(Ck,dk,ck))
        dk, uk = temp*rk, muladd(rkinv,uk,temp)
    end

    @inbounds ck, Ak, Ck = c[1], A[1], C[1]
    muladd(Ak*xm1,uk,muladd(Ck,dk,ck))
end

# Modified Clenshaw-Smith algorithm for θ near 1
# cpθ = cospi(θ/2)

function reinsch_b1(c::AbstractVector,N::Int,cpθ::Number,plan::RecurrencePlan{V}) where V
    A=plan.A
    C=plan.C
    r=plan.rb₁
    rinv=plan.rb₁inv

    T = promote_type(eltype(c),typeof(cpθ),V)
    xp1 = 2cpθ^2
    uk, dk = zero(T), zero(T)

    @inbounds for k=N:-1:2
        ck, Ak, Ck, rk, rkinv = c[k], A[k], C[k], r[k-1], rinv[k-1]
        temp = muladd(Ak*xp1,uk,muladd(Ck,dk,ck))
        dk, uk = temp*rk, muladd(rkinv,uk,temp)
    end

    @inbounds ck, Ak, Ck = c[1], A[1], C[1]
    muladd(Ak*xp1,uk,muladd(Ck,dk,ck))
end
