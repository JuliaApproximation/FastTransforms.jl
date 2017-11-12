abstract type HierarchicalPlan{T} <: AbstractMatrix{T} end

function *(P::HierarchicalPlan, x::AbstractVector)
    A_mul_B!(zero(x), P, x)
end

function *(P::HierarchicalPlan, X::AbstractMatrix)
    A_mul_B!(zero(X), P, X)
end

# A_mul_B!! mutates x while overwriting y. The generic fallback assumes it doesn't mutate x.
A_mul_B!!(y::AbstractVector, P::HierarchicalPlan, x::AbstractVector) = A_mul_B!(y, P, x)
A_mul_B!!(Y::AbstractMatrix, P::HierarchicalPlan, X::AbstractMatrix) = A_mul_B!(Y, P, X)
A_mul_B_col_J!!(Y::AbstractMatrix, P::HierarchicalPlan, X::AbstractMatrix, J::Int) = A_mul_B_col_J!(Y, P, X, J)

# A_mul_B! falls back to the mutating version with a copy.
Base.A_mul_B!(y::AbstractVector, P::HierarchicalPlan, x::AbstractVector) = A_mul_B!!(y, P, copy(x))
Base.A_mul_B!(Y::AbstractMatrix, P::HierarchicalPlan, X::AbstractMatrix) = A_mul_B!!(Y, P, copy(X))
A_mul_B_col_J!(Y::AbstractMatrix, P::HierarchicalPlan, X::AbstractMatrix) = A_mul_B_col_J!!(Y, P, copy(X), J)

function scale_col_J!(b::AbstractVector, A::AbstractVecOrMat, J::Int)
    m, n = size(A)
    COLSHIFT = m*(J-1)
    @inbounds @simd for i = 1:m
        A[i+COLSHIFT] *= b[i]
    end
    A
end

function scale_col_J!(b::Number, A::AbstractVecOrMat, J::Int)
    m, n = size(A)
    COLSHIFT = m*(J-1)
    @inbounds @simd for i = 1:m
        A[i+COLSHIFT] *= b
    end
    A
end

abstract type HierarchicalPlanWithParity{T} <: HierarchicalPlan{T} end

size(P::HierarchicalPlanWithParity) = (size(P.even, 1)+size(P.odd, 1), size(P.even, 2)+size(P.odd, 2))

evensize(v::AbstractVecOrMat, d) = (L = size(v, d); iseven(L) ? L÷2 : (L+1)÷2)
oddsize(v::AbstractVecOrMat, d) = (L = size(v, d); iseven(L) ? L÷2 : (L-1)÷2)

UpperTriangularHierarchicalMatrix{T}(::Type{T}, f::Function, bd::Int) = UpperTriangularHierarchicalMatrix(T, f, bd, bd)
UpperTriangularHierarchicalMatrix{T}(::Type{T}, f::Function, b::Int, d::Int) = UpperTriangularHierarchicalMatrix(T, f, 1, b, 1, d)

function UpperTriangularHierarchicalMatrix{T}(::Type{T}, f::Function, a::Int, b::Int, c::Int, d::Int)
    if (b-a+1) < BLOCKSIZE(T) && (d-c+1) < BLOCKSIZE(T)
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = T[j ≥ i ? f(T,i,j) : zero(T) for i=a:a+i, j=c:c+j]
        H[Block(1), Block(2)] = T[f(T,i,j) for i=a:a+i, j=c+j+1:d]
        H[Block(2), Block(2)] = T[j ≥ i ? f(T,i,j) : zero(T) for i=a+i+1:b, j=c+j+1:d]

        H
    else
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = UpperTriangularHierarchicalMatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = HierarchicalMatrix(T, f, a, a+i, c+j+1, d)
        H[Block(2), Block(2)] = UpperTriangularHierarchicalMatrix(T, f, a+i+1, b, c+j+1, d)

        H
    end
end

function HierarchicalMatrix{T}(::Type{T}, f::Function, a::Int, b::Int, c::Int, d::Int)
    if (b-a+1) < BLOCKSIZE(T) && (d-c+1) < BLOCKSIZE(T)
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = barycentricmatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = barycentricmatrix(T, f, a, a+i, c+j+1, d)
        H[Block(2), Block(1)] = T[f(T,i,j) for i=a+i+1:b, j=c:c+j]
        H[Block(2), Block(2)] = barycentricmatrix(T, f, a+i+1, b, c+j+1, d)

        H
    else
        i = (b-a)÷2
        j = (d-c)÷2
        H = HierarchicalMatrix(T, 2, 2)
        H[Block(1), Block(1)] = barycentricmatrix(T, f, a, a+i, c, c+j)
        H[Block(1), Block(2)] = barycentricmatrix(T, f, a, a+i, c+j+1, d)
        H[Block(2), Block(1)] = HierarchicalMatrix(T, f, a+i+1, b, c, c+j)
        H[Block(2), Block(2)] = barycentricmatrix(T, f, a+i+1, b, c+j+1, d)

        H
    end
end

function Meven{T}(::Type{T}, x, y)
    T(Λ(1.0*(y-x)).*Λ(1.0*(y+x-2)))
end

function Modd{T}(::Type{T}, x, y)
    T(Λ(1.0*(y-x)).*Λ(1.0*(y+x-1)))
end

function Leven{T}(::Type{T}, x, y)
    if x == y
        if x == 1.0
            T(1.0)
        else
            T(sqrt(π)/2/Λ(2.0*(x-1.0)))
        end
    else
        T(-(y-1.0)*(2.0*x-1.5)/(2.0*x+2.0*y-3.0)/(y-x)*Λ(1.0*(y-x-1.0)).*Λ(1.0*(y+x-2.5)))
    end
end

function Lodd{T}(::Type{T}, x, y)
    if x == y
        if x == 1.0
            T(1.0)
        else
            T(sqrt(π)/2/Λ(2.0*x-1.0))
        end
    else
        T(-(2.0*y-1.0)*(2.0*x-0.5)/(2.0*x+2.0*y-1.0)/(2.0*y-2.0*x)*Λ(1.0*(y-x-1)).*Λ(0.5*(2.0*y+2.0*x-3)))
    end
end

struct LegendreToChebyshevPlan{T} <: HierarchicalPlanWithParity{T}
    even::HierarchicalMatrix{T}
    odd::HierarchicalMatrix{T}
end

function getindex(P::LegendreToChebyshevPlan, i::Int, j::Int)
    if isodd(i) && isodd(j)
        P.even[(i+1)÷2,(j+1)÷2]*(2-δ(i,1))/π
    elseif iseven(i) && iseven(j)
        P.odd[i÷2,j÷2]*2/π
    else
        zero(eltype(P))
    end
end

struct ChebyshevToLegendrePlan{T} <: HierarchicalPlanWithParity{T}
    even::HierarchicalMatrix{T}
    odd::HierarchicalMatrix{T}
end

function getindex(P::ChebyshevToLegendrePlan, i::Int, j::Int)
    if isodd(i) && isodd(j)
        P.even[(i+1)÷2,(j+1)÷2]
    elseif iseven(i) && iseven(j)
        P.odd[i÷2,j÷2]
    else
        zero(eltype(P))
    end
end

function Base.A_mul_B!(y::Vector, P::LegendreToChebyshevPlan, x::AbstractVector)
    A_mul_B!(y, P.even, x, 1, 1, 2, 2)
    A_mul_B!(y, P.odd, x, 2, 2, 2, 2)
    scale!(2/π, y)
    y[1] *= 0.5
    y
end

function Base.A_mul_B!(y::Vector, P::ChebyshevToLegendrePlan, x::AbstractVector)
    A_mul_B!(y, P.even, x, 1, 1, 2, 2)
    A_mul_B!(y, P.odd, x, 2, 2, 2, 2)
end

function Base.A_mul_B!(Y::Matrix, P::LegendreToChebyshevPlan, X::Matrix)
    m, n = size(X)
    for j = 1:n
        A_mul_B!(Y, P.even, X, 1+m*(j-1), 1+m*(j-1), 2, 2)
        A_mul_B!(Y, P.odd, X, 2+m*(j-1), 2+m*(j-1), 2, 2)
    end
    scale!(2/π, Y)
    for j = 1:n
        Y[1+m*(j-1)] *= 0.5
    end
    Y
end

function Base.A_mul_B!(Y::Matrix, P::ChebyshevToLegendrePlan, X::Matrix)
    m, n = size(X)
    for j = 1:n
        A_mul_B!(Y, P.even, X, 1+m*(j-1), 1+m*(j-1), 2, 2)
        A_mul_B!(Y, P.odd, X, 2+m*(j-1), 2+m*(j-1), 2, 2)
    end
    Y
end

################################################################################
# NORMLEG2CHEB

function Mevennorm{T}(::Type{T}, x, y)
    T(Λ(1.0*(y-x)).*Λ(1.0*(y+x-2)))
end

function Moddnorm{T}(::Type{T}, x, y)
    T(Λ(1.0*(y-x)).*Λ(1.0*(y+x-1)))
end

function Levennorm{T}(::Type{T}, x, y)
    if x == y
        if x == 1.0
            two(T)
        else
            T(sqrt(π)/2/Λ(2.0*(x-1.0))/(2.0*x-1.5))
        end
    else
        T(-(y-1.0)/(2.0*x+2.0*y-3.0)/(y-x)*Λ(1.0*(y-x-1.0)).*Λ(1.0*(y+x-2.5)))
    end
end

function Loddnorm{T}(::Type{T}, x, y)
    if x == y
        if x == 1.0
            two(T)/T(3)
        else
            T(sqrt(π)/2/Λ(2.0*x-1.0)/(2.0*x-0.5))
        end
    else
        T(-(2.0*y-1.0)/(2.0*x+2.0*y-1.0)/(2.0*y-2.0*x)*Λ(1.0*(y-x-1)).*Λ(0.5*(2.0*y+2.0*x-3)))
    end
end

struct NormalizedLegendreToChebyshevPlan{T} <: HierarchicalPlanWithParity{T}
    even::HierarchicalMatrix{T}
    odd::HierarchicalMatrix{T}
    scl::Vector{T}
end

function getindex(P::NormalizedLegendreToChebyshevPlan, i::Int, j::Int)
    if isodd(i) && isodd(j)
        P.even[(i+1)÷2,(j+1)÷2]*(2-δ(i,1))/π*P.scl[j]
    elseif iseven(i) && iseven(j)
        P.odd[i÷2,j÷2]*2/π*P.scl[j]
    else
        zero(eltype(P))
    end
end

struct ChebyshevToNormalizedLegendrePlan{T} <: HierarchicalPlanWithParity{T}
    even::HierarchicalMatrix{T}
    odd::HierarchicalMatrix{T}
    scl::Vector{T}
end

function getindex(P::ChebyshevToNormalizedLegendrePlan, i::Int, j::Int)
    if isodd(i) && isodd(j)
        P.even[(i+1)÷2,(j+1)÷2]*P.scl[i]
    elseif iseven(i) && iseven(j)
        P.odd[i÷2,j÷2]*P.scl[i]
    else
        zero(eltype(P))
    end
end

function A_mul_B!!(y::Vector, P::NormalizedLegendreToChebyshevPlan, x::AbstractVector)
    unsafe_broadcasttimes!(x, P.scl)
    A_mul_B!(y, P.even, x, 1, 1, 2, 2)
    A_mul_B!(y, P.odd, x, 2, 2, 2, 2)
    scale!(2/π, y)
    y[1] *= 0.5
    y
end

function Base.A_mul_B!(y::Vector, P::ChebyshevToNormalizedLegendrePlan, x::AbstractVector)
    A_mul_B!(y, P.even, x, 1, 1, 2, 2)
    A_mul_B!(y, P.odd, x, 2, 2, 2, 2)
    unsafe_broadcasttimes!(y, P.scl)
end

function A_mul_B!!(Y::Matrix, P::NormalizedLegendreToChebyshevPlan, X::Matrix)
    m, n = size(X)
    scale!(P.scl, X)
    for j = 1:n
        A_mul_B!(Y, P.even, X, 1+m*(j-1), 1+m*(j-1), 2, 2)
        A_mul_B!(Y, P.odd, X, 2+m*(j-1), 2+m*(j-1), 2, 2)
    end
    scale!(2/π, Y)
    @inbounds @simd for j = 1:n
        Y[1+m*(j-1)] *= 0.5
    end
    Y
end

function Base.A_mul_B!(Y::Matrix, P::ChebyshevToNormalizedLegendrePlan, X::Matrix)
    m, n = size(X)
    for j = 1:n
        A_mul_B!(Y, P.even, X, 1+m*(j-1), 1+m*(j-1), 2, 2)
        A_mul_B!(Y, P.odd, X, 2+m*(j-1), 2+m*(j-1), 2, 2)
    end
    scale!(P.scl, Y)
end

function A_mul_B_col_J!!(Y::Matrix, P::NormalizedLegendreToChebyshevPlan, X::Matrix, J::Int)
    m, n = size(X)
    COLSHIFT = m*(J-1)
    scale_col_J!(P.scl, X, J)
    A_mul_B!(Y, P.even, X, 1+COLSHIFT, 1+COLSHIFT, 2, 2)
    A_mul_B!(Y, P.odd, X, 2+COLSHIFT, 2+COLSHIFT, 2, 2)
    scale_col_J!(2/π, Y, J)
    @inbounds Y[1+COLSHIFT] *= 0.5
    Y
end

function A_mul_B_col_J!(Y::Matrix, P::ChebyshevToNormalizedLegendrePlan, X::Matrix, J::Int)
    m, n = size(X)
    COLSHIFT = m*(J-1)
    A_mul_B!(Y, P.even, X, 1+COLSHIFT, 1+COLSHIFT, 2, 2)
    A_mul_B!(Y, P.odd, X, 2+COLSHIFT, 2+COLSHIFT, 2, 2)
    scale_col_J!(P.scl, Y, J)
end

################################################################################
# NORMLEG12CHEB2

function Mnormeven{T}(::Type{T}, x, y)
    T(Λ(1.0*(y-x))*Λ(1.0*(y+x-1))*(4.0*x-2.0)/π)
end

function Mnormodd{T}(::Type{T}, x, y)
    T(Λ(1.0*(y-x))*Λ(1.0*(y+x))*(4.0*x)/π)
end

function Lnormeven{T}(::Type{T}, x, y)
    -T(Λ(1.0*(y-x))/(2.0*y-2.0*x-1.0)*Λ(1.0*(y+x-1)+0.5)/(2.0*y+2.0*x-2.0)*(2.0*x-0.5))
end

function Lnormodd{T}(::Type{T}, x, y)
    -T(Λ(1.0*(y-x))/(2.0*y-2.0*x-1.0)*Λ(1.0*(y+x)+0.5)/(2.0*y+2.0*x)*(2.0*x+0.5))
end

struct NormalizedLegendre1ToChebyshev2Plan{T} <: HierarchicalPlanWithParity{T}
    even::HierarchicalMatrix{T}
    odd::HierarchicalMatrix{T}
    scl::Vector{T}
end

function getindex(P::NormalizedLegendre1ToChebyshev2Plan, i::Int, j::Int)
    if isodd(i) && isodd(j)
        P.even[(i+1)÷2,(j+1)÷2]*P.scl[j]
    elseif iseven(i) && iseven(j)
        P.odd[i÷2,j÷2]*P.scl[j]
    else
        zero(eltype(P))
    end
end

struct Chebyshev2ToNormalizedLegendre1Plan{T} <: HierarchicalPlanWithParity{T}
    even::HierarchicalMatrix{T}
    odd::HierarchicalMatrix{T}
    scl::Vector{T}
end

function getindex(P::Chebyshev2ToNormalizedLegendre1Plan, i::Int, j::Int)
    if isodd(i) && isodd(j)
        P.even[(i+1)÷2,(j+1)÷2]*P.scl[i]
    elseif iseven(i) && iseven(j)
        P.odd[i÷2,j÷2]*P.scl[i]
    else
        zero(eltype(P))
    end
end

function A_mul_B!!(y::Vector, P::NormalizedLegendre1ToChebyshev2Plan, x::AbstractVector)
    unsafe_broadcasttimes!(x, P.scl)
    A_mul_B!(y, P.even, x, 1, 1, 2, 2)
    A_mul_B!(y, P.odd, x, 2, 2, 2, 2)
end

function Base.A_mul_B!(y::Vector, P::Chebyshev2ToNormalizedLegendre1Plan, x::AbstractVector)
    A_mul_B!(y, P.even, x, 1, 1, 2, 2)
    A_mul_B!(y, P.odd, x, 2, 2, 2, 2)
    unsafe_broadcasttimes!(y, P.scl)
end

function A_mul_B!!(Y::Matrix, P::NormalizedLegendre1ToChebyshev2Plan, X::Matrix)
    m, n = size(X)
    scale!(P.scl, X)
    for j = 1:n
        A_mul_B!(Y, P.even, X, 1+m*(j-1), 1+m*(j-1), 2, 2)
        A_mul_B!(Y, P.odd, X, 2+m*(j-1), 2+m*(j-1), 2, 2)
    end
    Y
end

function Base.A_mul_B!(Y::Matrix, P::Chebyshev2ToNormalizedLegendre1Plan, X::Matrix)
    m, n = size(X)
    for j = 1:n
        A_mul_B!(Y, P.even, X, 1+m*(j-1), 1+m*(j-1), 2, 2)
        A_mul_B!(Y, P.odd, X, 2+m*(j-1), 2+m*(j-1), 2, 2)
    end
    scale!(P.scl, Y)
end

function A_mul_B_col_J!!(Y::Matrix, P::NormalizedLegendre1ToChebyshev2Plan, X::Matrix, J::Int)
    m, n = size(X)
    COLSHIFT = m*(J-1)
    scale_col_J!(P.scl, X, J)
    A_mul_B!(Y, P.even, X, 1+COLSHIFT, 1+COLSHIFT, 2, 2)
    A_mul_B!(Y, P.odd, X, 2+COLSHIFT, 2+COLSHIFT, 2, 2)
    Y
end

function A_mul_B_col_J!(Y::Matrix, P::Chebyshev2ToNormalizedLegendre1Plan, X::Matrix, J::Int)
    m, n = size(X)
    COLSHIFT = m*(J-1)
    A_mul_B!(Y, P.even, X, 1+COLSHIFT, 1+COLSHIFT, 2, 2)
    A_mul_B!(Y, P.odd, X, 2+COLSHIFT, 2+COLSHIFT, 2, 2)
    scale_col_J!(P.scl, Y, J)
end

### API

for (f, T, feven, fodd) in ((:leg2cheb, :LegendreToChebyshevPlan, :Meven, :Modd),
                          (:cheb2leg, :ChebyshevToLegendrePlan, :Leven, :Lodd),
                          (:normleg2cheb, :NormalizedLegendreToChebyshevPlan, :Mevennorm, :Moddnorm),
                          (:cheb2normleg, :ChebyshevToNormalizedLegendrePlan, :Levennorm, :Loddnorm),
                          (:normleg12cheb2, :NormalizedLegendre1ToChebyshev2Plan, :Mnormeven, :Mnormodd),
                          (:cheb22normleg1, :Chebyshev2ToNormalizedLegendre1Plan, :Lnormeven, :Lnormodd))
    plan_f = Meta.parse("plan_"*string(f))
    plan_even_f = Meta.parse("plan_even_"*string(f))
    plan_odd_f = Meta.parse("plan_odd_"*string(f))
    @eval begin
        $plan_f(v::VecOrMat) = $T(v)
        $f(v::VecOrMat) = $plan_f(v)*v
        $plan_even_f(v::VecOrMat) = UpperTriangularHierarchicalMatrix(eltype(v), $feven, evensize(v, 1))
        $plan_odd_f(v::VecOrMat) = UpperTriangularHierarchicalMatrix(eltype(v), $fodd, oddsize(v, 1))
    end
end

LegendreToChebyshevPlan(v::VecOrMat) = LegendreToChebyshevPlan(plan_even_leg2cheb(v), plan_odd_leg2cheb(v))
ChebyshevToLegendrePlan(v::VecOrMat) = ChebyshevToLegendrePlan(plan_even_cheb2leg(v), plan_odd_cheb2leg(v))

NormalizedLegendreToChebyshevPlan(v::VecOrMat) = NormalizedLegendreToChebyshevPlan(plan_even_normleg2cheb(v), plan_odd_normleg2cheb(v), eltype(v)[sqrt(j-0.5) for j in 1:size(v, 1)])
ChebyshevToNormalizedLegendrePlan(v::VecOrMat) = ChebyshevToNormalizedLegendrePlan(plan_even_cheb2normleg(v), plan_odd_cheb2normleg(v), eltype(v)[sqrt(i-0.5) for i in 1:size(v, 1)])

NormalizedLegendre1ToChebyshev2Plan(v::VecOrMat) = NormalizedLegendre1ToChebyshev2Plan(plan_even_normleg12cheb2(v), plan_odd_normleg12cheb2(v), eltype(v)[sqrt((j+0.5)/(j*(j+1))) for j in 1:size(v, 1)])
Chebyshev2ToNormalizedLegendre1Plan(v::VecOrMat) = Chebyshev2ToNormalizedLegendre1Plan(plan_even_cheb22normleg1(v), plan_odd_cheb22normleg1(v), eltype(v)[sqrt(i*(i+1)/(i+0.5)) for i in 1:size(v, 1)])

doc"""
Computes the Chebyshev expansion coefficients given the Legendre expansion coefficients:

```math
{\rm CLT} : \sum_{n=0}^N c_n^{\rm leg}P_n(x) \to \sum_{n=0}^N c_n^{\rm cheb}T_n(x).
```
"""
leg2cheb(::Vector)

doc"""
Computes the Legendre expansion coefficients given the Chebyshev expansion coefficients:

```math
{\rm iCLT} : \sum_{n=0}^N c_n^{\rm cheb}T_n(x) \to \sum_{n=0}^N c_n^{\rm leg}P_n(x).
```
"""
cheb2leg(::Vector)

doc"""
Pre-computes the Legendre--Chebyshev transform.
"""
plan_leg2cheb(::Vector)

doc"""
Pre-computes the Chebyshev--Legendre transform.
"""
plan_cheb2leg(::Vector)
