import Base.LinAlg: Givens, AbstractRotation

### These three A_mul_B! should be in Base, but for the time being they do not add methods to Base.A_mul_B!; they add methods to the internal A_mul_B!.

function A_mul_B!{T<:Real}(G::Givens{T}, A::AbstractVecOrMat)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > m
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    @inbounds @simd for i = 1:n
        a1, a2 = A[G.i1,i], A[G.i2,i]
        A[G.i1,i] =  G.c*a1 + G.s*a2
        A[G.i2,i] = -G.s*a1 + G.c*a2
    end
    return A
end

function A_mul_B!(A::AbstractMatrix, G::Givens)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > n
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    @inbounds @simd for i = 1:m
        a1, a2 = A[i,G.i1], A[i,G.i2]
        A[i,G.i1] = G.c*a1 - conj(G.s)*a2
        A[i,G.i2] = G.s*a1 +       G.c*a2
    end
    return A
end

function A_mul_B!{T<:Real}(A::AbstractMatrix, G::Givens{T})
    m, n = size(A, 1), size(A, 2)
    if G.i2 > n
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    @inbounds @simd for i = 1:m
        a1, a2 = A[i,G.i1], A[i,G.i2]
        A[i,G.i1] = G.c*a1 - G.s*a2
        A[i,G.i2] = G.s*a1 + G.c*a2
    end
    return A
end

struct Pnmp2toPlm{T} <: AbstractRotation{T}
    rotations::Vector{Givens{T}}
end

function Pnmp2toPlm{T}(::Type{T}, n::Int, m::Int)
    G = Vector{Givens{T}}(n)
    @inbounds for ℓ = 1:n
        c = sqrt(T((2m+2)*(2ℓ+2m+3))/T((ℓ+2m+2)*(ℓ+2m+3)))
        s = sqrt(T(ℓ*(ℓ+1))/T((ℓ+2m+2)*(ℓ+2m+3)))
        G[n+1-ℓ] = Givens(ℓ, ℓ+2, c, s)
    end
    Pnmp2toPlm(G)
end

@inline length(P::Pnmp2toPlm) = length(P.rotations)
@inline getindex(P::Pnmp2toPlm, i::Int) = P.rotations[i]

function Base.A_mul_B!(C::Pnmp2toPlm, A::AbstractVecOrMat)
    @inbounds for i = 1:length(C)
        A_mul_B!(C.rotations[i], A)
    end
    A
end

function Base.A_mul_B!(A::AbstractMatrix, C::Pnmp2toPlm)
    @inbounds for i = length(C):-1:1
        A_mul_B!(A, C.rotations[i])
    end
    A
end


struct RotationPlan{T} <: AbstractRotation{T}
    layers::Vector{Pnmp2toPlm{T}}
end

function RotationPlan{T}(::Type{T}, n::Int)
    layers = Vector{Pnmp2toPlm{T}}(n-1)
    @inbounds for m = 0:n-2
        layers[m+1] = Pnmp2toPlm(T, n-1-m, m)
    end
    RotationPlan(layers)
end

function Base.A_mul_B!(P::RotationPlan, A::AbstractMatrix)
    M, N = size(A)
    @inbounds for m = N÷2-2:-1:0
        layer = P.layers[m+1]
        for ℓ = 2*(m+2):4:N
            @simd for i = 1:length(layer)
                G = layer[i]
                a1, a2 = A[G.i1,ℓ], A[G.i2,ℓ]
                A[G.i1,ℓ] = G.c*a1 + G.s*a2
                A[G.i2,ℓ] = G.c*a2 - G.s*a1
                a1, a2 = A[G.i1,ℓ+1], A[G.i2,ℓ+1]
                A[G.i1,ℓ+1] = G.c*a1 + G.s*a2
                A[G.i2,ℓ+1] = G.c*a2 - G.s*a1
            end
        end
    end
    A
end

function Base.At_mul_B!(P::RotationPlan, A::AbstractMatrix)
    M, N = size(A)
    @inbounds for m = 0:N÷2-2
        layer = P.layers[m+1]
        for ℓ = 2*(m+2):4:N
            @simd for i = length(layer):-1:1
                G = layer[i]
                a1, a2 = A[G.i1,ℓ], A[G.i2,ℓ]
                A[G.i1,ℓ] = G.c*a1 - G.s*a2
                A[G.i2,ℓ] = G.s*a1 + G.c*a2
                a1, a2 = A[G.i1,ℓ+1], A[G.i2,ℓ+1]
                A[G.i1,ℓ+1] = G.c*a1 - G.s*a2
                A[G.i2,ℓ+1] = G.s*a1 + G.c*a2
            end
        end
    end
    A
end

Base.Ac_mul_B!(P::RotationPlan, A::AbstractMatrix) = At_mul_B!(P, A)


struct SlowSphericalHarmonicPlan{T} <: SphericalHarmonicPlan{T}
    RP::RotationPlan{T}
    p1::NormalizedLegendreToChebyshevPlan{T}
    p2::NormalizedLegendre1ToChebyshev2Plan{T}
    p1inv::ChebyshevToNormalizedLegendrePlan{T}
    p2inv::Chebyshev2ToNormalizedLegendre1Plan{T}
    B::Matrix{T}
end

function SlowSphericalHarmonicPlan{T}(A::Matrix{T})
    M, N = size(A)
    n = (N+1)÷2
    RP = RotationPlan(T, n-1)
    p1 = plan_normleg2cheb(A)
    p2 = plan_normleg12cheb2(A)
    p1inv = plan_cheb2normleg(A)
    p2inv = plan_cheb22normleg1(A)
    B = zeros(A)
    SlowSphericalHarmonicPlan(RP, p1, p2, p1inv, p2inv, B)
end

function Base.A_mul_B!(Y::Matrix, SP::SlowSphericalHarmonicPlan, X::Matrix)
    RP, p1, p2, B = SP.RP, SP.p1, SP.p2, SP.B
    copy!(B, X)
    A_mul_B!(RP, B)
    M, N = size(X)
    A_mul_B_col_J!!(Y, p1, B, 1)
    for J = 2:4:N
        A_mul_B_col_J!!(Y, p2, B, J)
        A_mul_B_col_J!!(Y, p2, B, J+1)
    end
    for J = 4:4:N
        A_mul_B_col_J!!(Y, p1, B, J)
        A_mul_B_col_J!!(Y, p1, B, J+1)
    end
    Y
end

function Base.At_mul_B!(Y::Matrix, SP::SlowSphericalHarmonicPlan, X::Matrix)
    RP, p1inv, p2inv, B = SP.RP, SP.p1inv, SP.p2inv, SP.B
    copy!(B, X)
    M, N = size(X)
    A_mul_B_col_J!!(Y, p1inv, B, 1)
    for J = 2:4:N
        A_mul_B_col_J!!(Y, p2inv, B, J)
        A_mul_B_col_J!!(Y, p2inv, B, J+1)
    end
    for J = 4:4:N
        A_mul_B_col_J!!(Y, p1inv, B, J)
        A_mul_B_col_J!!(Y, p1inv, B, J+1)
    end
    zero_spurious_modes!(At_mul_B!(RP, Y))
end

Base.Ac_mul_B!(Y::Matrix, SP::SlowSphericalHarmonicPlan, X::Matrix) = At_mul_B!(Y, SP, X)
