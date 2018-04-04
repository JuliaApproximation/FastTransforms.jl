struct Pnmp1toPlm{T} <: AbstractRotation{T}
    rotations::Vector{Givens{T}}
end

function Pnmp1toPlm(::Type{T}, n::Int, m::Int, α::T, β::T, γ::T) where T
    G = Vector{Givens{T}}(n)
    @inbounds for ℓ = 1:n
        c = sqrt((2m+β+γ+2)/(ℓ+2m+β+γ+2)*(2ℓ+2m+α+β+γ+2)/(ℓ+2m+α+β+γ+2))
        s = sqrt(ℓ/(ℓ+2m+β+γ+2)*(ℓ+α)/(ℓ+2m+α+β+γ+2))
        G[n+1-ℓ] = Givens(ℓ, ℓ+1, c, s)
    end
    Pnmp1toPlm(G)
end

@inline length(P::Pnmp1toPlm) = length(P.rotations)
@inline getindex(P::Pnmp1toPlm, i::Int) = P.rotations[i]

function Base.A_mul_B!(C::Pnmp1toPlm, A::AbstractVecOrMat)
    @inbounds for i = 1:length(C)
        A_mul_B!(C.rotations[i], A)
    end
    A
end

function Base.A_mul_B!(A::AbstractMatrix, C::Pnmp1toPlm)
    @inbounds for i = length(C):-1:1
        A_mul_B!(A, C.rotations[i])
    end
    A
end

struct TriRotationPlan{T} <: AbstractRotation{T}
    layers::Vector{Pnmp1toPlm{T}}
end

function TriRotationPlan(::Type{T}, n::Int, α::T, β::T, γ::T) where T
    layers = Vector{Pnmp1toPlm{T}}(n)
    @inbounds for m = 0:n-1
        layers[m+1] = Pnmp1toPlm(T, n-m, m, α, β, γ)
    end
    TriRotationPlan(layers)
end

function Base.A_mul_B!(P::TriRotationPlan, A::AbstractMatrix)
    M, N = size(A)
    @inbounds for m = N-1:-1:1
        layer = P.layers[m]
        for ℓ = (m+1):N
            @simd for i = 1:length(layer)
                G = layer[i]
                a1, a2 = A[G.i1,ℓ], A[G.i2,ℓ]
                A[G.i1,ℓ] = G.c*a1 + G.s*a2
                A[G.i2,ℓ] = G.c*a2 - G.s*a1
            end
        end
    end
    A
end

function Base.At_mul_B!(P::TriRotationPlan, A::AbstractMatrix)
    M, N = size(A)
    @inbounds for m = 1:N-1
        layer = P.layers[m]
        for ℓ = (m+1):N
            @simd for i = length(layer):-1:1
                G = layer[i]
                a1, a2 = A[G.i1,ℓ], A[G.i2,ℓ]
                A[G.i1,ℓ] = G.c*a1 - G.s*a2
                A[G.i2,ℓ] = G.s*a1 + G.c*a2
            end
        end
    end
    A
end

Base.Ac_mul_B!(P::TriRotationPlan, A::AbstractMatrix) = At_mul_B!(P, A)


struct SlowTriangularHarmonicPlan{T} <: TriangularHarmonicPlan{T}
    RP::TriRotationPlan{T}
    p::NormalizedLegendreToChebyshevPlan{T}
    pinv::ChebyshevToNormalizedLegendrePlan{T}
    B::Matrix{T}
end

function SlowTriangularHarmonicPlan(A::Matrix{T}, α, β, γ) where T
    @assert β == γ == -half(T)
    @assert α == zero(T)
    M, N = size(A)
    n = N
    RP = TriRotationPlan(T, n-1, α, β, γ)
    p = plan_normleg2cheb(A)
    pinv = plan_cheb2normleg(A)
    B = zeros(A)
    SlowTriangularHarmonicPlan(RP, p, pinv, B)
end

function Base.A_mul_B!(Y::Matrix, SP::SlowTriangularHarmonicPlan, X::Matrix)
    RP, p, B = SP.RP, SP.p, SP.B
    copy!(B, X)
    A_mul_B!(RP, B)
    M, N = size(X)
    for J = 1:N
        A_mul_B_col_J!!(Y, p, B, J)
    end
    @inbounds for J = 1:N
        nrm = sqrt((2-δ(J-1,0))/π)
        @simd for I = 1:M
            Y[I,J] *= nrm
        end
    end
    Y
end

function Base.At_mul_B!(Y::Matrix, SP::SlowTriangularHarmonicPlan, X::Matrix)
    RP, pinv, B = SP.RP, SP.pinv, SP.B
    copy!(B, X)
    M, N = size(X)
    @inbounds for J = 1:N
        nrm = sqrt(π/(2-δ(J-1,0)))
        @simd for I = 1:M
            B[I,J] *= nrm
        end
    end
    for J = 1:N
        A_mul_B_col_J!!(Y, pinv, B, J)
    end
    tri_zero_spurious_modes!(At_mul_B!(RP, Y))
end

Base.Ac_mul_B!(Y::Matrix, SP::SlowTriangularHarmonicPlan, X::Matrix) = At_mul_B!(Y, SP, X)
