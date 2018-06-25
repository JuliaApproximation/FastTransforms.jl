import Compat.LinearAlgebra: Givens, AbstractRotation

### These three mul! should be in Base, but for the time being they do not add methods to Base.mul!; they add methods to the internal mul!.

function mul!(G::Givens{T}, A::AbstractVecOrMat) where T<:Real
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

function mul!(A::AbstractMatrix, G::Givens)
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

function mul!(A::AbstractMatrix, G::Givens{T}) where T<:Real
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

function Pnmp2toPlm(::Type{T}, n::Int, m::Int) where T
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
        mul!(C.rotations[i], A)
    end
    A
end

function Base.A_mul_B!(A::AbstractMatrix, C::Pnmp2toPlm)
    @inbounds for i = length(C):-1:1
        mul!(A, C.rotations[i])
    end
    A
end


struct RotationPlan{T} <: AbstractRotation{T}
    layers::Vector{Pnmp2toPlm{T}}
    snm::Vector{T}
    cnm::Vector{T}
end

function RotationPlan(::Type{T}, n::Int) where T
    layers = Vector{Pnmp2toPlm{T}}(n-1)
    @inbounds for m = 0:n-2
        layers[m+1] = Pnmp2toPlm(T, n-1-m, m)
    end
    n = n+1
    snm = zeros(T, (n*(n+1))÷2)
    cnm = zeros(T, (n*(n+1))÷2)
    @inbounds for l = 0:n-1
        for m = 0:n-l-1
            nums = T((l+1)*(l+2))
            numc = T((2*m+2)*(2*l+2*m+5))
            den = T((l+2*m+3)*(l+2*m+4))
            snm[l+(m*(2*n+1-m))÷2+1] = sqrt(nums/den)
            cnm[l+(m*(2*n+1-m))÷2+1] = sqrt(numc/den)
        end
    end
    RotationPlan(layers, snm, cnm)
end

function Base.A_mul_B!(P::RotationPlan, A::AbstractMatrix)
    N, M = size(A)
    snm = P.snm
    cnm = P.cnm
    if isodd(M)
        m = M÷2
        @inbounds for j = m:-2:2
            for l = N-j:-1:1
                s = snm[l+(j-2)*(2*N+3-j)÷2]
                c = cnm[l+(j-2)*(2*N+3-j)÷2]
                a1 = A[l+N*(2*m-1)]
                a2 = A[l+2+N*(2*m-1)]
                a3 = A[l+N*(2*m)]
                a4 = A[l+2+N*(2*m)]
                A[l+N*(2*m-1)] = c*a1 + s*a2
                A[l+2+N*(2*m-1)] = c*a2 - s*a1
                A[l+N*(2*m)] = c*a3 + s*a4
                A[l+2+N*(2*m)] = c*a4 - s*a3
            end
        end
    else
        m = M÷2
        @inbounds for j = m:-2:2
            for l = N-j:-1:1
                s = snm[l+(j-2)*(2*N+3-j)÷2]
                c = cnm[l+(j-2)*(2*N+3-j)÷2]
                a1 = A[l+N*(2*m-1)]
                a2 = A[l+2+N*(2*m-1)]
                A[l+N*(2*m-1)] = c*a1 + s*a2
                A[l+2+N*(2*m-1)] = c*a2 - s*a1
            end
        end
    end
    @stepthreads for m = M÷2-1:-1:2
        @inbounds for j = m:-2:2
            for l = N-j:-1:1
                s = snm[l+(j-2)*(2*N+3-j)÷2]
                c = cnm[l+(j-2)*(2*N+3-j)÷2]
                a1 = A[l+N*(2*m-1)]
                a2 = A[l+2+N*(2*m-1)]
                a3 = A[l+N*(2*m)]
                a4 = A[l+2+N*(2*m)]
                A[l+N*(2*m-1)] = c*a1 + s*a2
                A[l+2+N*(2*m-1)] = c*a2 - s*a1
                A[l+N*(2*m)] = c*a3 + s*a4
                A[l+2+N*(2*m)] = c*a4 - s*a3
            end
        end
    end
    A
end

function Base.At_mul_B!(P::RotationPlan, A::AbstractMatrix)
    N, M = size(A)
    snm = P.snm
    cnm = P.cnm
    if isodd(M)
        m = M÷2
        @inbounds for j = reverse(m:-2:2)
            for l = 1:N-j
                s = snm[l+(j-2)*(2*N+3-j)÷2]
                c = cnm[l+(j-2)*(2*N+3-j)÷2]
                a1 = A[l+N*(2*m-1)]
                a2 = A[l+2+N*(2*m-1)]
                a3 = A[l+N*(2*m)]
                a4 = A[l+2+N*(2*m)]
                A[l+N*(2*m-1)] = c*a1 - s*a2
                A[l+2+N*(2*m-1)] = c*a2 + s*a1
                A[l+N*(2*m)] = c*a3 - s*a4
                A[l+2+N*(2*m)] = c*a4 + s*a3
            end
        end
    else
        m = M÷2
        @inbounds for j = reverse(m:-2:2)
            for l = 1:N-j
                s = snm[l+(j-2)*(2*N+3-j)÷2]
                c = cnm[l+(j-2)*(2*N+3-j)÷2]
                a1 = A[l+N*(2*m-1)]
                a2 = A[l+2+N*(2*m-1)]
                A[l+N*(2*m-1)] = c*a1 - s*a2
                A[l+2+N*(2*m-1)] = c*a2 + s*a1
            end
        end
    end
    @stepthreads for m = M÷2-1:-1:2
        @inbounds for j = reverse(m:-2:2)
            for l = 1:N-j
                s = snm[l+(j-2)*(2*N+3-j)÷2]
                c = cnm[l+(j-2)*(2*N+3-j)÷2]
                a1 = A[l+N*(2*m-1)]
                a2 = A[l+2+N*(2*m-1)]
                a3 = A[l+N*(2*m)]
                a4 = A[l+2+N*(2*m)]
                A[l+N*(2*m-1)] = c*a1 - s*a2
                A[l+2+N*(2*m-1)] = c*a2 + s*a1
                A[l+N*(2*m)] = c*a3 - s*a4
                A[l+2+N*(2*m)] = c*a4 + s*a3
            end
        end
    end
    A
end

#=
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
=#

Base.Ac_mul_B!(P::RotationPlan, A::AbstractMatrix) = At_mul_B!(P, A)


struct SlowSphericalHarmonicPlan{T} <: SphericalHarmonicPlan{T}
    RP::RotationPlan{T}
    p1::NormalizedLegendreToChebyshevPlan{T}
    p2::NormalizedLegendre1ToChebyshev2Plan{T}
    p1inv::ChebyshevToNormalizedLegendrePlan{T}
    p2inv::Chebyshev2ToNormalizedLegendre1Plan{T}
    B::Matrix{T}
end

function SlowSphericalHarmonicPlan(A::Matrix{T}) where T
    M, N = size(A)
    RP = RotationPlan(T, M-1)
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
    mul!(RP, B)
    M, N = size(X)
    mul_col_J!!(Y, p1, B, 1)
    @stepthreads for J = 2:4:N
        mul_col_J!!(Y, p2, B, J)
        J < N && mul_col_J!!(Y, p2, B, J+1)
    end
    @stepthreads for J = 4:4:N
        mul_col_J!!(Y, p1, B, J)
        J < N && mul_col_J!!(Y, p1, B, J+1)
    end
    Y
end

function Base.At_mul_B!(Y::Matrix, SP::SlowSphericalHarmonicPlan, X::Matrix)
    RP, p1inv, p2inv, B = SP.RP, SP.p1inv, SP.p2inv, SP.B
    copy!(B, X)
    M, N = size(X)
    mul_col_J!!(Y, p1inv, B, 1)
    @stepthreads for J = 2:4:N
        mul_col_J!!(Y, p2inv, B, J)
        J < N && mul_col_J!!(Y, p2inv, B, J+1)
    end
    @stepthreads for J = 4:4:N
        mul_col_J!!(Y, p1inv, B, J)
        J < N && mul_col_J!!(Y, p1inv, B, J+1)
    end
    sph_zero_spurious_modes!(At_mul_B!(RP, Y))
end

Base.Ac_mul_B!(Y::Matrix, SP::SlowSphericalHarmonicPlan, X::Matrix) = At_mul_B!(Y, SP, X)
