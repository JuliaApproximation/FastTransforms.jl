import Base.LinAlg: Givens, AbstractRotation

immutable Pnmp2toPlm{T} <: AbstractRotation{T}
    rotations::Vector{Givens{T}}
end

function Pnmp2toPlm{T}(::Type{T}, n::Int, m::Int)
    G = Vector{Givens{T}}(n)
    for ℓ = 1:n
        c = sqrt((2ℓ+2m+3)/(ℓ+2m+3)*(2m+2)/(ℓ+2m+2))
        s = sqrt((ℓ+1)/(ℓ+2m+3)*ℓ/(ℓ+2m+2))
        G[n+1-ℓ] = Givens(ℓ, ℓ+2, c, s)
    end
    Pnmp2toPlm(G)
end

@inline length(P::Pnmp2toPlm) = length(P.rotations)
@inline getindex(P::Pnmp2toPlm,i::Int) = P.rotations[i]

function A_mul_B!(C::Pnmp2toPlm,A::AbstractVecOrMat)
    for i = 1:length(C)
        A_mul_B!(C.rotations[i], A)
    end
    A
end


immutable RotationPlan{T} <: AbstractRotation{T}
    layers::Vector{Pnmp2toPlm{T}}
end

function RotationPlan{T}(::Type{T}, n::Int)
    layers = Vector{Pnmp2toPlm{T}}(n-1)
    for m = 0:n-2
        layers[m+1] = Pnmp2toPlm(T, n-1-m, m)
    end
    RotationPlan(layers)
end

function A_mul_B!(P::RotationPlan, A::AbstractMatrix)
    n = length(P.layers)+1

    @inbounds for m = n-2:-1:0
        layer = P.layers[m+1]
        for ℓ = m+2:2:n
            @simd for i = 1:length(layer)
                G = layer[i]
                a1, a2 = A[G.i1,ℓ+1], A[G.i2,ℓ+1]
                A[G.i1,ℓ+1] = G.c*a1 + G.s*a2
                A[G.i2,ℓ+1] = G.c*a2 - G.s*a1
            end
        end
    end
    A
end

function Ac_mul_B!(P::RotationPlan, A::AbstractMatrix)
    n = length(P.layers)+1

    @inbounds for m = 0:n-2
        layer = P.layers[m+1]
        for ℓ = m+2:2:n
            @simd for i = length(layer):-1:1
                G = layer[i]
                a1, a2 = A[G.i1,ℓ+1], A[G.i2,ℓ+1]
                A[G.i1,ℓ+1] = G.c*a1 - G.s*a2
                A[G.i2,ℓ+1] = G.s*a1 + G.c*a2
            end
        end
    end
    A
end


immutable SlowSphericalHarmonicPlan{T}
    RP::RotationPlan{T}
    p1::NormalizedLegendreToChebyshevPlan{T}
    p2::NormalizedLegendre1ToChebyshev2Plan{T}
    p1inv::ChebyshevToNormalizedLegendrePlan{T}
    p2inv::Chebyshev2ToNormalizedLegendre1Plan{T}
    B::Matrix{T}
end

function SlowSphericalHarmonicPlan{T}(A::Matrix{T})
    m, n = size(A)
    RP = RotationPlan(T, n-1)
    a1 = A[:,1]
    p1 = plan_normleg2cheb(a1)
    p2 = plan_normleg12cheb2(a1)
    p1inv = plan_cheb2normleg(a1)
    p2inv = plan_cheb22normleg1(a1)
    B = zeros(A)
    SlowSphericalHarmonicPlan(RP, p1, p2, p1inv, p2inv, B)
end

function A_mul_B!(Y::Matrix, SP::SlowSphericalHarmonicPlan, X::Matrix)
    RP, p1, p2, B = SP.RP, SP.p1, SP.p2, SP.B
    copy!(B, X)
    A_mul_B!(RP, B)
    A_mul_B_odd_cols!!(Y, p1, B)
    A_mul_B_even_cols!!(Y, p2, B)
end

function Ac_mul_B!(Y::Matrix, SP::SlowSphericalHarmonicPlan, X::Matrix)
    RP, p1inv, p2inv, B = SP.RP, SP.p1inv, SP.p2inv, SP.B
    copy!(B, X)
    A_mul_B_odd_cols!!(Y, p1inv, B)
    A_mul_B_even_cols!!(Y, p2inv, B)
    Ac_mul_B!(RP, Y)
end
