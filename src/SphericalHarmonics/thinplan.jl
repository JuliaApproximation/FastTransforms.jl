const LAYERSKELETON = 64

checklayer(j::Int) = j÷LAYERSKELETON == j/LAYERSKELETON

struct ThinSphericalHarmonicPlan{T} <: SphericalHarmonicPlan{T}
    RP::RotationPlan{T}
    BF::Vector{Butterfly{T}}
    p1::NormalizedLegendreToChebyshevPlan{T}
    p2::NormalizedLegendre1ToChebyshev2Plan{T}
    p1inv::ChebyshevToNormalizedLegendrePlan{T}
    p2inv::Chebyshev2ToNormalizedLegendre1Plan{T}
    B::Matrix{T}
end

function ThinSphericalHarmonicPlan(A::Matrix{T}, L::Int; opts...) where T
    M, N = size(A)
    n = (N+1)÷2
    RP = RotationPlan(T, n-1)
    p1 = plan_normleg2cheb(A)
    p2 = plan_normleg12cheb2(A)
    p1inv = plan_cheb2normleg(A)
    p2inv = plan_cheb22normleg1(A)
    B = zeros(A)
    Ce = eye(T, M)
    Co = eye(T, M)
    BF = Vector{Butterfly{T}}(n-2)
    P = Progress(n-2, 0.1, "Pre-computing...", 43)
    for j = 1:2:n-2
        A_mul_B!(Ce, RP.layers[j])
        checklayer(j+1) && (BF[j] = Butterfly(Ce, L; isorthogonal = true, opts...))
        next!(P)
    end
    for j = 2:2:n-2
        A_mul_B!(Co, RP.layers[j])
        checklayer(j) && (BF[j] = Butterfly(Co, L; isorthogonal = true, opts...))
        next!(P)
    end
    ThinSphericalHarmonicPlan(RP, BF, p1, p2, p1inv, p2inv, B)
end

ThinSphericalHarmonicPlan(A::Matrix; opts...) = ThinSphericalHarmonicPlan(A, floor(Int, log2(size(A, 1)+1)-6); opts...)

function Base.A_mul_B!(Y::Matrix, TP::ThinSphericalHarmonicPlan, X::Matrix)
    RP, BF, p1, p2, B = TP.RP, TP.BF, TP.p1, TP.p2, TP.B
    copy!(B, X)
    M, N = size(X)

    for j = 3:2:N÷2
        if checklayer(j-1)
            A_mul_B_col_J!(Y, BF[j-1], B, 2j)
            A_mul_B_col_J!(Y, BF[j-1], B, 2j+1)
        else
            ℓ = round(Int, (j-1)÷LAYERSKELETON)*LAYERSKELETON
            A_mul_B_col_J!(RP, B, 2j, ℓ+1, j-1)
            A_mul_B_col_J!(RP, B, 2j+1, ℓ+1, j-1)
            if ℓ > LAYERSKELETON-2
                A_mul_B_col_J!(Y, BF[ℓ], B, 2j)
                A_mul_B_col_J!(Y, BF[ℓ], B, 2j+1)
            else
                copy!(Y, 1+M*(2j-1), B, 1+M*(2j-1), 2M)
            end
        end
    end

    for j = 2:2:N÷2
        if checklayer(j)
            A_mul_B_col_J!(Y, BF[j-1], B, 2j)
            A_mul_B_col_J!(Y, BF[j-1], B, 2j+1)
        else
            ℓ = round(Int, j÷LAYERSKELETON)*LAYERSKELETON
            A_mul_B_col_J!(RP, B, 2j, ℓ, j-1)
            A_mul_B_col_J!(RP, B, 2j+1, ℓ, j-1)
            if ℓ > LAYERSKELETON-2
                A_mul_B_col_J!(Y, BF[ℓ-1], B, 2j)
                A_mul_B_col_J!(Y, BF[ℓ-1], B, 2j+1)
            else
                copy!(Y, 1+M*(2j-1), B, 1+M*(2j-1), 2M)
            end
        end
    end

    copy!(Y, 1, X, 1, 3M)
    copy!(B, Y)
    fill!(Y, zero(eltype(Y)))

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


function Base.At_mul_B!(Y::Matrix, TP::ThinSphericalHarmonicPlan, X::Matrix)
    RP, BF, p1inv, p2inv, B = TP.RP, TP.BF, TP.p1inv, TP.p2inv, TP.B
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

    copy!(B, Y)
    fill!(Y, zero(eltype(Y)))
    copy!(Y, 1, B, 1, 3M)

    for j = 3:2:N÷2
        if checklayer(j-1)
            At_mul_B_col_J!(Y, BF[j-1], B, 2j)
            At_mul_B_col_J!(Y, BF[j-1], B, 2j+1)
        else
            ℓ = round(Int, (j-1)÷LAYERSKELETON)*LAYERSKELETON
            if ℓ > LAYERSKELETON-2
                At_mul_B_col_J!(Y, BF[ℓ], B, 2j)
                At_mul_B_col_J!(Y, BF[ℓ], B, 2j+1)
            else
                copy!(Y, 1+M*(2j-1), B, 1+M*(2j-1), 2M)
            end
            At_mul_B_col_J!(RP, Y, 2j, ℓ+1, j-1)
            At_mul_B_col_J!(RP, Y, 2j+1, ℓ+1, j-1)
        end
    end

    for j = 2:2:N÷2
        if checklayer(j)
            At_mul_B_col_J!(Y, BF[j-1], B, 2j)
            At_mul_B_col_J!(Y, BF[j-1], B, 2j+1)
        else
            ℓ = round(Int, j÷LAYERSKELETON)*LAYERSKELETON
            if ℓ > LAYERSKELETON-2
                At_mul_B_col_J!(Y, BF[ℓ-1], B, 2j)
                At_mul_B_col_J!(Y, BF[ℓ-1], B, 2j+1)
            else
                copy!(Y, 1+M*(2j-1), B, 1+M*(2j-1), 2M)
            end
            At_mul_B_col_J!(RP, Y, 2j, ℓ, j-1)
            At_mul_B_col_J!(RP, Y, 2j+1, ℓ, j-1)
        end
    end

    sph_zero_spurious_modes!(Y)
end

Base.Ac_mul_B!(Y::Matrix, TP::ThinSphericalHarmonicPlan, X::Matrix) = At_mul_B!(Y, TP, X)

allranks(TP::ThinSphericalHarmonicPlan) = mapreduce(i->allranks(TP.BF[i]),vcat,sort!([LAYERSKELETON-1:LAYERSKELETON:length(TP.BF);LAYERSKELETON:LAYERSKELETON:length(TP.BF)]))


function A_mul_B_col_J!(P::RotationPlan, A::AbstractMatrix, J::Int, L1::Int, L2::Int)
    M, N = size(A)
    @inbounds for m = L2-1:-2:L1
        layer = P.layers[m+1]
        @simd for i = 1:length(layer)
            G = layer[i]
            a1, a2 = A[G.i1,J], A[G.i2,J]
            A[G.i1,J] = G.c*a1 + G.s*a2
            A[G.i2,J] = G.c*a2 - G.s*a1
        end
    end
    A
end

function At_mul_B_col_J!(P::RotationPlan, A::AbstractMatrix, J::Int, L1::Int, L2::Int)
    M, N = size(A)
    @inbounds for m = L1:2:L2-1
        layer = P.layers[m+1]
        @simd for i = length(layer):-1:1
            G = layer[i]
            a1, a2 = A[G.i1,J], A[G.i2,J]
            A[G.i1,J] = G.c*a1 - G.s*a2
            A[G.i2,J] = G.c*a2 + G.s*a1
        end
    end
    A
end
