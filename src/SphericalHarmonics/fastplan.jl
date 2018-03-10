struct FastSphericalHarmonicPlan{T} <: SphericalHarmonicPlan{T}
    RP::RotationPlan{T}
    BF::Vector{Butterfly{T}}
    p1::NormalizedLegendreToChebyshevPlan{T}
    p2::NormalizedLegendre1ToChebyshev2Plan{T}
    p1inv::ChebyshevToNormalizedLegendrePlan{T}
    p2inv::Chebyshev2ToNormalizedLegendre1Plan{T}
    B::Matrix{T}
end

function FastSphericalHarmonicPlan(A::Matrix{T}, L::Int; opts...) where T
    M, N = size(A)
    n = (N+1)÷2
    RP = RotationPlan(T, M-1)
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
        BF[j] = Butterfly(Ce, L; isorthogonal = true, opts...)
        next!(P)
    end
    for j = 2:2:n-2
        A_mul_B!(Co, RP.layers[j])
        BF[j] = Butterfly(Co, L; isorthogonal = true, opts...)
        next!(P)
    end
    FastSphericalHarmonicPlan(RP, BF, p1, p2, p1inv, p2inv, B)
end

FastSphericalHarmonicPlan(A::Matrix; opts...) = FastSphericalHarmonicPlan(A, floor(Int, log2(size(A, 1)+1)-6); opts...)

function Base.A_mul_B!(Y::Matrix, FP::FastSphericalHarmonicPlan, X::Matrix)
    RP, BF, p1, p2, B = FP.RP, FP.BF, FP.p1, FP.p2, FP.B
    fill!(B, zero(eltype(B)))
    M, N = size(X)
    copy!(B, 1, X, 1, 3M)
    for J = 2:N÷2
        A_mul_B_col_J!(B, BF[J-1], X, 2J)
        2J < N && A_mul_B_col_J!(B, BF[J-1], X, 2J+1)
    end

    A_mul_B_col_J!!(Y, p1, B, 1)
    for J = 2:4:N
        A_mul_B_col_J!!(Y, p2, B, J)
        J < N && A_mul_B_col_J!!(Y, p2, B, J+1)
    end
    for J = 4:4:N
        A_mul_B_col_J!!(Y, p1, B, J)
        J < N && A_mul_B_col_J!!(Y, p1, B, J+1)
    end
    Y
end

function Base.At_mul_B!(Y::Matrix, FP::FastSphericalHarmonicPlan, X::Matrix)
    RP, BF, p1inv, p2inv, B = FP.RP, FP.BF, FP.p1inv, FP.p2inv, FP.B
    copy!(B, X)
    M, N = size(X)
    A_mul_B_col_J!!(Y, p1inv, B, 1)
    for J = 2:4:N
        A_mul_B_col_J!!(Y, p2inv, B, J)
        J < N && A_mul_B_col_J!!(Y, p2inv, B, J+1)
    end
    for J = 4:4:N
        A_mul_B_col_J!!(Y, p1inv, B, J)
        J < N && A_mul_B_col_J!!(Y, p1inv, B, J+1)
    end

    copy!(B, Y)
    for J = 2:N÷2
        At_mul_B_col_J!(Y, BF[J-1], B, 2J)
        2J < N && At_mul_B_col_J!(Y, BF[J-1], B, 2J+1)
    end
    sph_zero_spurious_modes!(Y)
end

Base.Ac_mul_B!(Y::Matrix, FP::FastSphericalHarmonicPlan, X::Matrix) = At_mul_B!(Y, FP, X)

allranks(FP::FastSphericalHarmonicPlan) = mapreduce(allranks,vcat,FP.BF)
