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
    B = zero(A)
    Ce = Matrix{T}(I, M, M)
    Co = Matrix{T}(I, M, M)
    BF = Vector{Butterfly{T}}(undef, n-2)
    P = Progress(n-2, 0.1, "Pre-computing...", 43)
    for j = 1:2:n-2
        mul!(Ce, RP.layers[j])
        BF[j] = Butterfly(Ce, L; isorthogonal = true, opts...)
        next!(P)
    end
    for j = 2:2:n-2
        mul!(Co, RP.layers[j])
        BF[j] = Butterfly(Co, L; isorthogonal = true, opts...)
        next!(P)
    end
    FastSphericalHarmonicPlan(RP, BF, p1, p2, p1inv, p2inv, B)
end

FastSphericalHarmonicPlan(A::Matrix; opts...) = FastSphericalHarmonicPlan(A, floor(Int, log2(size(A, 1)+1)-6); opts...)

adjoint(P::FastSphericalHarmonicPlan) = Adjoint(P)
transpose(P::FastSphericalHarmonicPlan) = Transpose(P)



function LinearAlgebra.mul!(Y::Matrix, FP::FastSphericalHarmonicPlan, X::Matrix)
    RP, BF, p1, p2, B = FP.RP, FP.BF, FP.p1, FP.p2, FP.B
    fill!(B, zero(eltype(B)))
    M, N = size(X)
    copyto!(B, 1, X, 1, 3M)
    @stepthreads for J = 2:N÷2
        mul_col_J!(B, BF[J-1], X, 2J)
        2J < N && mul_col_J!(B, BF[J-1], X, 2J+1)
    end

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

function LinearAlgebra.mul!(Y::Matrix, FPt::Transpose{T,<:FastSphericalHarmonicPlan}, X::Matrix) where T
    FP = parent(FPt)
    RP, BF, p1inv, p2inv, B = FP.RP, FP.BF, FP.p1inv, FP.p2inv, FP.B
    copyto!(B, X)
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

    copyto!(B, Y)
    @stepthreads for J = 2:N÷2
        mul_col_J!(Y, tranpose(BF[J-1]), B, 2J)
        2J < N && mul_col_J!(Y, transpose(BF[J-1]), B, 2J+1)
    end
    sph_zero_spurious_modes!(Y)
end

LinearAlgebra.mul!(Y::Matrix, FPc::Adjoint{T,<:FastSphericalHarmonicPlan}, X::Matrix) where T =
    mul!(Y, transpose(parent(FPc)), X)

allranks(FP::FastSphericalHarmonicPlan) = mapreduce(allranks,vcat,FP.BF)
