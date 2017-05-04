immutable FastSphericalHarmonicPlan{T}
    RP::RotationPlan{T}
    BF::Vector{Butterfly{T}}
    p1::NormalizedLegendreToChebyshevPlan{T}
    p2::NormalizedLegendre1ToChebyshev2Plan{T}
    p1inv::ChebyshevToNormalizedLegendrePlan{T}
    p2inv::Chebyshev2ToNormalizedLegendre1Plan{T}
    B::Matrix{T}
end

function FastSphericalHarmonicPlan{T}(A::Matrix{T}, L::Int)
    m, n = size(A)
    RP = RotationPlan(T, n-1)
    a1 = A[:,1]
    p1 = plan_normleg2cheb(a1)
    p2 = plan_normleg12cheb2(a1)
    p1inv = plan_cheb2normleg(a1)
    p2inv = plan_cheb22normleg1(a1)
    B = zeros(A)
    Ce = eye(A)
    Co = eye(A)
    BF = Vector{Butterfly{T}}(n-2)
    for j = 1:2:n-2
        A_mul_B!(Ce, RP.layers[j])
        BF[j] = Butterfly(Ce, L)
        println("Level: ",j)
    end
    for j = 2:2:n-2
        A_mul_B!(Co, RP.layers[j])
        BF[j] = Butterfly(Co, L)
        println("Level: ",j)
    end
    FastSphericalHarmonicPlan(RP, BF, p1, p2, p1inv, p2inv, B)
end


function A_mul_B!(Y::Matrix, FP::FastSphericalHarmonicPlan, X::Matrix)
    RP, BF, p1, p2, B = FP.RP, FP.BF, FP.p1, FP.p2, FP.B
    fill!(B, zero(eltype(B)))
    copy!(B, 1, X, 1, 2size(X, 1))
    for j = 3:size(X, 2)
        A_mul_B_col_J!(B, BF[j-2], X, j)
    end
    A_mul_B_odd_cols!!(Y, p1, B)
    A_mul_B_even_cols!!(Y, p2, B)
end

allranks(FP::FastSphericalHarmonicPlan) = mapreduce(allranks,vcat,FP.BF)
