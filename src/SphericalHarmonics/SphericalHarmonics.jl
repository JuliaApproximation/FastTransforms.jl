function zero_spurious_modes!(A::AbstractMatrix)
    M, N = size(A)
    n = N÷2
    @inbounds for j = 1:n
        @simd for i = M-j+1:M
            A[i,2j] = 0
            A[i,2j+1] = 0
        end
    end
    A
end

@compat abstract type SphericalHarmonicPlan{T} end

function *(P::SphericalHarmonicPlan, X::AbstractMatrix)
    A_mul_B!(zero(X), P, X)
end

function \(P::SphericalHarmonicPlan, X::AbstractMatrix)
    At_mul_B!(zero(X), P, X)
end

include("slowplan.jl")
include("Butterfly.jl")
include("fastplan.jl")
include("thinplan.jl")

function plan_sph2fourier(A::AbstractMatrix; opts...)
    M, N = size(A)
    if M ≤ 1022
        SlowSphericalHarmonicPlan(A)
    else
        ThinSphericalHarmonicPlan(A; opts...)
    end
end

sph2fourier(A::AbstractMatrix; opts...) = plan_sph2fourier(A; opts...)*A
fourier2sph(A::AbstractMatrix; opts...) = plan_sph2fourier(A; opts...)\A
