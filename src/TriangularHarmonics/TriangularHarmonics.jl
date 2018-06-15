@compat abstract type TriangularHarmonicPlan{T} end

function *(P::TriangularHarmonicPlan, X::AbstractMatrix)
    mul!(zero(X), P, X)
end

function \(P::TriangularHarmonicPlan, X::AbstractMatrix)
    At_mul_B!(zero(X), P, X)
end

include("trifunctions.jl")
include("slowplan.jl")

function plan_tri2cheb(A::AbstractMatrix, α, β, γ; opts...)
    M, N = size(A)
    if M ≤ 1023
        SlowTriangularHarmonicPlan(A, α, β, γ)
    else
        ThinTriangularHarmonicPlan(A, α, β, γ; opts...)
    end
end

tri2cheb(A::AbstractMatrix, α, β, γ; opts...) = plan_tri2cheb(A, α, β, γ; opts...)*A
cheb2tri(A::AbstractMatrix, α, β, γ; opts...) = plan_tri2cheb(A, α, β, γ; opts...)\A
