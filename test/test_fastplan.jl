using FastTransforms

import FastTransforms: RotationPlan, SlowSphericalHarmonicPlan, FastSphericalHarmonicPlan
import FastTransforms: allranks

function sphrand{T}(::Type{T}, m, n)
    A = zeros(T, m, 2n-1)
    for i = 1:m
        A[i,1] = rand(T)
    end
    for j = 1:n
        for i = 1:m-j
            A[i,2j] = rand(T)
            A[i,2j+1] = rand(T)
        end
    end
    A
end

function sphrandn{T}(::Type{T}, m, n)
    A = zeros(T, m, 2n-1)
    for i = 1:m
        A[i,1] = randn(T)
    end
    for j = 1:n
        for i = 1:m-j
            A[i,2j] = randn(T)
            A[i,2j+1] = randn(T)
        end
    end
    A
end

function normalizecolumns!(A::AbstractMatrix)
    m, n = size(A)
    @inbounds for j = 1:n
        nrm = zero(eltype(A))
        for i = 1:m
            nrm += abs2(A[i,j])
        end
        nrm = sqrt(nrm)
        for i = 1:m
            A[i,j] /= nrm
        end
    end
    A
end

function maxcolnorm(A::AbstractMatrix)
    m, n = size(A)
    nrm = zeros(n)
    @inbounds for j = 1:n
        nrm[n] = 0
        for i = 1:m
            nrm[n] += abs2(A[i,j])
        end
        nrm[n] = sqrt(nrm[n])
    end
    norm(nrm, Inf)
end

function zero_spurious_modes!(A::AbstractMatrix)
    M, N = size(A)
    n = N÷2
    for j = 1:n
        for i = M-j+1:M
            A[i,2j] = 0
            A[i,2j+1] = 0
        end
    end
    A
end

n = 255

A = sphrandn(Float64, n+1, n+1)
normalizecolumns!(A)
Ac = deepcopy(A)

FP = FastSphericalHarmonicPlan(A, 2);

B = zero(A);

@time A_mul_B!(B, FP, A);

SP = SlowSphericalHarmonicPlan(A);

C = zero(A);

@time A_mul_B!(C, SP, A);

println("The difference between slow and fast plans: ", maxcolnorm(B-C))

D = zero(A);

@time At_mul_B!(D, FP, B);

E = zero(A);

@time At_mul_B!(E, SP, C);

println("The difference between fast plan and original: ", maxcolnorm(A-D))
println("The difference between slow plan and original: ", maxcolnorm(A-E))
println("The difference between slow and fast plans: ", maxcolnorm(D-E))

zero_spurious_modes!(D);
zero_spurious_modes!(E);

println("The difference between fast plan and original: ", maxcolnorm(A-D))
println("The difference between slow plan and original: ", maxcolnorm(A-E))
println("The difference between slow and fast plans: ", maxcolnorm(D-E))
