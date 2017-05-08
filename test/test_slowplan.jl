using FastTransforms

import FastTransforms: RotationPlan, SlowSphericalHarmonicPlan

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

N = round.(Int,logspace(1,3,40))

t = zeros(length(N))
err = zeros(length(N))

Nr = 10

j = 1
for n in N
    RP = RotationPlan(Float64, n)
    nrms = zeros(Nr)
    for kk = 1:Nr
        A = sphrandn(Float64, n+1, n+1)
        normalizecolumns!(A)
        Ac = deepcopy(A)
        println("The bandlimit n is: ",n)
        t[j] += @elapsed At_mul_B!(RP, A_mul_B!(RP, A))
        nrms[kk] = maxcolnorm(A - Ac)
    end
    t[j] /= Nr
    err[j] = mean(nrms)#norm(nrms,Inf)
    println("The maximum 2-norm in the columns over 10 trials is: ",err[j])
    j+=1
end
