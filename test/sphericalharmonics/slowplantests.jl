using FastTransforms, Test

import FastTransforms: normalizecolumns!, maxcolnorm


@testset "Slow plan" begin
    N = round.([Int],10 .^ range(1,stop=2.5,length=10))

    t = zeros(length(N))
    err = zeros(length(N))

    Nr = 2

    j = 1
    for n in N
        nrms = zeros(Nr)
        for kk = 1:Nr
            A = sphrandn(Float64, n+1, n+1)
            normalizecolumns!(A)
            Ac = copy(A)
            B = zero(A)
            SP = SlowSphericalHarmonicPlan(A)
            mul!(B, SP, A)
            fill!(A, 0.0)
            t[j] += @elapsed mul!(A, transpose(SP), B)
            nrms[kk] = maxcolnorm(A - Ac)
        end
        t[j] /= Nr
        err[j] = Statistics.mean(nrms)
        println("At a bandlimit of ",n,", the maximum 2-norm in the columns over ",Nr," trials is: ",err[j])
        @test err[j] < 10eps()*sqrt(n)
        j+=1
    end
end
