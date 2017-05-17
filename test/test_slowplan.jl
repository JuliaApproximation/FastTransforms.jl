import FastTransforms: normalizecolumns!, maxcolnorm

N = round.([Int],logspace(1,2.5,10))

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
        A_mul_B!(B, SP, A)
        fill!(A, 0.0)
        t[j] += @elapsed At_mul_B!(A, SP, B)
        nrms[kk] = maxcolnorm(A - Ac)
    end
    t[j] /= Nr
    err[j] = mean(nrms)
    println("At a bandlimit of ",n,", the maximum 2-norm in the columns over ",Nr," trials is: ",err[j])
    j+=1
end
