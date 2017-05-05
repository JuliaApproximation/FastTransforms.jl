using FastTransforms, LowRankApprox
using Base.Test

println("A_mul_B!, At_mul_B!, and Ac_mul_B!")

for T in (Float64, Complex128)
    r = rand(T)
    A = idfact([r/(i+j-1) for i in 1:200, j = 1:50])
    P = A[:P]
    k, n = size(A)

    x = rand(T, n)
    y = zeros(T, k)

    @test A_mul_B!(y, A, P, x, 1, 1) == A*x

    x = rand(T, k)
    y = zeros(T, n)

    @test norm(At_mul_B!(y, A, P, x, 1, 1) - A.'x,Inf) < 10eps()

    fill!(y, zero(T))
    @test norm(Ac_mul_B!(y, A, P, x, 1, 1) - A'x,Inf) < 10eps()
end
