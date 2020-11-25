using FastTransforms, Test

FastTransforms.set_num_threads(ceil(Int, Base.Sys.CPU_THREADS/2))

@testset "libfasttransforms" begin
    n = 64
    for T in (Float32, Float64)
        c = one(T) ./ (1:n)
        x = collect(-1 .+ 2*(0:n-1)/T(n))
        f = similar(x)
        @test FastTransforms.horner!(c, x, f) == f
        fd = T[sum(c[k]*x^(k-1) for k in 1:length(c)) for x in x]
        @test f ≈ fd
        @test FastTransforms.clenshaw!(c, x, f) == f
        fd = T[sum(c[k]*cos((k-1)*acos(x)) for k in 1:length(c)) for x in x]
        @test f ≈ fd
        A = T[(2k+one(T))/(k+one(T)) for k in 0:length(c)-1]
        B = T[zero(T) for k in 0:length(c)-1]
        C = T[k/(k+one(T)) for k in 0:length(c)]
        phi0 = ones(T, length(x))
        c = cheb2leg(c)
        @test FastTransforms.clenshaw!(c, A, B, C, x, phi0, f) == f
        @test f ≈ fd
    end

    α, β, γ, δ, λ, μ = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    function test_1d_plans(p1, p2, x)
        y = p1*x
        z = p2*y
        @test z ≈ x
        y = p1*x
        z = p1'y
        y = transpose(p1)*z
        z = transpose(p1)\y
        y = p1'\z
        z = p1\y
        @test z ≈ x
        y = p2*x
        z = p2'y
        y = transpose(p2)*z
        z = transpose(p2)\y
        y = p2'\z
        z = p2\y
        @test z ≈ x
        P = p1*I
        Q = p2*P
        @test Q ≈ I
        P = p1*I
        Q = p1'P
        P = transpose(p1)*Q
        Q = transpose(p1)\P
        P = p1'\Q
        Q = p1\P
        @test Q ≈ I
        P = p2*I
        Q = p2'P
        P = transpose(p2)*Q
        Q = transpose(p2)\P
        P = p2'\Q
        Q = p2\P
        @test Q ≈ I
    end

    for T in (Float32, Float64, Complex{Float32}, Complex{Float64}, BigFloat, Complex{BigFloat})
        x = T(1)./(1:n)
        Id = Matrix{T}(I, n, n)
        for (p1, p2) in ((plan_leg2cheb(Id), plan_cheb2leg(Id)),
                         (plan_ultra2ultra(Id, λ, μ), plan_ultra2ultra(Id, μ, λ)),
                         (plan_jac2jac(Id, α, β, γ, δ), plan_jac2jac(Id, γ, δ, α, β)),
                         (plan_lag2lag(Id, α, β), plan_lag2lag(Id, β, α)),
                         (plan_jac2ultra(Id, α, β, λ), plan_ultra2jac(Id, λ, α, β)),
                         (plan_jac2cheb(Id, α, β), plan_cheb2jac(Id, α, β)),
                         (plan_ultra2cheb(Id, λ), plan_cheb2ultra(Id, λ)))
            test_1d_plans(p1, p2, x)
        end
    end

    function test_nd_plans(p, ps, pa, A)
        B = copy(A)
        C = ps*(p*A)
        A = p\(pa*C)
        @test A ≈ B
    end

    A = sphones(Float64, n, 2n-1)
    p = plan_sph2fourier(A)
    ps = plan_sph_synthesis(A)
    pa = plan_sph_analysis(A)
    test_nd_plans(p, ps, pa, A)
    A = sphones(Float64, n, 2n-1) + im*sphones(Float64, n, 2n-1)
    p = plan_sph2fourier(A)
    ps = plan_sph_synthesis(A)
    pa = plan_sph_analysis(A)
    test_nd_plans(p, ps, pa, A)

    A = sphvones(Float64, n, 2n-1)
    p = plan_sphv2fourier(A)
    ps = plan_sphv_synthesis(A)
    pa = plan_sphv_analysis(A)
    test_nd_plans(p, ps, pa, A)
    A = sphvones(Float64, n, 2n-1) + im*sphvones(Float64, n, 2n-1)
    p = plan_sphv2fourier(A)
    ps = plan_sphv_synthesis(A)
    pa = plan_sphv_analysis(A)
    test_nd_plans(p, ps, pa, A)

    A = diskones(Float64, n, 4n-3)
    p = plan_disk2cxf(A, α, β)
    ps = plan_disk_synthesis(A)
    pa = plan_disk_analysis(A)
    test_nd_plans(p, ps, pa, A)
    A = diskones(Float64, n, 4n-3) + im*diskones(Float64, n, 4n-3)
    p = plan_disk2cxf(A, α, β)
    ps = plan_disk_synthesis(A)
    pa = plan_disk_analysis(A)
    test_nd_plans(p, ps, pa, A)

    A = rectdiskones(Float64, n, n)
    p = plan_rectdisk2cheb(A, β)
    ps = plan_rectdisk_synthesis(A)
    pa = plan_rectdisk_analysis(A)
    test_nd_plans(p, ps, pa, A)
    A = rectdiskones(Float64, n, n) + im*rectdiskones(Float64, n, n)
    p = plan_rectdisk2cheb(A, β)
    ps = plan_rectdisk_synthesis(A)
    pa = plan_rectdisk_analysis(A)
    test_nd_plans(p, ps, pa, A)

    A = triones(Float64, n, n)
    p = plan_tri2cheb(A, α, β, γ)
    ps = plan_tri_synthesis(A)
    pa = plan_tri_analysis(A)
    test_nd_plans(p, ps, pa, A)
    A = triones(Float64, n, n) + im*triones(Float64, n, n)
    p = plan_tri2cheb(A, α, β, γ)
    ps = plan_tri_synthesis(A)
    pa = plan_tri_analysis(A)
    test_nd_plans(p, ps, pa, A)

    A = tetones(Float64, n, n, n)
    p = plan_tet2cheb(A, α, β, γ, δ)
    ps = plan_tet_synthesis(A)
    pa = plan_tet_analysis(A)
    test_nd_plans(p, ps, pa, A)
    A = tetones(Float64, n, n, n) + im*tetones(Float64, n, n, n)
    p = plan_tet2cheb(A, α, β, γ, δ)
    ps = plan_tet_synthesis(A)
    pa = plan_tet_analysis(A)
    test_nd_plans(p, ps, pa, A)

    A = spinsphones(Complex{Float64}, n, 2n-1, 2) + im*spinsphones(Complex{Float64}, n, 2n-1, 2)
    p = plan_spinsph2fourier(A, 2)
    ps = plan_spinsph_synthesis(A, 2)
    pa = plan_spinsph_analysis(A, 2)
    test_nd_plans(p, ps, pa, A)
end
