using FastTransforms, Test

FastTransforms.set_num_threads(ceil(Int, Base.Sys.CPU_THREADS/2))

@testset "libfasttransforms" begin
    n = 64
    α, β, γ, δ, λ, μ = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    for T in (Float32, Float64, BigFloat)
        for (p1, p2) in ((plan_leg2cheb(T, n), plan_cheb2leg(T, n)),
                          (plan_ultra2ultra(T, n, T(λ), T(μ)), plan_ultra2ultra(T, n, T(μ), T(λ))),
                          (plan_jac2jac(T, n, T(α), T(β), T(γ), T(δ)), plan_jac2jac(T, n, T(γ), T(δ), T(α), T(β))),
                          (plan_lag2lag(T, n, T(α), T(β)), plan_lag2lag(T, n, T(β), T(α))),
                          (plan_jac2ultra(T, n, T(α), T(β), T(λ)), plan_ultra2jac(T, n, T(λ), T(α), T(β))),
                          (plan_jac2cheb(T, n, T(α), T(β)), plan_cheb2jac(T, n, T(α), T(β))),
                          (plan_ultra2cheb(T, n, T(λ)), plan_cheb2ultra(T, n, T(λ))))
            Id = Matrix{T}(I, n, n)
            P = deepcopy(Id)
            lmul!(p1, P)
            lmul!(p2, P)
            @test P ≈ Id
            lmul!(p1, P)
            lmul!(p1', P)
            ldiv!(p1', P)
            ldiv!(p1, P)
            @test P ≈ Id
            lmul!(p1, P)
            lmul!(p1', P)
            ldiv!(p1', P)
            ldiv!(p1, P)
            @test P ≈ Id
        end
    end

    p = plan_sph2fourier(Float64, n)
    ps = plan_sph_synthesis(Float64, n, 2n-1)
    pa = plan_sph_analysis(Float64, n, 2n-1)
    A = sphones(Float64, n, 2n-1)
    B = copy(A)
    lmul!(p, A)
    lmul!(ps, A)
    lmul!(pa, A)
    ldiv!(p, A)
    @test A ≈ B

    p = plan_sphv2fourier(Float64, n)
    ps = plan_sphv_synthesis(Float64, n, 2n-1)
    pa = plan_sphv_analysis(Float64, n, 2n-1)
    A = sphvones(Float64, n, 2n-1)
    B = copy(A)
    lmul!(p, A)
    lmul!(ps, A)
    lmul!(pa, A)
    ldiv!(p, A)
    @test A ≈ B

    p = plan_disk2cxf(Float64, n)
    ps = plan_disk_synthesis(Float64, n, 4n-3)
    pa = plan_disk_analysis(Float64, n, 4n-3)
    A = diskones(Float64, n, 4n-3)
    B = copy(A)
    lmul!(p, A)
    lmul!(ps, A)
    lmul!(pa, A)
    ldiv!(p, A)
    @test A ≈ B

    p = plan_tri2cheb(Float64, n, α, β, γ)
    ps = plan_tri_synthesis(Float64, n, n)
    pa = plan_tri_analysis(Float64, n, n)
    A = triones(Float64, n, n)
    B = copy(A)
    lmul!(p, A)
    lmul!(ps, A)
    lmul!(pa, A)
    ldiv!(p, A)
    @test A ≈ B

    p = plan_tet2cheb(Float64, n, α, β, γ, δ)
    ps = plan_tet_synthesis(Float64, n, n, n)
    pa = plan_tet_analysis(Float64, n, n, n)
    A = tetones(Float64, n, n, n)
    B = copy(A)
    lmul!(p, A)
    lmul!(ps, A)
    lmul!(pa, A)
    ldiv!(p, A)
    @test A ≈ B
end
