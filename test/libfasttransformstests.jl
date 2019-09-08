using FastTransforms, Test

FastTransforms.set_num_threads(ceil(Int, Base.Sys.CPU_THREADS/2))

@testset "libfasttransforms" begin
    n = 64
    α, β, γ, δ, λ, μ = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    for T in (Float32, Float64, BigFloat)
        x = T(1)./(1:n)
        Id = Matrix{T}(I, n, n)
        for (p1, p2) in ((plan_leg2cheb(Id), plan_cheb2leg(Id)),
                          (plan_ultra2ultra(Id, T(λ), T(μ)), plan_ultra2ultra(Id, T(μ), T(λ))),
                          (plan_jac2jac(Id, T(α), T(β), T(γ), T(δ)), plan_jac2jac(Id, T(γ), T(δ), T(α), T(β))),
                          (plan_lag2lag(Id, T(α), T(β)), plan_lag2lag(Id, T(β), T(α))),
                          (plan_jac2ultra(Id, T(α), T(β), T(λ)), plan_ultra2jac(Id, T(λ), T(α), T(β))),
                          (plan_jac2cheb(Id, T(α), T(β)), plan_cheb2jac(Id, T(α), T(β))),
                          (plan_ultra2cheb(Id, T(λ)), plan_cheb2ultra(Id, T(λ))))
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
            P = p1*Id
            Q = p2*P
            @test Q ≈ Id
            P = p1*Id
            Q = p1'P
            P = transpose(p1)*Q
            Q = transpose(p1)\P
            P = p1'\Q
            Q = p1\P
            @test Q ≈ Id
            P = p2*Id
            Q = p2'P
            P = transpose(p2)*Q
            Q = transpose(p2)\P
            P = p2'\Q
            Q = p2\P
            @test Q ≈ Id
        end
    end

    A = sphones(Float64, n, 2n-1)
    p = plan_sph2fourier(A)
    ps = plan_sph_synthesis(A)
    pa = plan_sph_analysis(A)
    B = copy(A)
    C = ps*(p*A)
    A = p\(pa*C)
    @test A ≈ B

    A = sphvones(Float64, n, 2n-1)
    p = plan_sphv2fourier(A)
    ps = plan_sphv_synthesis(A)
    pa = plan_sphv_analysis(A)
    B = copy(A)
    C = ps*(p*A)
    A = p\(pa*C)
    @test A ≈ B

    A = diskones(Float64, n, 4n-3)
    p = plan_disk2cxf(A)
    ps = plan_disk_synthesis(A)
    pa = plan_disk_analysis(A)
    B = copy(A)
    C = ps*(p*A)
    A = p\(pa*C)
    @test A ≈ B

    A = triones(Float64, n, n)
    p = plan_tri2cheb(A, α, β, γ)
    ps = plan_tri_synthesis(A)
    pa = plan_tri_analysis(A)
    B = copy(A)
    C = ps*(p*A)
    A = p\(pa*C)
    @test A ≈ B

    A = tetones(Float64, n, n, n)
    p = plan_tet2cheb(A, α, β, γ, δ)
    ps = plan_tet_synthesis(A)
    pa = plan_tet_analysis(A)
    B = copy(A)
    C = ps*(p*A)
    A = p\(pa*C)
    @test A ≈ B
end
