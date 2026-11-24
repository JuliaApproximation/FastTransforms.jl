using FastTransforms, LinearAlgebra, Test
import FastTransforms: ArrayPlan, NDimsPlan

@testset "Array transform"  begin
    @testset "ArrayPlan" begin
        c = randn(5,20,10)
        F = plan_cheb2leg(c)
        FT = ArrayPlan(F, c)

        @test size(FT) == size(c)

        f = similar(c);
        for k in axes(c,3)
            f[:,:,k] = (F*c[:,:,k])
        end
        @test f ≈ FT*c
        @test c ≈ FT\f

        F = plan_cheb2leg(Vector{Float64}(axes(c,2)))
        FT = ArrayPlan(F, c, (2,))
        for k in axes(c,3)
            f[:,:,k] = (F*c[:,:,k]')'
        end
        @test f ≈ FT*c
        @test c ≈ FT\f
    end

    @testset "NDimsPlan" begin
        c = randn(20,10,20)
        @test_throws ErrorException("Different size in dims axes not yet implemented in N-dimensional transform.") NDimsPlan(ArrayPlan(plan_cheb2leg(c), c), size(c), (1,2))        

        c = randn(5,20)
        F = plan_cheb2leg(c)
        FT = ArrayPlan(F, c)
        P = NDimsPlan(F, size(c), (1,))
        @test F*c ≈ FT*c ≈ P*c

        c = randn(20,20,5);
        F = plan_cheb2leg(c)
        FT = ArrayPlan(F, c)
        P = NDimsPlan(FT, size(c), (1,2))

        @test size(P) == size(c)

        f = similar(c);
        for k in axes(f,3)
            f[:,:,k] = (F*(F*c[:,:,k])')'
        end
        @test f ≈ P*c
        @test c ≈ P\f

        c = randn(5,10,10,60)
        F = plan_cheb2leg(randn(10))
        P = NDimsPlan(F, size(c), (2,3))
        f = similar(c)
        for i in axes(f,1), j in axes(f,4)
            f[i,:,:,j] = (F*(F*c[i,:,:,j])')'
        end
        @test f ≈ P*c
        @test c ≈ P\f
    end

    @testset "NDimsPlan on slices" begin
        @testset "vector transform" begin
            c = randn(5,10,10)
            F = plan_cheb2leg(randn(10))
            P = NDimsPlan(F, size(c), ((2,), (3,)))
            @test P*c ≈ NDimsPlan(F, size(c), (2,3))*c
            @test P\(P*c) ≈ c
        end

        n, α, β = 5, 0.5, 0.25
        m = 4n-3
        F = plan_disk2cxf(Float64, n, α, β)

        @testset "disk2cxf" begin
            A = randn(n, 3, m)
            P = plan_disk2cxf(A, α, β, ((1,3),))
            @test P isa NDimsPlan
            @test size(P) == size(A)
            @test contains(sprint(show, P), "NDimsPlan for 5×3×17-element array along dims ((1, 3),)")
            B = similar(A)
            for k in axes(A,2)
                B[:,k,:] = F*A[:,k,:]
            end
            @test P*A ≈ B
            @test P\B ≈ A
            @test disk2cxf(A, α, β, ((1,3),)) ≈ B
            @test cxf2disk(B, α, β, ((1,3),)) ≈ A
            @test P'*A ≈ P'*complex(A) ≈ transpose(P)*A
            @test P*complex.(A, A) ≈ complex.(B, B)
            @test P\complex.(B, B) ≈ complex.(A, A)

            # default parameters
            F₀ = plan_disk2cxf(Float64, n)
            for k in axes(A,2)
                B[:,k,:] = F₀*A[:,k,:]
            end
            @test plan_disk2cxf(A, ((1,3),))*A ≈ B
            @test plan_disk2cxf(A, 0, ((1,3),))*A ≈ B

            # rows along dimension 3, columns along dimension 1
            Aᵀ = permutedims(A, (3,2,1))
            @test plan_disk2cxf(Aᵀ, α, β, ((3,1),))*Aᵀ ≈ permutedims(P*A, (3,2,1))

            # a matrix is a 2-array
            @test plan_disk2cxf(A[:,1,:], α, β, ((1,2),))*A[:,1,:] ≈ F*A[:,1,:]

            # 4-array: A[:,k,:,j] then A[k,:,j,:]
            A = randn(n, n, m, m)
            P = plan_disk2cxf(A, α, β, ((1,3), (2,4)))
            B = similar(A)
            for k in axes(A,2), j in axes(A,4)
                B[:,k,:,j] = F*A[:,k,:,j]
            end
            for k in axes(A,1), j in axes(A,3)
                B[k,:,j,:] = F*B[k,:,j,:]
            end
            @test P*A ≈ B
            @test P\B ≈ A

            # overlapping dims do not commute, so \ and ' must apply the factors in reverse order
            A, C = randn(n, n, m), randn(n, n, m)
            P = plan_disk2cxf(A, α, β, ((1,3), (2,3)))
            B = similar(A)
            for k in axes(A,2)
                B[:,k,:] = F*A[:,k,:]
            end
            for k in axes(A,1)
                B[k,:,:] = F*B[k,:,:]
            end
            @test P*A ≈ B
            @test P\B ≈ A
            @test dot(P'*C, A) ≈ dot(C, P*A)

            A = randn(n, 3, m)
            @test_throws ArgumentError plan_disk2cxf(A, (1,3))
            @test_throws ArgumentError plan_disk2cxf(A, ((1,4),))
            @test_throws ArgumentError plan_disk2cxf(A, ((1,1),))
            @test_throws ErrorException plan_disk2cxf(randn(n, n+1, m, m+4), ((1,3), (2,4)))
            @test_throws DimensionMismatch plan_disk2cxf(A, ((1,3),))*randn(n, 4, m)
        end

        @testset "disk analysis and synthesis" begin
            Fa, Fs = plan_disk_analysis(Float64, n, m), plan_disk_synthesis(Float64, n, m)

            A = randn(n, 3, m)
            pa, ps = plan_disk_analysis(A, ((1,3),)), plan_disk_synthesis(A, ((1,3),))
            Ba, Bs = similar(A), similar(A)
            for k in axes(A,2)
                Ba[:,k,:] = Fa*A[:,k,:]
                Bs[:,k,:] = Fs*A[:,k,:]
            end
            @test pa*A ≈ Ba
            @test ps*A ≈ Bs
            @test pa*(ps*A) ≈ A
            @test inv(pa) isa typeof(ps)
            @test inv(ps) isa typeof(pa)
            @test inv(pa)*A ≈ ps*A
            @test inv(ps)*A ≈ pa*A
            @test pa*complex.(A, A) ≈ complex.(Ba, Ba)
            for k in axes(A,2)
                Ba[:,k,:] = Fa'*A[:,k,:]
            end
            @test pa'*A ≈ transpose(pa)*A ≈ Ba
            @test plan_disk_analysis(A, ((1,3),); flags=FastTransforms.FFTW.MEASURE)*A ≈ pa*A

            A = randn(n, n, m, m)
            pa, ps = plan_disk_analysis(A, ((1,3), (2,4))), plan_disk_synthesis(A, ((1,3), (2,4)))
            B = similar(A)
            for k in axes(A,2), j in axes(A,4)
                B[:,k,:,j] = Fa*A[:,k,:,j]
            end
            for k in axes(A,1), j in axes(A,3)
                B[k,:,j,:] = Fa*B[k,:,j,:]
            end
            @test pa*A ≈ B
            @test ps*(pa*A) ≈ A
            @test inv(pa)*B ≈ A
        end

        @testset "all transforms" begin
            n, K = 5, 3
            α, β, γ, δ, ρ, s = -0.1, -0.2, -0.3, -0.4, 2/3, 2

            # compare plan_f(A, params..., dims) with applying plan_f(slice, params...) slice by slice
            function test_slices(plan_f, params, A, dims, slice; inverse)
                P = plan_f(A, params..., dims)
                F = plan_f(A[slice(1)...], params...)
                @test P isa NDimsPlan
                for (Q, G) in ((P, F), (P', F'), (transpose(P), transpose(F)))
                    B = similar(A)
                    for k in axes(A, findfirst(i -> i ∉ only(dims), 1:ndims(A)))
                        B[slice(k)...] = G*A[slice(k)...]
                    end
                    @test Q*A ≈ B
                    if inverse
                        C = similar(A)
                        for k in axes(A, findfirst(i -> i ∉ only(dims), 1:ndims(A)))
                            C[slice(k)...] = G\B[slice(k)...]
                        end
                        @test Q\B ≈ C
                    end
                end
            end

            # (plan, parameters, number of columns, element type, whether the plan supports \)
            for (plan_f, params, m, S, inverse) in ((plan_sph2fourier, (), 2n-1, Float64, true),
                                                    (plan_sphv2fourier, (), 2n-1, Float64, true),
                                                    (plan_disk2cxf, (α, β), 4n-3, Float64, true),
                                                    (plan_ann2cxf, (α, β, 0, ρ), 4n-3, Float64, true),
                                                    (plan_rectdisk2cheb, (β,), n, Float64, true),
                                                    (plan_tri2cheb, (α, β, γ), n, Float64, true),
                                                    (plan_spinsph2fourier, (s,), 2n-1, ComplexF64, true),
                                                    (plan_sph_synthesis, (), 2n-1, Float64, false),
                                                    (plan_sph_analysis, (), 2n-1, Float64, false),
                                                    (plan_sphv_synthesis, (), 2n-1, Float64, false),
                                                    (plan_sphv_analysis, (), 2n-1, Float64, false),
                                                    (plan_disk_synthesis, (), 4n-3, Float64, false),
                                                    (plan_disk_analysis, (), 4n-3, Float64, false),
                                                    (plan_annulus_synthesis, (ρ,), 4n-3, Float64, false),
                                                    (plan_annulus_analysis, (ρ,), 4n-3, Float64, false),
                                                    (plan_rectdisk_synthesis, (), n, Float64, false),
                                                    (plan_rectdisk_analysis, (), n, Float64, false),
                                                    (plan_tri_synthesis, (), n, Float64, false),
                                                    (plan_tri_analysis, (), n, Float64, false),
                                                    (plan_spinsph_synthesis, (s,), 2n-1, ComplexF64, false),
                                                    (plan_spinsph_analysis, (s,), 2n-1, ComplexF64, false))
                @testset "$plan_f" begin
                    test_slices(plan_f, params, randn(S, n, K, m), ((1,3),), k -> (:, k, :); inverse)
                    @test_throws ArgumentError plan_f(randn(S, n, K, m), params..., ((1,2,3),))
                end
            end

            for (plan_f, params, inverse) in ((plan_tet2cheb, (α, β, γ, δ), true),
                                              (plan_tet_synthesis, (), false),
                                              (plan_tet_analysis, (), false))
                @testset "$plan_f" begin
                    test_slices(plan_f, params, randn(n, n, K, n), ((1,2,4),), k -> (:, :, k, :); inverse)
                    @test_throws ArgumentError plan_f(randn(n, n, K, n), params..., ((1,2),))
                end
            end

            # the inverse transforms accept dims too
            A = randn(n, K, n)
            P = plan_tri2cheb(A, α, β, γ, ((1,3),))
            @test cheb2tri(P*A, α, β, γ, ((1,3),)) ≈ A
            @test tri2cheb(A, α, β, γ, ((1,3),)) ≈ P*A
            A = randn(n, K, 2n-1)
            @test fourier2sph(sph2fourier(A, ((1,3),)), ((1,3),)) ≈ plan_sph2fourier(A, ((1,3),))\(plan_sph2fourier(A, ((1,3),))*A)
        end
    end
end


