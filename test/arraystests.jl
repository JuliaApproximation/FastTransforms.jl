using FastTransforms, Test
import FastTransforms: ArrayPlan, NDimsPlan

@testset "Array transform"  begin
    @testset "ArrayPlan" begin
        c = randn(5,20,10)
        F = plan_cheb2leg(c)
        FT = ArrayPlan(F, c)

        @test size(FT) == size(c)
        @test size(FT,1) == size(c,1)
        @test size(FT,1,2) == (size(c,1), size(c,2))

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
        @test_throws ErrorException("Different size in dims axes not yet implemented in N-dimensional transform.") NDimsPlan(ArrayPlan(plan_cheb2leg(c), c), (1,2))        

        c = randn(20,20,5);
        F = plan_cheb2leg(c)
        FT = ArrayPlan(F, c)
        P = NDimsPlan(FT, (1,2))

        @test size(P) == size(c)
        @test size(P,1) == size(c,1)
        @test size(P,1,2) == (size(c,1), size(c,2))


        f = similar(c);
        for k in axes(f,3)
            f[:,:,k] = (F*(F*c[:,:,k])')'
        end
        @test f ≈ P*c
        @test c ≈ P\f

        c = randn(10,5,10,60)
        F = plan_cheb2leg(c)
        P = NDimsPlan(ArrayPlan(F, c), (1,3))
        f = similar(c)
        for i in axes(f,2), j in axes(f,4)
            f[:,i,:,j] = (F*(F*c[:,i,:,j])')'
        end
        @test f ≈ P*c
        @test c ≈ P\f
    end
end


