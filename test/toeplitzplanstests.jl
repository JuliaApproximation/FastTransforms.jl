using FastTransforms, Test
import FastTransforms: plan_uppertoeplitz!

@testset "ToeplitzPlan" begin
    P = plan_uppertoeplitz!([1,2,3])
    T = [1 2 3; 0 1 2; 0 0 1]
    x = randn(3)
    @test P * copy(x) ≈ T * x

    @testset "Matrix" begin
        X = randn(3,3)
        P = plan_uppertoeplitz!([1,2,3], size(X), 1)
        @test P * copy(X) ≈ T * X
        P = plan_uppertoeplitz!([1,2,3], size(X), 2)
        @test P * copy(X) ≈ X * T'

        P = plan_uppertoeplitz!([1,2,3], size(X))
        @test P * copy(X) ≈ T * X * T'

        X = randn(3,4)
        P1 = plan_uppertoeplitz!([1,2,3], size(X), 1)
        @test P1 * copy(X) ≈ T * X
        P2 = plan_uppertoeplitz!([1,2,3,4], size(X), 2)
        T̃ = [1 2 3 4; 0 1 2 3; 0 0 1 2; 0 0 0 1]
        @test P2 * copy(X) ≈ X * T̃'
        P = plan_uppertoeplitz!([1,2,3,4], size(X))
        @test P * copy(X) ≈ T * X * T̃'
    end

    @testset "Tensor" begin
        X = randn(3,3,3)
        P = plan_uppertoeplitz!([1,2,3], size(X), 1)
        PX = P * copy(X)
        for ℓ = 1:size(X,3)
            @test PX[:,:,ℓ] ≈ T*X[:,:,ℓ]
        end

        P = plan_uppertoeplitz!([1,2,3], size(X), 2)
        PX = P * copy(X)
        for ℓ = 1:size(X,3)
            @test PX[:,:,ℓ] ≈ X[:,:,ℓ]*T'
        end

        P = plan_uppertoeplitz!([1,2,3], size(X), 3)
        PX = P * copy(X)
        for j = 1:size(X,2)
            @test PX[:,j,:] ≈ X[:,j,:]*T'
        end
    end
end