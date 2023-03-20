using FastTransforms, Test
import FastTransforms: plan_uppertoeplitz!

@testset "ToeplitzPlan" begin
    P = plan_uppertoeplitz!([1,2,3])
    T = [1 2 3; 0 1 2; 0 0 1]
    x = randn(3)
    @test P * copy(x) ≈ T * x

    X = randn(3,3)
    P = plan_uppertoeplitz!([1,2,3], size(X), 1)
    @test P * copy(X) ≈ T * X
    P = plan_uppertoeplitz!([1,2,3], size(X), 2)
    @test P * copy(X) ≈ X * T'

    P = plan_uppertoeplitz!([1,2,3], size(X))
    @test P * copy(X) ≈ T * X * T'
end