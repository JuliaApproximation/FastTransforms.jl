using FastTransforms, Test

@testset "ToeplitzHankel" begin
    @test FastTransforms.th_leg2cheb([1.0,2,3,4,5]) ≈ leg2cheb([1.0,2,3,4,5])
    @test FastTransforms.th_cheb2leg([1.0,2,3,4,5]) ≈ cheb2leg([1.0,2,3,4,5])
    @test FastTransforms.th_leg2chebu([1.0,2,3,4,5]) ≈ ultra2ultra([1.0,2,3,4,5], 0.5, 1.0)


    X = randn(5,4)
    FastTransforms.th_leg2cheb(X)
end

