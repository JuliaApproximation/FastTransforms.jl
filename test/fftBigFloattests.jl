using DSP, FastTransforms, Test

@testset "BigFloat Convolution" begin
    s = big(1) ./ (1:10)
    s64 = Float64.(s)
    @test Float64.(conv(s, s)) ≈ conv(s64, s64)
    @test s == big(1) ./ (1:10) #67, ensure conv doesn't overwrite input
    @test all(s64 .=== Float64.(big(1) ./ (1:10)))
end
