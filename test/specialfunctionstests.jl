using FastTransforms
using Compat.Test

@testset "Pochhammer" begin
    @test pochhammer(2,3) == 24
    @test pochhammer(0.5,3) == 0.5*1.5*2.5
    @test pochhammer(0.5,0.5) == 1/sqrtpi
end
