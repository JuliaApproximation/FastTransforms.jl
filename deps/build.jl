if Sys.isapple()
    const libfasttransforms = joinpath(dirname(@__DIR__), "deps", "libfasttransforms.dylib")
    download("https://github.com/MikaelSlevinsky/FastTransforms/releases/download/v0.2.3/libfasttransforms.dylib", libfasttransforms)
else
    @warn "Didn't build properly"
end
