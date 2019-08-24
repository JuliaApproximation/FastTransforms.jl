if Sys.isapple()
    const libfasttransforms = joinpath(dirname(@__DIR__), "deps", "libfasttransforms.dylib")
    download("https://github.com/MikaelSlevinsky/FastTransforms/releases/download/v0.2.3/libfasttransforms.dylib", libfasttransforms)
    cd(joinpath(dirname(@__DIR__), "deps"))
    run(`ls`)
else
    warn("Didn't build properly")
end
