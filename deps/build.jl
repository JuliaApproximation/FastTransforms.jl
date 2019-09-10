using BinaryProvider

if Sys.isapple()
    run(`brew upgrade`)
    run(`brew install gcc@8 fftw mpfr`)
    const libfasttransforms = joinpath(dirname(@__DIR__), "deps", "libfasttransforms.dylib")
    GCC = BinaryProvider.detect_compiler_abi().gcc_version
    println("Building with ", GCC, ".")
    const release = "https://github.com/MikaelSlevinsky/FastTransforms/releases/download/v0.2.6/libfasttransforms.v0.2.6"
    if GCC == :gcc4
        download(release*".gcc-4.9.dylib", libfasttransforms)
    elseif GCC == :gcc5
        download(release*".gcc-5.dylib", libfasttransforms)
    elseif GCC == :gcc6
        download(release*".gcc-6.dylib", libfasttransforms)
    elseif GCC == :gcc7
        download(release*".gcc-7.dylib", libfasttransforms)
    elseif GCC == :gcc8
        download(release*".gcc-8.dylib", libfasttransforms)
    elseif GCC == :gcc9
        download(release*".gcc-9.dylib", libfasttransforms)
    else
        @warn "Please ensure you have a version of gcc from gcc-4.9 to gcc-9."
    end
elseif Sys.islinux()
    run(`apt-get gcc-8 libblas-dev libopenblas-base libfftw3-dev libmpfr-dev`)
    const libfasttransforms = joinpath(dirname(@__DIR__), "deps", "libfasttransforms.so")
    if arch(platform_key_abi()) != :x86_64
        @warn "FastTransforms only has compiled binaries for x86_64 architectures."
    else
        GCC = BinaryProvider.detect_compiler_abi().gcc_version
        println("Building with ", GCC, ".")
        const release = "https://github.com/MikaelSlevinsky/FastTransforms/releases/download/v0.2.6/libfasttransforms.v0.2.6"
        if GCC == :gcc4
            download(release*".gcc-4.9.so", libfasttransforms)
        elseif GCC == :gcc5
            download(release*".gcc-5.so", libfasttransforms)
        elseif GCC == :gcc6
            download(release*".gcc-6.so", libfasttransforms)
        elseif GCC == :gcc7
            download(release*".gcc-7.so", libfasttransforms)
        elseif GCC == :gcc8
            download(release*".gcc-8.so", libfasttransforms)
        elseif GCC == :gcc9
            download(release*".gcc-9.so", libfasttransforms)
        else
            @warn "Please ensure you have a version of gcc from gcc-4.9 to gcc-9."
        end
    end
else
    @warn "FastTransforms is not properly installed."
end
