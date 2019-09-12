using BinaryProvider

if arch(platform_key_abi()) != :x86_64
    @warn "FastTransforms has only been tested on x86_64 architectures."
end

ft_build_from_source = get(ENV, "FT_BUILD_FROM_SOURCE", "false")

if ft_build_from_source == "true"
    println("Building from source.")
    if Sys.isapple()
        script = raw"""
        brew update
        brew install gcc@8 fftw mpfr
        rm -rf FastTransforms
        git clone -b v0.2.7 https://github.com/MikaelSlevinsky/FastTransforms.git FastTransforms
        cd FastTransforms
        make lib CC=gcc-8 FT_USE_APPLEBLAS=1
        cd ..
        ln -sf FastTransforms/libfasttransforms.dylib libfasttransforms.dylib
        """
        run(`/bin/bash -c $(script)`)
    elseif Sys.islinux()
        script = raw"""
        sudo apt-get update
        sudo add-apt-repository ppa:ubuntu-toolchain-r/test
        sudo apt-get install gcc-8 libblas-dev libopenblas-base libfftw3-dev libmpfr-dev
        rm -rf FastTransforms
        git clone -b v0.2.7 https://github.com/MikaelSlevinsky/FastTransforms.git FastTransforms
        cd FastTransforms
        make lib CC=gcc-8
        cd ..
        ln -sf FastTransforms/libfasttransforms.so libfasttransforms.so
        """
        run(`/bin/bash -c $(script)`)
    else
        @warn "FastTransforms could not be built from source with the current build.jl script. Have you considered filing an issue? https://github.com/JuliaApproximation/FastTransforms.jl/issues"
    end
else
    println("Installing by downloading binaries.")
    if Sys.isapple()
        run(`brew update`)
        run(`brew install gcc@8 fftw mpfr`)
        const libfasttransforms = joinpath(dirname(@__DIR__), "deps", "libfasttransforms.dylib")
        GCC = BinaryProvider.detect_compiler_abi().gcc_version
        println("Building with ", GCC, ".")
        const release = "https://github.com/MikaelSlevinsky/FastTransforms/releases/download/v0.2.7/libfasttransforms.v0.2.7"
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
        run(`sudo add-apt-repository ppa:ubuntu-toolchain-r/test`)
        run(`sudo apt-get update`)
        run(`sudo apt-get install gcc-8 libblas-dev libopenblas-base libfftw3-dev libmpfr-dev`)
        const libfasttransforms = joinpath(dirname(@__DIR__), "deps", "libfasttransforms.so")
        GCC = BinaryProvider.detect_compiler_abi().gcc_version
        println("Building with ", GCC, ".")
        const release = "https://github.com/MikaelSlevinsky/FastTransforms/releases/download/v0.2.7/libfasttransforms.v0.2.7"
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
    else
        @warn "FastTransforms could not be installed by downloading binaries. Have you tried building from source?"
    end
end
