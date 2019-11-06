using BinaryProvider
import Libdl

version = v"0.2.9"

if arch(platform_key_abi()) != :x86_64
    @warn "FastTransforms has only been tested on x86_64 architectures."
end

const extension = Sys.isapple() ? "dylib" : Sys.islinux() ? "so" : Sys.iswindows() ? "dll" : ""

print_error() = error(
    "FastTransforms could not be properly installed.\n Please check that you have all dependencies installed. " *
    "Sample installation of dependencies:\n" *
    print_platform_error(platform_key_abi()) * "$(platform_key_abi())"
)

print_platform_error(p::Platform) = "On $(BinaryProvider.platform_name(p)), please consider opening a pull request to add support.\n"
print_platform_error(p::MacOS) = "On MacOS\n\tbrew install gcc@8 fftw mpfr\n"
print_plaftorm_error(p::Linux) = "On Linux\n\tsudo apt-get install gcc-8 libblas-dev libopenblas-base libfftw3-dev libmpfr-dev\n"
print_plaftorm_error(p::Windows) = "On Windows\n\tvcpkg install openblas:x64-windows fftw3[core,threads]:x64-windows mpir:x64-windows mpfr:x64-windows\n"

print_error()
# Rationale is as follows: The build is pretty fast, so on Linux it is typically easiest
# to just use the gcc of the system to build the library and include it. On MacOS, however,
# we need to actually install a gcc first, because Apple's OS comes only shipped with clang,
# so here we download the binary.
ft_build_from_source = get(ENV, "FT_BUILD_FROM_SOURCE", Sys.isapple() ? "false" : "true")
if ft_build_from_source == "true"
    make = Sys.iswindows() ? "mingw32-make" : "make"
    compiler = Sys.isapple() ? "CC=gcc-8" : "CC=gcc"
    flags = Sys.isapple() ? "FT_USE_APPLEBLAS=1" : Sys.iswindows() ? "FT_FFTW_WITH_COMBINED_THREADS=1" : ""
    script = """
        set -e
        set -x
        if [ -d "FastTransforms" ]; then
            cd FastTransforms
            git fetch
            git checkout v$version
            cd ..
        else
            git clone -b v$version https://github.com/MikaelSlevinsky/FastTransforms.git FastTransforms
        fi
        cd FastTransforms
        $make lib $compiler $flags
        cd ..
        mv -f FastTransforms/libfasttransforms.$extension libfasttransforms.$extension
    """
    try
        run(`bash -c $(script)`)
    catch
        print_error()
    end
    println("FastTransforms built from source.")
else
    const GCC = BinaryProvider.detect_compiler_abi().gcc_version
    namemap = Dict(:gcc4 => "gcc-4.9", :gcc5 => "gcc-5", :gcc6 => "gcc-6",
                   :gcc7 => "gcc-7", :gcc8 => "gcc-8", :gcc9 => "gcc-9")
    if !(GCC in keys(namemap))
        error("Please ensure you have a version of gcc from gcc-4.9 to gcc-9.")
    end
    try
        download("https://github.com/MikaelSlevinsky/FastTransforms/releases/download/" *
                 "v$version/libfasttransforms.v$version.$(namemap[GCC]).$extension",
                 joinpath(dirname(@__DIR__), "deps", "libfasttransforms.$extension"))
    catch
        print_error()
    end
    println("FastTransforms installed by downloading binaries.")
end
