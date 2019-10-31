using BinaryProvider
import Libdl

version = "0.2.7"

if arch(platform_key_abi()) != :x86_64
    @warn "FastTransforms has only been tested on x86_64 architectures."
end
if !Sys.islinux() && !Sys.isapple()
    error("Sorry ... unsupported OS. Feel free to file a PR to add support.")
end

const extension = Sys.isapple() ? "dylib" : "so"
print_error() = error(
    "FastTransforms could not be properly installed.\nCheck you have all dependencies installed." *
   " To install the dependencies you can use:\n" *
   "On Ubuntu / Debian \n" *
   "   sudo apt install gcc libblas-dev libopenblas-base libfftw3-dev libmpfr-dev\n" *
   "On MacOS \n" *
   "   brew install gcc@8 fftw mpfr\n"
)

# Rationale is as follows: The build is pretty fast, so on Linux it is typically easiest
# to just use the gcc of the system to build the library and include it. On MacOS, however,
# we need to actually install a gcc first, because Apple's OS comes only shipped with clang,
# so here we download the binary.
ft_build_from_source = get(ENV, "FT_BUILD_FROM_SOURCE", Sys.isapple() ? "false" : "true")
if ft_build_from_source == "true"
    println("Building from source.")

    extra = Sys.isapple() ? "FT_USE_APPLEBLAS=1" : ""
    script = """
        set -e
        set -x

        if [ -d "FastTransforms" ]; then
            cd FastTransforms
            git fetch
            git reset --hard
            git checkout -b v$version
            cd ..
        else
            git clone -b v$version https://github.com/MikaelSlevinsky/FastTransforms.git FastTransforms
        fi
        ln -sf FastTransforms/libfasttransforms.$extension libfasttransforms.$extension

        echo
        echo

        cd FastTransforms
        make clean
        make lib $extra
    """

    try
        run(`/bin/bash -c $(script)`)
    catch IOError
        print_error()
    end
else
    println("Installing by downloading binaries.")

    const GCC = BinaryProvider.detect_compiler_abi().gcc_version
    namemap = Dict(:gcc4 => "gcc-4.9", :gcc5 => "gcc-5", :gcc6 => "gcc-6",
                   :gcc7 => "gcc-7", :gcc8 => "gcc-8", :gcc9 => "gcc-9")
    if !(GCC in keys(namemap))
        error("Please ensure you have a version of gcc from gcc-4.9 to gcc-9.")
    end
    download("https://github.com/MikaelSlevinsky/FastTransforms/releases/download/" *
             "v$version/libfasttransforms.v$version.$(namemap[GCC]).$extension",
             joinpath(dirname(@__DIR__), "deps", "libfasttransforms.$extension"))
end

const lft_directiory = joinpath(dirname(@__DIR__), "deps")
const libfasttransforms = Libdl.find_library("libfasttransforms", [lft_directiory])
if libfasttransforms === nothing || length(libfasttransforms) == 0
    print_error()
end
