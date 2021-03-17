using BinaryProvider
import Libdl

const extension = Sys.isapple() ? "dylib" : Sys.islinux() ? "so" : Sys.iswindows() ? "dll" : ""

print_error() = error(
    "FastTransforms could not be properly installed.\n Please check that you have all dependencies installed. " *
    "Sample installation of dependencies:\n" *
    print_platform_error(platform_key_abi())
)

print_platform_error(p::Platform) = "On $(BinaryProvider.platform_name(p)), please consider opening a pull request to add support to build from source.\n"
print_platform_error(p::MacOS) = "On MacOS\n\tbrew install libomp fftw mpfr\n"
print_platform_error(p::Linux) = "On Linux\n\tsudo apt-get install libomp-dev libblas-dev libopenblas-base libfftw3-dev libmpfr-dev\n"
print_platform_error(p::Windows) = "On Windows\n\tvcpkg install openblas:x64-windows fftw3[core,threads]:x64-windows mpir:x64-windows mpfr:x64-windows\n"

ft_build_from_source = get(ENV, "FT_BUILD_FROM_SOURCE", "false")
if ft_build_from_source == "true"
    make = Sys.iswindows() ? "mingw32-make" : "make"
    flags = Sys.isapple() ? "FT_USE_APPLEBLAS=1" : Sys.iswindows() ? "FT_FFTW_WITH_COMBINED_THREADS=1" : ""
    script = """
        set -e
        set -x
        if [ -d "FastTransforms" ]; then
            cd FastTransforms
            git fetch
            git checkout master
            git pull
            cd ..
        else
            git clone https://github.com/MikaelSlevinsky/FastTransforms.git FastTransforms
        fi
        cd FastTransforms
        $make assembly
        $make lib $flags
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
    println("FastTransforms using precompiled binaries.")
end
