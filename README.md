# NAME

Inline::CUDA - Inline NVIDIA's CUDA code and GPU processing from within any Perl script.

# VERSION

Version 0.16

# SYNOPSIS

**WARNING:** see section ["INSTALLATION"](#installation) for how to install this package.

**WARNING:** prior to installation, please install [https://github.com/hadjiprocopis/perl-nvidia2-ml](https://github.com/hadjiprocopis/perl-nvidia2-ml)

`Inline::CUDA` is a module that allows you to write Perl subroutines in C
or C++ with CUDA extensions.

Similarly to [Inline::C](https://metacpan.org/pod/Inline%3A%3AC),
`Inline::CUDA` is not meant to be used directly but rather in this way:

        #Firstly, specify some configuration options:

        use Inline CUDA => Config =>
        # optionally specify some options,
        # you don't have to
        # if they are already stored in a configuration file
        # which is consulted before running any CUDA program
        #    host_compiler_bindir => '/usr/local/gcc82/bin',
        #    cc => '/usr/local/gcc82/bin/gcc82',
        #    cxx => '/usr/local/gcc82/bin/g++82',
        #    ld => '/usr/local/gcc82/bin/gcc82',
        #    nvcc => '/usr/local/cuda/bin/nvcc',
        #    nvld => '/usr/local/cuda/bin/nvcc',
        # pass options to nvcc:
        #  this is how to deal with unknown compiler flags passed on to nvcc: pass them all to gcc
        #  only supported in nvcc versions 11+
        #    nvccflags => '--forward-unknown-to-host-compiler',
        #  do not check compiler version, use whatever the user wants
        #    nvccflags => '--allow-unsupported-compiler',
        # this will use CC or CXX depending on the language specified here
        # you can use C++ in your CUDA code, and there are tests in t/*
        # which check if c or c++ and show how to do this:
            host_code_language => 'c', # or 'c++' or 'cpp', case insensitive, see also cxx =>
        # optional extra Include and Lib dirs
            #inc => '-I...',
            #libs => '-L... -l...',
        # for debugging
            BUILD_NOISY => 1,
        # code will be left in ./_Inline/build/ after successful build
            clean_after_build => 0,
            warnings => 10,
          ;

        # and then, suck in code from __DATA__ and run it at runtime
        # notice that Inline->use(CUDA => <<'EOCUDA') is run at compiletime
        my $codestr;
        { local $/ = undef; $codestr = <DATA> }
        Inline->bind( CUDA => $codestr );

        if( do_add() ){ die "error running do_add()..." }

        1;
        __DATA__
        /* this is C code with CUDA extensions */
        #include <stdio.h>
        
        #define N 1000
        
        /* This is the CUDA Kernel which nvcc compiles: */
        __global__
        void add(int *a, int *b) {
                int i = blockIdx.x;
                if (i<N) b[i] = a[i]+b[i];
        }
        /* this function can be called from Perl.
           It returns 0 on success or 1 on failure.
           This simple code does not support passing parameters in,
           which is covered elsewhere.
        */
        int do_add() {
                cudaError_t err;
        
                // Create int arrays on the CPU.
                // ('h' stands for "host".)
                int ha[N], hb[N];
        
                // Create corresponding int arrays on the GPU.
                // ('d' stands for "device".)
                int *da, *db;
                if( (err=cudaMalloc((void **)&da, N*sizeof(int))) != cudaSuccess ){
                        fprintf(stderr, "do_add(): error, call to cudaMalloc() has failed for %zu bytes for da: %s\n",
                                N*sizeof(int), cudaGetErrorString(err)
                        );
                        return 1;
                }
                if( (err=cudaMalloc((void **)&db, N*sizeof(int))) != cudaSuccess ){
                        fprintf(stderr, "do_add(): error, call to cudaMalloc() has failed for %zu bytes for db: %s\n",
                                N*sizeof(int), cudaGetErrorString(err)
                        );
                        return 1;
                }
        
                // Initialise the input data on the CPU.
                for (int i = 0; i<N; ++i) ha[i] = i;
        
                // Copy input data to array on GPU.
                if( (err=cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess ){
                        fprintf(stderr, "do_add(): error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for %zu bytes for ha->da: %s\n",
                                N*sizeof(int), cudaGetErrorString(err)
                        );
                        return 1;
                }
        
                // Launch GPU code with N threads, one per array element.
                add<<<N, 1>>>(da, db);
                if( (err=cudaGetLastError()) != cudaSuccess ){
                        fprintf(stderr, "do_add(): error, failed to launch the kernel into the device: %s\n",
                                cudaGetErrorString(err)
                        );
                        return 1;
                }
        
                // Copy output array from GPU back to CPU.
                if( (err=cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess ){
                        fprintf(stderr, "do_add(): error, call to cudaMemcpy(cudaMemcpyDeviceToHost) has failed for %zu bytes for db->ha: %s\n",
                                N*sizeof(int), cudaGetErrorString(err)
                        );
                        return 1;
                }
        
                //for (int i = 0; i<N; ++i) printf("%d\n", hb[i]); // print results
        
                // Free up the arrays on the GPU.
                if( (err=cudaFree(da)) != cudaSuccess ){
                        fprintf(stderr, "do_add(): error, call to cudaFree() has failed for da: %s\n",
                                cudaGetErrorString(err)
                        );
                        return 1;
                }
                if( (err=cudaFree(db)) != cudaSuccess ){
                        fprintf(stderr, "do_add(): error, call to cudaFree() has failed for db: %s\n",
                                cudaGetErrorString(err)
                        );
                        return 1;
                }
        
                return 0;
        }

The statement: `use Inline::CUDA => ...;` is executed at
compile-time. Often this is not desirable because you may
want to read code from file, modify code at runtime or
even auto-generate the inlined code at runtime.
In these situations [Inline](https://metacpan.org/pod/Inline) provides `bind()`.

Here is how to inline code read at runtime from a file
called `my_cruncher.cu`, whose contents are exactly the same
as the `__DATA__` section in the previous example,

          use Inline;
          use File::Slurp;

          my $data = read_file('my_cruncher.cu');
          Inline->bind(CUDA => $data);
    

Using `Inline->use(CUDA => "DATA")` seems to have a problem
when `__DATA__` section contains identifiers enclosed
in double underscores, e.g. `__global__` (this is a CUDA reserved keyword)
one workaround is to declare `#define CUDA_GLOBAL __global__`
and then replace all `__global__` with `CUDA_GLOBAL`.

Sometimes, it is more convenient to configure [Inline::CUDA](https://metacpan.org/pod/Inline%3A%3ACUDA)
not in a `use` statement (as above) but in a `require` statement.
The latter is executed during the runtime of your script as opposed
to loading the file during compile time for the former. This has
certain benefits as you can enclose it in a conditional,
eval or try/catch blocks. This is how
(thank you [Corion@PerlMonks.org](https://perlmonks.org/?node_id=11159977)):

    require Inline;
    # configuration:
    Inline->import(
      CUDA => Config =>
        ccflagsex => '...'
    );
    # compile your code:
    Inline->import(
      CUDA => $my_code
    );

# CUDA

The somewhat old news, at least since 2007, is that a Graphics Processing Unit (GPU)
has found uses beyond its traditional role in calculating and displaying graphics
to our computer monitor. This stems from the fact that a GPU is a highly parallel
computing machinery. Similar to the operating system sending data and instructions
to that GPU frame-after-frame from the time it is booted in order to display
windows, widgets, transparent menus, spinning animations, video games and
visual effects, a developer can now send data and instructions to the GPU
for doing any sort of arithmetic calculation in a highly parallel manner.
Case in point is matrix multiplication where thousands of GPU computing elements
are processing the matrices' elements in parallel. True parallelism, that is.
As opposed to the emulated or limited, by the number of cores, 2, 4, 8 for cheap desktops,
CPU's parallelism. It goes without saying that GPU processing is very powerful
and opens up to a new world of nunber-crunching possibilities without
the need for expensive super-computer capabilities.

NVIDIA's CUDA is "a parallel computing platform and programming
model that makes using a GPU for general purpose computing simple
and elegant"
(from [NVIDIA's site](https://blogs.nvidia.com/blog/2012/09/10/what-is-cuda-2/)).
In short, we use CUDA to dispatch number-crunching code
to a Graphics Processing Unit (GPU) and then get the results back.

NVIDIA's CUDA comprises of a few keywords which can be inserted in C, C++, Fortran,
etc. code. In effect, developers still write programs in their preferred language (C, C++ etc.)
and whenever they need to access the GPU they use the CUDA
extensions. For more information check
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) .

A CUDA program is, therefore, a C or C++ program with a few CUDA keywords added.
Generally, compiling such a program is done by a CUDA compiler, namely nvcc (nvidia cuda compiler)
which, simplistically put, splits the code in two parts, the CUDA part and the C part.
The C part is delegated to a C compiler, like gcc, and the CUDA part is handled by
nvcc. Finally nvcc links these components into an ordinary standalone executable.
For more information read
[CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#cuda-source)

Notice that in NVIDIA jargon, a "device" is (one of) the GPU
and "host" is the CPU and the OS.

# CAVEATS

In practice there are huge caveats which their conquering can be surprisingly easy
with some CLI magic. This is fine in Linux or even OSX but for poor M$-windows
victims, the same process can be painfully tortuous and possibly ending
to a mental breaker. As I don't belong to that category I will not be able to
help you with very specific requests regarding the so-called OS.

And on to the caveats.

### Does your GPU support CUDA?

First of all, not all GPUs support CUDA. But new NVIDIA ones usually do and
at a price of less or around 100 euros.

### Different CUDA SDK exists for different hardware

Secondly, different GPUs have different "compute capability" requiring different
versions of the CUDA SDK, which provides the nvcc and friends. For example my `GeForce GTX 650`
has a compute capability of `3.0` and that requires a SDK version of `10.2`.
That's the last SDK to support a `3.x` capability GPU. Currently, the SDK has reached
version 11.4 and supports compute capabilities of `3.5` to `8.6`. See the
[Wikipedia article on CUDA](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
for what GPUs are supported and by what CUDA SDK version.

### CUDA compiler requires specific compiler version

Thirdly and most importantly, `nvcc` has specific and strict
requirements regarding the version of the "host compiler", for example,
`gcc/g++`, `clang`, `cl.exe`. See which compilers are supported at

- [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), 
- [mac-OSX](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html),
- [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)

For example, my GPU's compute capability (`3.0`) requires CUDA SDK version `10.2`
which requires gcc version less or equal to `8`. Find out what compiler
your CUDA SDK supports in this
[ax3l's gist](https://gist.github.com/ax3l/9489132)

There is a hack to stop `nvcc` checking compiler version and using whatever
compiler it is specified by the user. Simply pass `--allow-unsupported-compiler` to `nvcc`
and hope for the best. According to
[CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html),
this flag has no effect in MacOS.

`xt/30-check-basic-with-system-compiler.t` shows how to tell `Inline::CUDA`
to use the system compiler and also tell `nvcc` to not check
compiler version. This test can fail in particular OS/versions. It seems
to have worked for my particular setting. With this option you
are at least safe from getting into trouble because of
["Perl and XS objects with mismatched compiler versions"](#perl-and-xs-objects-with-mismatched-compiler-versions).

### GPU Programming: memory transfers overheads

Additionally, general GPU programming, in practice, has quite some caveats of
its own that the potential GPU programmer must be aware of. To start with,
there are some quite large overheads associated with sending data to the
GPU and receiving it back. Because the memory generally accessible to any program
running on the CPU (e.g. the C-part of the CUDA code) is not available to the
GPU in the simple and elegant manner C programmers take for granted
when presented with a memory pointer
and read the memory space it points to. And vice versa. Memory in the C-part of the
code must be `cudaMemcpy()`'ed (the equivalent of `memcpy()` for
host-to-device and device-to-host data transfers) to the GPU. And the results calculated
in the GPU remain there until are transfered back to host using another `cudaMemcpy()` call.

Add to this the overhead of copying the value of each item of a Perl array into
a C array which `cudaMemcpy()` understands and expects and you get quite
a significant overhead and a lot of paper-pushing for finally getting the same block
of data onto the GPU. And the same applies in doing the reverse.

Here is a rough sketch of what memory transfers are required
for calling an `Inline::CUDA` function from Perl and doing GPU processing:

        my @array = (1..5); # memory allocated for Perl array
        inline_cuda_function(\@array, $result);
        ...
        // now inside a Inline::CUDA code block
        int inline_cuda_function(SV *in, SV *out){
                // allocate memory for copying Perl array (in) to C
                h_in = malloc(5*sizeof(int));
                // allocate memory for holding the results on host
                h_out = malloc(5*sizeof(int));
                // allocate memory on the GPU for this same data
                cudaMalloc((void **)&d_in, 5*sizeof(int));
                // allocate memory on the GPU for the result
                cudaMalloc((void **)&d_out, 5*sizeof(int));
                // transfer Perl data onto host's C-array
                AV *anAV = (AV *)SvRV(in);
                for(int i=0;i<5;i++){
                        SV *anSV = *av_fetch((AV *)SvRV(anAV), i, FALSE);
                        h_in[i] = SvNV(anSV);
                }
                // and now transfer host's C-array onto the GPU
                cudaMemcpy(d_in, h_in, 5*sizeof(int), cudaMemcpyHostToDevice);
                // launch the kernel and do the processing onto the GPU
                ...
                // extract results from the GPU onto host memory
                cudaMemcpy(h_out, d_in, 5*sizeof(int), cudaMemcpyDeviceToHost);
                // and now from host memory (the C array) onto Perl
                // we have been passed a scalar, we create a new arrayref
                // and place it to its RV slot
                anAV = newAV();
                av_extend(anAV, 5); // resize the Perl array to fit the result
                // sv_setrv() is a macro created by LeoNerd, see above
                // it places the new array we created onto the passed scalar (out)
                sv_setrv(SvRV(out), (SV *)av);
                for(int i=0;i<5;i++){
                        av_store(av, i, newSVnv(h_out[i]));
                }
                free(h_in); free(h_out);
                cudaFree(d_in); cudaFree(d_out);
                return 0; // success
        }

There are some benchmarks in `xt/benchmarks/*.b` which compare the
performance of a `small` (size ~10x10), `medium` (size ~100x100) and
`large` (size ~1000x1000) data scenario for
doing matrix multiplication (run them with `make benchmark`).
In my computer at least the pure-C,
CPU-hosted outperforms the GPU for the `small`, `medium` scenaria
exactly because of these overheads. But the GPU is a clear
winner for `large` data scenario.

See for example this particular benchmark: `xt/benchmarks/30-matrix-multiply.b`

### Perl and XS objects with mismatched compiler versions

Finally, there is an issue with compiling XS code, which is essentially what `Inline::CUDA` does,
with a compiler which is different to the compiler current Perl is built with. This is
the case when a special host compiler had to be installed because of
the CUDA SDK version. if that's true then you are essentially loading XS code
compiled with `gcc82` (as per the example in section ["INSTALLATION"](#installation)) with
a perl executable which was compiled with system compiler, for example `gcc11`.
If that is really an issue then it will be insurmountable and the only
solution will be to [perlbrew](https://perlbrew.pl/) a new Perl built
with the special host compiler, e.g. `gcc82`.

The manual on
[installing Perl](https://metacpan.org/dist/perl/view/INSTALL#C-compiler) states
that specifying the compiler is as simple as `sh Configure -Dcc=/usr/local/gcc82/bin/gcc82`

If you want to compile and install a new Perl using [perlbrew](https://perlbrew.pl/)
then this will do it (thank you [Fletch@Perlmonks.org](https://perlmonks.org/?node_id=11159958):

    PERLBREW_CONFIGURE_FLAGS='-d -Dcc=/usr/local/gcc82/bin/gcc' perlbrew install 5.38.2 --as 5.38.2.gcc82

The `-d` is for not being asked trivial questions about the compilation options
and use sane defaults. The `--as 5.38.2.gcc82` tells [perlbrew](https://perlbrew.pl/)
to rename the new installed perl in case there is already one with the same name.

# INSTALLATION

Installation of `Inline::CUDA` is a nightmare because it depends on external
dependencies. It needs NVIDIA's CUDA SDK (providing `nvcc` (the nvidia cuda compiler)
which requires specific host compiler versions. Which means that it is
very likely that you will also need to install in your system
an older compiler compatible with `nvcc` version. Even if your
GPU supports the latest CUDA SDK version (at `11.4` as of July 2021),
the maximum `gcc` version allowed with that is `10.21`.
Currently, `gcc` is at version `11.2` and upgrades monthly.

Installing a "private" compiler, in Linux, can be easy or hard depending
whether the package manager allows it. Mine does not. See ["how-to-install-compiler"](#how-to-install-compiler)
for instructions on how to do that on Linux and label the new compiler with
its own name so that one can have system compiler and older compiler
living in parallel and not disturbing each other.

**That said**, there is a workaround: add this to pass
the `--allow-unsupported-compiler` flag to `nvcc`.
This can be achieved via the `use Inline =` Config => ...>, as below:

        use Inline => Config =>
                nvccflags => '--allow-unsupported-compiler',
                ... # other config options
        ;
        ... # Inline::CUDA etc.

The long and proper way of installing `Inline::CUDA` is described below.

So, if all goes Merfy you will have to install
`nvcc` and an additional host compiler `gcc`. The latter is not
the most pleasant of experiences in Linux. I don't know what's the situation
with Windows. I can only imagine the horror.

Here is a rough sketch of what one should do.

- Find the NVIDIA GPU name+version you have installed on your hardware kit. For example,
`GeForce GTX 650`. This can be easy
or hard. 
    - If you already have the executable `nvidia-smi` installed or want to install it
    (e.g. in Fedora CLI do `dnf provides nvidia-smi` and make sure you have repo `rpmfusion-nonfree`
    enabled, somehow).
    - Install [nvidia::ml](https://metacpan.org/pod/nvidia%3A%3Aml) and run the script I provide with `Inline::CUDA` at `scripts/nvidia-ml-test.pl`
- With the NVIDIA GPU name+version available search this
[Wikipedia article](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
for the "compute capability" of the GPU. For example this is `3.0` for `GeForce GTX 650`.
- Use the "compute capability" of the GPU in order to find the CUDA SDK version you must install in the
same
[Wikipedia article](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) . For example, for the GPU
`GeForce GTX 650`, one should download and install CUDA SDK 10.2.
- Download, but not yet install, the specific version of the CUDA SDK from the
[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- If you are lucky, your system's C compiler will be compatible with the CUDA SDK version
you downloaded and installing the above archive will be successful. it is worth to
give it a try, i.e. try to install and see if it will complain about incompatible host
compiler version. If it doesn't then you are good to go.
- If installing the above archive yields errors about incompatible host
compiler then you must install a supported host compiler at a private path
(so as not to interfere with your actual system compiler) and provide that
path during installation (see below) of the CUDA SDK and also during
installation of `Inline::CUDA` (see below).
- Find the maximum host compiler version supported by your CUDA SDK you
downloaded. For example, CUDA SDK 10.2 in Linux is documented at
[https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-linux/](https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-linux/). It states
that the maximum gcc version is `8.2.1` for `RHEL 8.1`.
I suspect that it is the compiler's major version, e.g. `8`, that matters.
I can confirm that `gcc 8.4.0` works fine
for `Linux, Fedora 34, kernel 5.12, perl v5.32, GeForce GTX 650`.
- Once you decide on the compiler version, download it and install it to a private
path so as not to interfere with the system compiler. Note that path for later use.
-  I have instructions on how to do the above, in Linux for `gcc`.
Download specific
`gcc` version from: [ftp://ftp.fu-berlin.de/unix/languages/gcc/releases/](ftp://ftp.fu-berlin.de/unix/languages/gcc/releases/)
(other mirrors exist here [https://gcc.gnu.org/mirrors.html](https://gcc.gnu.org/mirrors.html)). Compile
the compiler and make sure you give it a `prefix` and a `suffix`. You must also
download packages [https://ftp.gnu.org/gnu/mpfr](https://ftp.gnu.org/gnu/mpfr), [https://ftp.gnu.org/gnu/mpc/](https://ftp.gnu.org/gnu/mpc/)
and [https://ftp.gnu.org/gnu/gmp/](https://ftp.gnu.org/gnu/gmp/), choosing versions compatible with the gcc version
you have already downloaded. The crucial line in the configuration stage of
compiling gcc is `configure --prefix=/usr/local/gcc82 --program-suffix=82 --enable-languages=c,c++ --disable-multilib --disable-libstdcxx-pch` .
Here is a gist from [https://stackoverflow.com/questions/58859081/how-to-install-an-older-version-of-gcc-on-fedora](https://stackoverflow.com/questions/58859081/how-to-install-an-older-version-of-gcc-on-fedora):

            tar xvf gcc-8.2.0.tar.xz 
            cd gcc-8.2.0/
            tar xvf mpfr-4.0.2.tar.xz && mv -v mpfr-4.0.2 mpfr
            tar xvf gmp-6.1.2.tar.xz && mv -v gmp-6.1.2 gmp
            tar xvf mpc-1.1.0.tar.gz && mv -v mpc-1.1.0 mpc
            cd ../
            mkdir build-gcc820
            cd build-gcc820/
            ../gcc-8.2.0/configure --prefix=/usr/local/gcc82 --program-suffix=82 --enable-languages=c,c++,fortran --disable-multilib --disable-libstdcxx-pch
            make && make install

    From now on, I will be using `/usr/local/gcc82/bin/gcc82` and `/usr/local/gcc82/bin/g++82` as my
    host compilers.

- Now you have our special compiler at `/usr/local/gcc82` under the name `/usr/local/gcc82/bin/gcc82`
and also `/usr/local/gcc82/bin/g++82`. We need to install the CUDA SDK and tell it to skip
checking host compiler compatibility (I don't think there is a way to point it to the correct compiler to use).
In Linux, this is like `sh cuda_10.2.89_440.33.01_linux.run --override`. After a successful installation
you should be able to see `/usr/local/cuda/bin/nvcc`. Optionally add this to your PATH,
`export PATH="${PATH}:/usr/local/cuda/bin"`
- In general, compiling CUDA code, for example [this one](https://gist.github.com/dpiponi/1502434),
is as simple as:

            nvcc --compiler-bindir /usr/local/gcc82/bin simple.cu && a.out

    **Notice the cuda program extension `.cu`**. Without this extension,
    in the best case `nvcc` will not know what to do with that file
    and with the `C` extension, `nvcc` will bomb.
    It is important to keep the CUDA compiler happy.
    Also note that if your CUDA SDK does not require installing an older
    version of a compiler but instead it is happy with your system compiler,
    then you can omit this: `--compiler-bindir /usr/local/gcc82/bin`

- If you did compile the simple cuda program and managed to run it, then
you are ready to install `Inline::CUDA`. If your system compiler is acceptable
by CUDA SDK, then it is as simple as running

            perl Makefile.PL
            make
            make install

    But if you need to declare a special host compiler (re: `/usr/local/gcc82/bin/gcc82`)
    because your system compiler is not accepted by CUDA SDK then you need to
    specify that to the installation process via one of the following two methods:

    - The first method is more permanent but assumes that you
    can (re-)install the module.
    During installation, specify the following environment variables,
    assuming a bash-based terminal, then this should do it:

                CC=/usr/local/gcc82/bin/gcc82 \
                CXX=/usr/local/gcc82/bin/g++82 \
                LD=/usr/local/gcc82/bin/g++82 \
                perl Makefile.PL
                make
                make install

    - The second method assumes you can edit `Inline::CUDA`'s
    configuration file located to a place like:
    `/usr/local/share/perl5/5.32/auto/share/dist/Inline-CUDA/Inline-CUDA.conf`
    (different systems will have a slightly different path),
    and modify the entries for 'cc', 'cxx' and 'ld'.
    - The third and most versatile method is to specify
    a custom configuration file
    at the `Config` section of [Inline::CUDA](https://metacpan.org/pod/Inline%3A%3ACUDA), like so:

            use Inline CUDA => Config =>
              ...
              CONFIGURATION_FILE => 'abc/mycuda.conf',
              ...
            ;

        with only those entries you want to modify. Any entry
        not specified in your custom configuration file will
        be read from the system-wide configuration.

- Whatever the host compiler was, the configuration will be saved in
a file called `Inline-CUDA.conf`. This file will be saved
in a `share-dir` relative to your current Perl installation path. As an
example mine is at `/usr/local/share/perl5/5.32/auto/share/dist/Inline-CUDA/Inline-CUDA.conf`

    This configuration file will be consulted every time you use `Inline::CUDA` and will
    know where the special host compiler resides.

- Finally, `make test` will
run a suite of test scripts and if all goes well all will succeed.
Additionally, `make benchmark` will run a matrix multiplication benchmark
which will reveal if you can indeed get any benefits using GPGPU on your
specific hardware for this specific problem. Feel free to extend benchmarks
for your use-case.
- At this stage I would urge people installing the code to run
also `make author-test` and report back errors.

# DEMO

The folder `demos/` in the base dir of the current distribution
contains self-contained `Inline::CUDA` demo(s). One of which
produces the Mandelbrot Fractal on the GPU using Cuda code
copied from [marioroy](https://perlmonks.org/?node=marioroy)'s
excellent work at [https://github.com/marioroy/mandelbrot-python](https://github.com/marioroy/mandelbrot-python),
see also PerlMonks post at [https://perlmonks.org/?node\_id=11139880](https://perlmonks.org/?node_id=11139880).
The demo is not complete, it just plugs [marioroy](https://perlmonks.org/?node=marioroy)'s
Cuda code into `Inline::CUDA`.

From the base dir of the current distribution run:

    make demo

# CAVEATS

In your CUDA code do not implement `main()`!
Place your CUDA code in your own functions which
you call from Perl. If you get segmentation faults
check the above first.

# CONTRIBUTIONS BY OTHERS

This is a module which stands on the shoulders of Giants.

Literally!

To start with, CUDA and `nvidia cuda compiler` are
two NVIDIA projects which offer general programming on the GPU
to the masses opening a new world of computational
capabilities as an alternative to the traditional CPU model.
A big thank you to NVIDIA.

Then there is Perl's [Inline](https://metacpan.org/pod/Inline) module created by [Ingy döt Net](https://metacpan.org/author/INGY).
This module makes it easy to inline a lot of computer languages and call
them within a Perl script, passing Perl data structures and obtaining
results back.

This module is the key to opening many doors for Perl scripts.

A big thank you to [Ingy döt Net](https://metacpan.org/author/INGY).

Then there is Perl's [Inline::C](https://metacpan.org/pod/Inline%3A%3AC) module created/co-created/maintained
by [Ingy döt Net](https://metacpan.org/author/INGY),
[Sisyphus](https://metacpan.org/author/SISYPHUS) and [Tina Müller](https://metacpan.org/author/TINITA).

The current `Inline::CUDA` module relies heavily on [Inline::C](https://metacpan.org/pod/Inline%3A%3AC). Because
the underlying CUDA language is C, I decided that instead of copying what
[Inline::C](https://metacpan.org/pod/Inline%3A%3AC) does and modifying the section where the Makefile is written,
I decided to inject all [Inline::C](https://metacpan.org/pod/Inline%3A%3AC)'s subs into `Inline::CUDA` except
some sections which require special treatment, like when writing the Makefile
and also allowing some special `Config` keywords. The sub injection happens
every time the module is called, and that definetely adds a tiny overhead
which, in my opinion, is compensated by the huge advantage of not
copy-pasting code from [Inline::C](https://metacpan.org/pod/Inline%3A%3AC) into  `Inline::CUDA` and then
incorporating my changes every time [Inline::C](https://metacpan.org/pod/Inline%3A%3AC) updates.
A big thank you to [Ingy döt Net](https://metacpan.org/author/INGY) (again!),
[Sisyphus](https://metacpan.org/author/SISYPHUS) and [Tina Müller](https://metacpan.org/author/TINITA).

For writing test cases and benchmarks I had to descend into C and become
acquainted with [perlguts](https://metacpan.org/pod/perlguts),
e.g. what is an [SV](https://perldoc.perl.org/perlguts#Working-with-SVs).
In this process I had to ask for the wisdom of [PerlMonks.org](https://metacpan.org/pod/PerlMonks.org) and
[#perl](https://web.libera.chat/#perl). A particular question was
how to pass in a C function an arrayref, a scalar or a scalarref,
store the results of the computation in there, in a call-by-reference manner.
Fortunately LeoNerd at [#perl](https://kiwiirc.com/nextclient/#irc://irc.perl.org/#perl)
created the following `sv_setrv()` macro which saved the day. Big thank you LeoNerd.

        /************************************************************/
        /* MONKEYPATCH by LeoNerd to set an arrayref into a scalarref
           As posted on https://kiwiirc.com/nextclient/#irc://irc.perl.org/#perl
           at 10:50 23/07/2021
           A BIG THANK YOU LeoNerd
        */
        #define HAVE_PERL_VERSION(R, V, S) \
            (PERL_REVISION > (R) || (PERL_REVISION == (R) && (PERL_VERSION > (V) || (PERL_VERSION == (V) && (PERL_SUBVERSION >= (S))))))

        #define sv_setrv(s, r)  S_sv_setrv(aTHX_ s, r)
        static void S_sv_setrv(pTHX_ SV *sv, SV *rv)
        {
          sv_setiv(sv, (IV)rv);
        #if !HAVE_PERL_VERSION(5, 24, 0)
          SvIOK_off(sv);
        #endif
          SvROK_on(sv);
        }

I copied numerical recipes (as C code, Cuda kernels, etc.) from the repository of
[Zhengchun Liu](https://github.com/lzhengchun) this code resides in 'C/inlinecuda'
of the current distribution and offers shortcuts to GPU-based matrix multiplication,
for example.

The idea of this project came to me when [kcott](https://www.perlmonks.org/?node=kcott)
asked whether there are [https://www.perlmonks.org/?node\_id=11134476](https://www.perlmonks.org/?node_id=11134476)
which I responded with the [preliminary idea](https://www.perlmonks.org/?node_id=11134582) for what is
now `Inline::CUDA`. A big thank you to [kcott](https://www.perlmonks.org/?node=kcott).

I got helpful comments, advice and the odd smiley from
`LeoNerd`, `mst`, `Bojte`, `shlomif` at [#perl](https://kiwiirc.com/nextclient/#irc://irc.perl.org/#perl),
thank you.

I got helpful comments and advice in this [PerlMonks.org post](https://perlmonks.org/?node_id=11135324)
from [syphilis](https://perlmonks.org/?node=syphilis) and [perlfan](https://perlmonks.org/?node=perlfan),
although the problem was cracked by LeoNerd [#perl](https://kiwiirc.com/nextclient/#irc://irc.perl.org/#perl).

I also got helpful comments and advice from [Ed J](https://metacpan.org/author/ETJ) when I filed a bug over
at [ExtUtils::MakeMaker](https://metacpan.org/pod/ExtUtils%3A%3AMakeMaker) (see [https://rt.cpan.org/Ticket/Display.html?id=138022](https://rt.cpan.org/Ticket/Display.html?id=138022) and
[https://rt.cpan.org/Ticket/Display.html?id=137912](https://rt.cpan.org/Ticket/Display.html?id=137912)).

# AUTHOR

Andreas Hadjiprocopis,
`<bliako at cpan dot org>`,
`<andreashad2 at gmail dot com>`,
[https://perlmonks.org/?node=bliako](https://perlmonks.org/?node=bliako)

# DEDICATIONS

!Almaz!

# BUGS

Please report any bugs or feature requests to `bug-inline-cuda at rt.cpan.org`, or through
the web interface at [https://rt.cpan.org/NoAuth/ReportBug.html?Queue=Inline-CUDA](https://rt.cpan.org/NoAuth/ReportBug.html?Queue=Inline-CUDA).  I will be notified, and then you'll
automatically be notified of progress on your bug as I make changes.

NOTE: this project is not yet on CPAN so report bugs by email to the author. I am not
very comfortable with github so cloning and merging and pushing and pulling are beyond me.

# SUPPORT

You can find documentation for this module with the perldoc command.

    perldoc Inline::CUDA

You can also look for information at:

- RT: CPAN's request tracker (report bugs here)

    [https://rt.cpan.org/NoAuth/Bugs.html?Dist=Inline-CUDA](https://rt.cpan.org/NoAuth/Bugs.html?Dist=Inline-CUDA)

- PerlMonks.org : a great forum to find Perl support and wisdom

    The main side is this [https://perlmonks.org](https://perlmonks.org) where you can post
    questions. The author's page is this [https://perlmonks.org/?node=bliako](https://perlmonks.org/?node=bliako)

- CPAN Ratings

    [https://cpanratings.perl.org/d/Inline-CUDA](https://cpanratings.perl.org/d/Inline-CUDA)

- Search CPAN

    [https://metacpan.org/release/Inline-CUDA](https://metacpan.org/release/Inline-CUDA)

# ACKNOWLEDGEMENTS

This module stands on the shoulders of giants, namely
the authors of [Inline](https://metacpan.org/pod/Inline) and [Inline::C](https://metacpan.org/pod/Inline%3A%3AC). I wish
to thank them here and pass most credit to them. I will keep 1%.

A big thank you to NVIDIA for providing tools
and support for doing numerical programming on their GPU.

All mentioned above provided keys to many doors, all free and
open source. Thank you!

# LICENSE AND COPYRIGHT

This software is Copyright (c) 2021 by Andreas Hadjiprocopis.

This is free software, licensed under:

    The Artistic License 2.0 (GPL Compatible)
