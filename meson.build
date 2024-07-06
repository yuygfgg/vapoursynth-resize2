project('resize2', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'c_std=c89', 'cpp_std=c++17'],
  meson_version : '>=0.51.0',
  version : '8'
)

cxx = meson.get_compiler('cpp')

compile_args = [
    '-Wno-unused-parameter', '-Wno-missing-field-initializers', '-Wno-format', '-Wno-endif-labels',
    '-lstdc++', '-fvisibility=hidden', '-DGRAPHENGINE_IMPL_NAMESPACE=zimg'
]

host_cpu_family = host_machine.cpu_family()
host_system = host_machine.system()

opt_ARMSIMD = false
opt_X86SIMD = false
opt_X86SIMD_AVX512 = false

link_libs = []

if host_cpu_family in ['arm', 'aarch64']
    opt_ARMSIMD = get_option('ENABLE_ARMSIMD')
    compile_args += ['-DZIMG_ARM']
elif host_cpu_family.startswith('x86')
    opt_X86SIMD = get_option('ENABLE_X86SIMD')

    
    code = '''
#include <immintrin.h>
int main()
{
	volatile __m512 x = _mm512_setzero_ps();
	x = _mm512_add_ps(x, x);
#if defined(__GNUC__) && defined(__x86_64__)
	asm volatile ("" : : : "zmm16", "zmm17", "zmm18", "zmm19");
#endif
	return 0;
}
    '''
    result = cxx.compiles(code, args : ['-mavx512f', '-mavx512pf', '-mavx512er', '-mavx512cd', '-mavx512vl', '-mavx512bw', '-mavx512dq', '-mavx512ifma', '-mavx512vbmi', '-mavx512vbmi2', '-mavx512bitalg', '-mavx512vpopcntdq', '-mavx512vnni'], name : 'avx512 checking')

    opt_X86SIMD_AVX512 = result and get_option('ENABLE_X86SIMD_AVX512')

    if opt_X86SIMD_AVX512
        compile_args += ['-DZIMG_X86_AVX512']
    endif

    if opt_X86SIMD or opt_X86SIMD_AVX512
        compile_args += ['-DZIMG_X86']
    endif

    compile_args += ['-mfpmath=sse', '-msse2']
endif



incl_dirs = [
    include_directories('zimg/graphengine/include'),
    include_directories('zimg/src/zimg'),
    include_directories('zimg/src'),
    include_directories('vapoursynth/include'),
]

sources = [
    'resize.cpp',

    'zimg/graphengine/graphengine/cpuinfo.h',
    'zimg/graphengine/graphengine/cpuinfo.cpp',
    'zimg/graphengine/graphengine/graph.cpp',
    'zimg/graphengine/graphengine/node.cpp',
    'zimg/graphengine/graphengine/node.h',
    'zimg/graphengine/graphengine/state.h',
    'zimg/graphengine/graphengine/x86/cpuinfo_x86.cpp',
    'zimg/graphengine/graphengine/x86/cpuinfo_x86.h',
    'zimg/graphengine/include/graphengine/filter.h',
    'zimg/graphengine/include/graphengine/graph.h',
    'zimg/graphengine/include/graphengine/namespace.h',
    'zimg/graphengine/include/graphengine/types.h',

    'zimg/src/zimg/api/zimg.cpp',
    'zimg/src/zimg/api/zimg.h',
    'zimg/src/zimg/api/zimg++.hpp',

    'zimg/src/zimg/colorspace/colorspace.cpp',
    'zimg/src/zimg/colorspace/colorspace.h',
    'zimg/src/zimg/colorspace/colorspace_graph.cpp',
    'zimg/src/zimg/colorspace/colorspace_graph.h',
    'zimg/src/zimg/colorspace/colorspace_param.cpp',
    'zimg/src/zimg/colorspace/colorspace_param.h',

    'zimg/src/zimg/colorspace/gamma.cpp',
    'zimg/src/zimg/colorspace/gamma.h',
    'zimg/src/zimg/colorspace/matrix3.cpp',
    'zimg/src/zimg/colorspace/matrix3.h',
    'zimg/src/zimg/colorspace/operation.cpp',
    'zimg/src/zimg/colorspace/operation.h',
    'zimg/src/zimg/colorspace/operation_impl.cpp',
    'zimg/src/zimg/colorspace/operation_impl.h',

    'zimg/src/zimg/common/align.h',
    'zimg/src/zimg/common/alloc.h',
    'zimg/src/zimg/common/builder.h',
    'zimg/src/zimg/common/ccdep.h',
    'zimg/src/zimg/common/checked_int.h',
    'zimg/src/zimg/common/cpuinfo.cpp',
    'zimg/src/zimg/common/cpuinfo.h',
    'zimg/src/zimg/common/except.h',
    'zimg/src/zimg/common/libm_wrapper.h',
    'zimg/src/zimg/common/libm_wrapper.cpp',
    'zimg/src/zimg/common/make_array.h',
    'zimg/src/zimg/common/matrix.cpp',
    'zimg/src/zimg/common/matrix.h',
    'zimg/src/zimg/common/pixel.h',
    'zimg/src/zimg/common/static_map.h',
    'zimg/src/zimg/common/unroll.h',
    'zimg/src/zimg/common/zassert.h',

    'zimg/src/zimg/depth/blue.cpp',
    'zimg/src/zimg/depth/blue.h',
    'zimg/src/zimg/depth/depth_convert.cpp',
    'zimg/src/zimg/depth/depth_convert.h',
    'zimg/src/zimg/depth/depth.cpp',
    'zimg/src/zimg/depth/depth.h',
    'zimg/src/zimg/depth/dither.cpp',
    'zimg/src/zimg/depth/dither.h',
    'zimg/src/zimg/depth/quantize.h',
    'zimg/src/zimg/depth/quantize.cpp',

    'zimg/src/zimg/graph/filter_base.cpp',
    'zimg/src/zimg/graph/filter_base.h',
    'zimg/src/zimg/graph/filtergraph.cpp',
    'zimg/src/zimg/graph/filtergraph.h',
    'zimg/src/zimg/graph/graphbuilder.cpp',
    'zimg/src/zimg/graph/graphbuilder.h',
    'zimg/src/zimg/graph/graphengine_except.cpp',
    'zimg/src/zimg/graph/graphengine_except.h',
    'zimg/src/zimg/graph/simple_filters.cpp',
    'zimg/src/zimg/graph/simple_filters.h',

    'zimg/src/zimg/resize/filter.cpp',
    'zimg/src/zimg/resize/filter.h',
    'zimg/src/zimg/resize/resize.cpp',
    'zimg/src/zimg/resize/resize.h',
    'zimg/src/zimg/resize/resize_impl.cpp',
    'zimg/src/zimg/resize/resize_impl.h',

    'zimg/src/zimg/unresize/bilinear.cpp',
    'zimg/src/zimg/unresize/bilinear.h',
    'zimg/src/zimg/unresize/unresize.cpp',
    'zimg/src/zimg/unresize/unresize.h',
    'zimg/src/zimg/unresize/unresize_impl.cpp',
    'zimg/src/zimg/unresize/unresize_impl.h'
]

if opt_ARMSIMD
    sources += [
        'zimg/src/zimg/colorspace/arm/operation_impl_arm.cpp',
        'zimg/src/zimg/colorspace/arm/operation_impl_arm.h',
        'zimg/src/zimg/common/arm/cpuinfo_arm.cpp',
        'zimg/src/zimg/common/arm/cpuinfo_arm.h',
        'zimg/src/zimg/common/arm/neon_util.cpp',
        'zimg/src/zimg/common/arm/neon_util.h',
        'zimg/src/zimg/depth/arm/depth_convert_arm.cpp',
        'zimg/src/zimg/depth/arm/depth_convert_arm.h',
        'zimg/src/zimg/depth/arm/dither_arm.cpp',
        'zimg/src/zimg/depth/arm/dither_arm.h',
        'zimg/src/zimg/depth/arm/f16c_arm.h',
        'zimg/src/zimg/resize/arm/resize_impl_arm.cpp',
        'zimg/src/zimg/resize/arm/resize_impl_arm.h'
    ]

    NEON_CFLAGS = ['-march=armv7-a', '-mfpu=neon-vfpv4']
    if cxx.has_argument('-mfp16-format=ieee')
        NEON_CFLAGS += ['-mfp16-format=ieee']
    endif

    link_libs += static_library(
        'neon',
        [
            'zimg/src/zimg/colorspace/arm/operation_impl_neon.cpp',
            'zimg/src/zimg/depth/arm/depth_convert_neon.cpp',
            'zimg/src/zimg/depth/arm/dither_neon.cpp',
            'zimg/src/zimg/depth/arm/f16c_neon.cpp',
            'zimg/src/zimg/resize/arm/resize_impl_neon.cpp'
        ],
        c_args: compile_args + NEON_CFLAGS,
        cpp_args: compile_args + NEON_CFLAGS,
        gnu_symbol_visibility: 'hidden',
        include_directories: incl_dirs
    )
endif

if opt_X86SIMD
    sources += [
        'zimg/src/zimg/colorspace/x86/operation_impl_x86.cpp',
        'zimg/src/zimg/colorspace/x86/operation_impl_x86.h',

        'zimg/src/zimg/common/x86/avx_util.h',
        'zimg/src/zimg/common/x86/avx2_util.h',
        'zimg/src/zimg/common/x86/avx512_util.h',
        'zimg/src/zimg/common/x86/cpuinfo_x86.cpp',
        'zimg/src/zimg/common/x86/cpuinfo_x86.h',
        'zimg/src/zimg/common/x86/sse_util.h',
        'zimg/src/zimg/common/x86/sse2_util.h',
        'zimg/src/zimg/common/x86/x86util.cpp',
        'zimg/src/zimg/common/x86/x86util.h',

        'zimg/src/zimg/depth/x86/depth_convert_x86.cpp',
        'zimg/src/zimg/depth/x86/depth_convert_x86.h',
        'zimg/src/zimg/depth/x86/dither_x86.cpp',
        'zimg/src/zimg/depth/x86/dither_x86.h',
        'zimg/src/zimg/depth/x86/f16c_x86.h',

        'zimg/src/zimg/resize/x86/resize_impl_x86.cpp',
        'zimg/src/zimg/resize/x86/resize_impl_x86.h',
        'zimg/src/zimg/unresize/x86/unresize_impl_x86.cpp',
        'zimg/src/zimg/unresize/x86/unresize_impl_x86.h'
    ]

    link_libs += static_library(
        'sse',
        [
            'zimg/src/zimg/colorspace/x86/operation_impl_sse.cpp',
            'zimg/src/zimg/resize/x86/resize_impl_sse.cpp',
            'zimg/src/zimg/unresize/x86/unresize_impl_sse.cpp',
        ],
        c_args: compile_args + ['-msse'],
        cpp_args: compile_args + ['-msse'],
        gnu_symbol_visibility: 'hidden',
        include_directories: incl_dirs
    )

    link_libs += static_library(
        'sse2',
        [
            'zimg/src/zimg/colorspace/x86/operation_impl_sse2.cpp',
            'zimg/src/zimg/depth/x86/depth_convert_sse2.cpp',
            'zimg/src/zimg/depth/x86/dither_sse2.cpp',
            'zimg/src/zimg/depth/x86/error_diffusion_sse2.cpp',
            'zimg/src/zimg/depth/x86/f16c_sse2.cpp',
            'zimg/src/zimg/resize/x86/resize_impl_sse2.cpp',
        ],
        c_args: compile_args + ['-msse2'],
        cpp_args: compile_args + ['-msse2'],
        gnu_symbol_visibility: 'hidden',
        include_directories: incl_dirs
    )

    SNB_CFLAGS = []
    if cxx.has_argument('-mtune=sandybridge')
        SNB_CFLAGS += ['-mtune=sandybridge']
    endif

    link_libs += static_library(
        'avx',
        [
            'zimg/src/zimg/colorspace/x86/operation_impl_avx.cpp',
            'zimg/src/zimg/resize/x86/resize_impl_avx.cpp',
        ],
        c_args: compile_args + ['-mavx'] + SNB_CFLAGS,
        cpp_args: compile_args + ['-mavx'] + SNB_CFLAGS,
        gnu_symbol_visibility: 'hidden',
        include_directories: incl_dirs
    )

    IVB_CFLAGS = []
    if cxx.has_argument('-mtune=ivybridge')
        IVB_CFLAGS += ['-mtune=ivybridge']
    endif

    link_libs += static_library(
        'f16c',
        [
            'zimg/src/zimg/depth/x86/f16c_ivb.cpp',
        ],
        c_args: compile_args + ['-mavx', '-mf16c'] + IVB_CFLAGS,
        cpp_args: compile_args + ['-mavx', '-mf16c'] + IVB_CFLAGS,
        gnu_symbol_visibility: 'hidden',
        include_directories: incl_dirs
    )

    HSW_CFLAGS = []
    if cxx.has_argument('-mtune=haswell')
        HSW_CFLAGS += ['-mtune=haswell']
    endif

    link_libs += static_library(
        'avx2',
        [
            'zimg/src/zimg/colorspace/x86/operation_impl_avx2.cpp',
            'zimg/src/zimg/depth/x86/depth_convert_avx2.cpp',
            'zimg/src/zimg/depth/x86/dither_avx2.cpp',
            'zimg/src/zimg/depth/x86/error_diffusion_avx2.cpp',
            'zimg/src/zimg/resize/x86/resize_impl_avx2.cpp'
        ],
        c_args: compile_args + ['-mavx2', '-mf16c', '-mfma'] + HSW_CFLAGS,
        cpp_args: compile_args + ['-mavx2', '-mf16c', '-mfma'] + HSW_CFLAGS,
        gnu_symbol_visibility: 'hidden',
        include_directories: incl_dirs
    )
endif # X86SIMD

if opt_X86SIMD_AVX512
    avx512_args = ['-mavx512f', '-mavx512cd', '-mavx512vl', '-mavx512bw', '-mavx512dq']
    avx512vnni_args = ['-mavx512f', '-mavx512cd', '-mavx512vl', '-mavx512bw', '-mavx512dq', '-mavx512vnni']

    if cxx.has_argument('-mtune=skylake-avx512')
        avx512_args += ['-mtune=skylake-avx512']
    endif

    if cxx.has_argument('-mtune=cascadelake')
        avx512vnni_args += ['-mtune=cascadelake']
    endif

    link_libs += static_library(
        'avx512',
        [
            'zimg/src/zimg/colorspace/x86/gamma_constants_avx512.cpp',
            'zimg/src/zimg/colorspace/x86/gamma_constants_avx512.h',
            'zimg/src/zimg/colorspace/x86/operation_impl_avx512.cpp',
            'zimg/src/zimg/depth/x86/depth_convert_avx512.cpp',
            'zimg/src/zimg/depth/x86/dither_avx512.cpp',
            'zimg/src/zimg/resize/x86/resize_impl_avx512.cpp',
            'zimg/src/zimg/resize/x86/resize_impl_avx512_common.h',
        ],
        c_args: compile_args + avx512_args,
        cpp_args: compile_args + avx512_args,
        gnu_symbol_visibility: 'hidden',
        include_directories: incl_dirs
    )

    link_libs += static_library(
        'avx512vnni',
        [
            'zimg/src/zimg/resize/x86/resize_impl_avx512_vnni.cpp',
        ],
        c_args: compile_args + avx512vnni_args,
        cpp_args: compile_args + avx512vnni_args,
        gnu_symbol_visibility: 'hidden',
        include_directories: incl_dirs
    )
endif # X86SIMD_AVX512

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args : true, includes : true)

deps = [
    vapoursynth_dep
]

if host_machine.system() == 'windows'
    deps += [meson.get_compiler('cpp').find_library('winpthread', static: true)]
else
    deps += [dependency('threads')]
endif

shared_module('resize2', sources,
    c_args: compile_args,
    cpp_args: compile_args,
    dependencies : deps,
    link_args: ['-static'],
    link_with: link_libs,
    install : true,
    install_dir : join_paths(vapoursynth_dep.get_variable(pkgconfig: 'libdir'), 'vapoursynth'),
    include_directories: incl_dirs,
    gnu_symbol_visibility : 'hidden'
)