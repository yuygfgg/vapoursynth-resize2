project('resize2', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'c_std=c89', 'cpp_std=c++17', 'b_lto=true'],
  meson_version : '>=1.0.0',
  version : '8'
)

OPT_CFLAGS = ['-march=native', '-mllvm' ,'-polly', '-mllvm', '-polly-vectorizer=stripmine', '-mllvm', '-polly-tiling']

add_project_arguments(OPT_CFLAGS, language: 'c')
add_project_arguments(OPT_CFLAGS, language: 'cpp')

sources = [
    'resize.cpp'
]

vapoursynth_dep = dependency('vapoursynth').partial_dependency(compile_args : true, includes : true)

deps = [
    vapoursynth_dep,
    dependency('zimg_patched'),
]

shared_module('resize2', sources,
    dependencies: deps,
    link_args: ['-static'],
    install : true,
    install_dir : join_paths(vapoursynth_dep.get_variable(pkgconfig: 'libdir'), 'vapoursynth'),
    gnu_symbol_visibility : 'hidden',
)