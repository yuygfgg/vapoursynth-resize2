```bash
git clone --recurse-submodules --remote-submodules https://github.com/vapoursynth/vapoursynth.git
git clone --recurse-submodules --remote-submodules https://bitbucket.org/the-sekrit-twc/zimg.git
meson setup build
meson compile -C build
```