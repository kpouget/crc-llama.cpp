cmake -S . -B ../build.remoting-frontend \
      -DGGML_REMOTINGFRONTEND=ON \
      -DGGML_CPU_ARM_ARCH=native \
      -DGGML_NATIVE=OFF \
      -DCMAKE_BUILD_TYPE=Debug \
      "$@"
