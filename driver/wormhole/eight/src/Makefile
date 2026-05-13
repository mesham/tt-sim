CXX=clang++-17

INCLUDES=-I${TT_METAL_HOME}/tt_metal/api -I${TT_METAL_HOME}/tt_metal/api/tt-metalium -I${TT_METAL_HOME}/build_Release/include -I${TT_METAL_HOME}/build_Release/include/metalium-thirdparty -I${TT_METAL_HOME}/build_Release/include/umd/device/device/api -I${TT_METAL_HOME}/tt_metal/hostdevcommon/api/ -I${TT_METAL_HOME}/tt_metal/third_party/tracy/public -I${TT_METAL_HOME}/build_Release/include/fmt

CFLAGS=${INCLUDES} -g -Wno-int-to-pointer-cast -mavx2 -fPIC -DFMT_HEADER_ONLY -DSPDLOG_FMT_EXTERNAL -fvisibility-inlines-hidden -fno-lto -DARCH_WORMHOLE -DDISABLE_ISSUE_3487_FIX -Werror -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter -Wsometimes-uninitialized -Wno-c++11-narrowing -Wno-c++23-extensions -Wno-error=local-type-template-args -Wno-delete-non-abstract-non-virtual-dtor -Wno-c99-designator -Wno-shift-op-parentheses -Wno-non-c-typedef-for-linkage -Wno-deprecated-this-capture -Wno-deprecated-volatile -Wno-deprecated-builtins -Wno-deprecated-declarations -std=c++20

LINKER=clang++-17
LFLAGS=-rdynamic -L${TT_METAL_HOME}/build_Release/lib -ltt_metal -ldl -lstdc++fs -pthread -lm -ldevice

all:
		${CXX} ${CFLAGS} -c eight.cpp
		${LINKER} eight.o -o eight ${LFLAGS}

clean:
		rm -rf eight eight.o generated/
