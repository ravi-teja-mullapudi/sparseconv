falcon=$(pwd)/third_party/FALCON/build
memkind=$(pwd)/third_party/memkind/.libs

LD_LIBRARY_PATH=$falcon:$memkind:$LD_LIBRARY_PATH ./bench $1
