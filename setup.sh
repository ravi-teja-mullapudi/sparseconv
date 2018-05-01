# Setup Memkind
echo "Building MemKind in third_party/memkind"

cd third_party/memkind
./build.sh

exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Error building MemKind"
    exit $exit_status
fi

# Setup FALCON
echo "Building FALCON in third_parth/FALCON"

cd ../FALCON
./install.sh

exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Error building FALCON"
    exit $exit_status
fi
