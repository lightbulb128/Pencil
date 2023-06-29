CHEETAH_LIBRARY_PATH=../OpenCheetah/build/lib
SCI_LIBRARY_PATH=../EzPC/SCI/build/lib
CUDA_CKKS_NEW_PATH=../seal-cuda

mkdir -p tools

cp ${CHEETAH_LIBRARY_PATH}/cheetah_provider*.so tools/

cp ${SCI_LIBRARY_PATH}/sci_provider*.so tools/

cp ${CUDA_CKKS_NEW_PATH}/binder/pytroy.cpython-38-x86_64-linux-gnu.so tools/
