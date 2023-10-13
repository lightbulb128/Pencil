cd seal-cuda
git submodule update --init --recursive

bash install_seal.sh
bash makepackage.sh

cd ..
