git submodule update --init --recursive

bash scripts/build_opencheetah.sh
bash scripts/build_seal_cuda.sh
bash scripts/build_ezpc.sh

cd pencil-fullhe
mkdir -p logs
bash gather_tools.sh
cd ..

cd pencil-prep
bash gather_tools.sh
mkdir -p logs
mkdir -p preprocess
cd ..