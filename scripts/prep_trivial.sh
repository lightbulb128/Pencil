cd pencil-prep
python3 train_priv_server.py -p mnist_aby3 -n 0 -C 8 -o SGD -om 0.8 -lr 0.01 > /dev/null &
sleep 5
python3 train_priv_client.py -s None -e 1 > logs/trivial.log &
cd ..
