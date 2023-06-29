## How to run

* Place the dependency modules in `tools/`
    * Pencil-edition of the OpenCheetah/SCI library. `cheetah_provider_alice/bob.cpython`. It could be obtained from [Gitlab repo](https://gitlab.com/lightbulb1281/opencheetah-Penciledition).
    * seal-cuda HE library. `pytroy.cpython`. It could be obtained from [Gitlab repo](https://gitlab.com/lightbulb1281/seal-cuda).
* Run
    * `train_priv_server.py` and `train_priv_client.py` run as two processes. You could use `-p` for different model/dataset setting in `server`, as defined in `model.py` (`get_model` method). Noise and gradient bound could be set by `-n` and `-C` respectively.
    * `train_priv_plain.py` is a simulator run on fixed-point representations, but not encrypted.

## Main components
* `communication_*.py` for communication and non-linear functions in SCI library.
* `crypto_gpu_cheetah.py` mainly for linear evaluation using seal-cuda.
* `client_modules/server_modules.py` for various layers implementation.
* `optimizer.py` for SGD-Momentum optimizer.
* `models.py` for different models.