## How to run

* Place the dependency modules in `tools/`
    * Pencil-edition of the OpenCheetah/SCI library. `cheetah_provider_alice/bob.cpython`. It could be obtained from [Github repo](https://github.com/lightbulb128/OpenCheetah).
    * seal-cuda HE library. `pytroy.cpython`. It could be obtained from [Github repo](https://github.com/lightbulb128/troy).
* Run
    * `train_priv_server.py` and `train_priv_client.py` run as two processes. You could use `-p` for different model/dataset setting in `server`, as defined in `model.py` (`get_model` method). Noise and gradient bound could be set by `-n` and `-C` respectively.
    * When the scripts are run, the resulting shares of preprocessing will be stored in `preprocess/` folder. Re-running will load the files, instead of regenerating the shares. This behavior could be turned off in `config.py - PREPARE_FORCE_REGENERATE`.
    * `train_priv_plain.py` is a simulator run on fixed-point representations, but not encrypted.

## Main components
* `communication_*.py` for communication and non-linear functions in SCI library.
* `crypto_gpu_cheetah.py` mainly for linear evaluation using seal-cuda.
* `client_modules/server_modules.py` for various layers implementation.
* `optimizer.py` for SGD-Momentum optimizer.
* `trusted_preparation.py` generate the proprocessing shares in plaintext.
* `models.py` for different models.