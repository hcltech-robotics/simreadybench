# Dependency locks

`runtime.txt` contains the hash-locked Linux amd64 Python 3.12 dependencies for Harbor, validation and Daytona. `build.txt` contains the package build tools. Their `.in` files declare the direct dependencies.

```sh
python -m pip install --require-hashes -r locks/runtime.txt
```

`storm-linux-64.json` pins the OpenUSD imaging environment used by the CPU renderer. The actor has its own Python and npm locks in `containers/actor/`.
