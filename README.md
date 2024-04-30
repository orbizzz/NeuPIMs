# NeuPIMs Simulator

### Python Package

- torch >= 1.10.1
- conan == 1.57.0
- onnxruntime >= 1.10.0

### Package

- cmake >= 3.22.1 (You need to build manually)
- gcc == 8.3

---

# Getting Started

## method 1 (Docker Image)

```
$ git clone https://github.com/casys-kaist/NeuPIMs.git
$ cd ai-framwork-sim
$ docker build . -t neupims
```

build docker image

```
$ docker run -it neupims
(docker) cd ai-framwork-sim
(docker) ./brun.sh
```

run docker image

## method 2 (Mannual)

### Instrallation

```
$ git clone https://github.com/casys-kaist/NeuPIMs.git
$ cd ai-framwork-sim
$ git submodule update --recursive --init
```

### Build

```
$ mkdir build && cd build
$ conan install .. --build missing
$ cmake ..
$ make -j
```

### Run Simulator

```
$ cd ..
$ ./brun.sh
```

### Baselines

1. NPU-only: Codes on `npu-only` branch, all operations in LLM batched inference are executed on NPU.
2. NPU+PIM: Codes on `npu+pim` branch, attention GEMV operations on PIM. PIM is single row buffered PIM on this baseline.
3. NeuPIMs: Codes on `main` branch, we use dual row buffer PIM and sub-batch interleaving technique. (Sub-batch interleaving is enabled only when batch size>=256)
