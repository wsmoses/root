# Enzyme Automatic Differentiation Backend for ROOT: Architecture and Performance

## 1. Overview

This branch integrates Enzyme (`v0.0.297`, built against ROOT's bundled LLVM/Clang 22) as the default Automatic Differentiation (AD) backend across ROOT (`-Denzyme=ON -Dclad=OFF` by default; passing `-Dclad=ON` switches back to Clad).

All AD-enabled subsystems in ROOT are supported:
- **Cling Incremental JIT (`interpreter/cling/lib/Interpreter/BackendPasses.cpp`)**:
  - Cross-transaction LLVM IR archiving (`EnzymePreArchivePass`) and on-demand callee body materialization (`EnzymeMaterializePass`) so functions defined in earlier Cling REPL transactions can be differentiated in later transactions without re-parsing source.
  - Automatic `-O2` module promotion whenever a Cling module invokes `__enzyme_autodiff` or `__enzyme_fwddiff`, while keeping ordinary non-AD Cling prompt lines at fast `-O0`.
  - Function-pointer and global/cross-transaction evaluator specialization (`EnzymeSpecializeThunkPass`) so top-level interactive calls such as `auto df = clad::gradient(f); df.execute(...);` specialize into direct calls before `EnzymeNewPM` executes.
  - Native Fortran OpenBLAS differentiation (`BLAS::sgemm_`) with pointer-width custom allocation (`clingEnzymeAllocator`) and synthesized `slascl_`/`dlascl_` scaling bodies (`materializeDifferentiableLascl`) enabling both first-order reverse-mode gradients and second-order forward-over-reverse Hessians over BLAS-backed models without external LAPACK symbol dependencies.
  - Differentiable LLVM IR materialization for `tgamma`, `lgamma`, and `digamma`, plus scalar `errno` TBAA cleanup prior to `EnzymeNewPM`.
- **MathCore (`math/mathcore/inc/Math/EnzymeDerivator.h` and `CladDerivator.h`)**:
  - Native `#include <enzyme/enzyme>` support in Cling with zero extra `-I` or `-fplugin` flags.
  - `ROOT::Math::Enzyme::{differentiate, gradient, hessian, array_ref}` with drop-in `clad::` namespace forwarding when built with `-Denzyme=ON -Dclad=OFF`.
  - Arbitrary-order Taylor forward-mode differentiation (`Differentiator<Order>`), reverse-mode gradients (`GradientEvaluator`), and vector-mode batched forward-over-reverse Hessians (`HessianEvaluator` using `enzyme_width` batches of `32/16/8/4/1`).
  - Differentiable bodies for over 40 `TMath::*` and `ROOT::Math::*` special functions, PDFs, and CDFs.
- **`TFormula` (`hist/hist/src/TFormula.cxx`)**:
  - Direct `__enzyme_autodiff` formula gradients and batched forward-over-reverse Hessians.
- **`RooFit`, `Minuit2`, and `HistFactory` (`roofit/codegen`, `roofit/roofitcore`, `roofit/roofit`)**:
  - `RooFit::EvalBackend::Codegen()` NLL reverse-mode gradients and batched width-32 forward-over-reverse Hessians (`RooEvaluatorWrapper::generateHessian()`), integrated end-to-end with `Minuit2` (`migrad` and `hesse`) and `RooONNXFunc`.
- **`TMVA::SOFIE` (`tmva/sofie/src/RModel.cxx`)**:
  - Automatic `Session::CreateZeroShadow() const` generation for ONNX inference sessions and hardware-accelerated OpenBLAS `sgemm_` differentiation.

---

## 2. Test Suite Verification and Stability

All **21 AD CTest suites** plus `coreclingutilstestUnit` pass (`100%`, `0` failures, `0` Enzyme diagnostic warnings):
- `gtest-hist-hist-TFormulaGradientTests` and `gtest-hist-hist-TFormulaHessianTests`
- `gtest-math-mathcore-CladDerivatorTests` and `gtest-math-mathcore-EnzymeDerivatorTests`
- `minuit2_testADMinim`
- `gtest-roofit-codegen-testCodegen`, `gtest-roofit-roofitcore-testRooFuncWrapper`, `test-stressroofit-codegen_no_grad`, and `test-stressroofit-codegen`
- `gtest-roofit-histfactory-testHistFactory` and `gtest-roofit-histfactory-testHistFactoryPlotting`
- `roofit-create-onnx-model` and `gtest-roofit-roofit-testRooONNXFunc`
- `tmva-sofie-test-emitFromONNX-build`, `SofieGenerateModels_ONNX`, `SofieCompileModels_ONNX`, `TestCustomModelsFromONNX`, `gtest-tmva-sofie-TestGemmDerivative`, `TestCladAutodiff`, and `TestEnzymeAutodiff`
- `gtest-core-clingutils-coreclingutilstestUnit`

### Stability Improvements Over Clad
1. **Repeated SOFIE Reverse-Mode Calls**: Reusing a `Session` shadow buffer across iterations in Clad (`Linear_16` 10-layer MLP pullback) leaves stale intermediate activation adjoints unzeroed, causing exponential error blowup (`MaxErr = 2.31e+15` after warmup/timing iterations). Enzyme zeroes reverse-pass intermediate adjoints automatically and maintains `MaxErr = 3.08e-05` across 500 repeated calls even with a single preallocated shadow session.
2. **Second-Order SOFIE Hessians**: Computing the `48 x 48` Hessian of `LinearWithSigmoid` in Clad cannot use `clad::hessian` directly on `Session const &` (requiring a manual `pushforward` + `pullback` loop) and emits Clang custom-derivative signature errors on `exp_pushforward<float, float>`. With Enzyme, `clad::hessian(bench_sig, "input[0:47]").execute(...)` runs in a single call with zero errors or warnings.

---

## 3. Performance Benchmarks (Enzyme vs. Clad vs. Finite Differences)

### Benchmark Configuration
- **CPU**: `AMD EPYC 7B13`, single-thread pinned execution (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`) on an idle host, sequential back-to-back runs.
- **Compiler & Flags**: GCC `15.2.0`, `-DCMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG`), in-tree LLVM `22`, system OpenBLAS, `RooBatchCompute_AVX512`.
- **Source**: `math/mathcore/test/EnzymeADBenchmark.cxx` (compiled identically against the `-Denzyme=ON -Dclad=OFF` build and the `-Dclad=ON -Denzyme=OFF` baseline build).

### 3.1 `TFormula` Gradient and Hessian (`NPar = 6, 15, 30`)

Multi-Gaussian formula parameterized by `NPar` parameters:

| `NPar` | Enzyme `GradJIT` | Clad `GradJIT` | Enzyme `AD_Grad` | Clad `AD_Grad` | `FD_Grad` | **Grad Speedup vs Clad (vs FD)** | Enzyme `AD_Hess` | Clad `AD_Hess` | `FD_Hess` | **Hess Speedup vs Clad (vs FD)** |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | :--- |
| **6** | **103.14 ms** | 847.25 ms | **28.6 ns** | 33.8 ns | 300.9 ns | **1.18x** (**10.5x**) | **42.9 ns** | 399.4 ns | 1,781.2 ns | **9.31x** (**41.5x**) |
| **15** | **11.85 ms** | 13.90 ms | **63.0 ns** | 74.4 ns | 1,280.9 ns | **1.18x** (**20.3x**) | **114.0 ns** | 2,439.0 ns | 18,748.7 ns | **21.39x** (**164.5x**) |
| **30** | 19.96 ms | 19.37 ms | **121.1 ns** | 145.5 ns | 4,363.4 ns | **1.20x** (**36.0x**) | **3,758.6 ns** | 9,952.3 ns | 129,279.2 ns | **2.65x** (**34.4x**) |

Both backends match to machine precision (`MaxGradErr <= 4.57e-10`, `MaxHessErr <= 1.29e-07` vs. central finite differences). Initial AD JIT startup (`NPar = 6`) is **8.21x faster** with Enzyme.

---

### 3.2 `RooFit` Unbinned Mixture NLL — Parameter Scaling (`N = 10,000` Events)

Unbinned `RooAddPdf` mixture of `2, 6, 10, 20` Gaussians plus an exponential background (`NPar = 7, 19, 31, 61` free parameters) using `RooFit::EvalBackend::Codegen()`, including full `Minuit2` `migrad() + hesse()` fit time vs. vectorized AVX-512 `EvalBackend::Cpu()`:

| `NPar` | Enzyme `AD_Grad` | Clad `AD_Grad` | `FD_Grad` | **Grad vs Clad (vs FD)** | Enzyme `AD_Hess` | Clad `AD_Hess` | `FD_Hess` | **Hess vs Clad (vs FD)** | Enzyme `AD_Fit` | Clad `AD_Fit` | AVX512 `CPU_Fit` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| **7** | **579.58 us** | 2,317.21 us | 4,646.07 us | **4.00x** (**8.0x**) | **3.49 ms** | 61.79 ms | 32.84 ms | **17.71x** (**9.4x**) | **90.02 ms** | 106.77 ms | 41.65 ms |
| **19** | **1,389.94 us** | 4,624.98 us | 22,597.63 us | **3.33x** (**16.3x**) | **27.45 ms** | 345.24 ms | 433.18 ms | **12.58x** (**15.8x**) | **221.49 ms** | 285.93 ms | 324.62 ms |
| **31** | **2,088.14 us** | 6,885.01 us | 52,910.13 us | **3.30x** (**25.3x**) | **87.34 ms** | 774.31 ms | 1,641.49 ms | **8.87x** (**18.8x**) | 3,005.57 ms | 2,677.21 ms | 3,975.59 ms |
| **61** | **3,876.05 us** | 12,805.73 us | 204,772.44 us | **3.30x** (**52.8x**) | **403.30 ms** | 3,339.51 ms | 12,484.40 ms | **8.28x** (**31.0x**) | **9,372.32 ms** | 10,207.20 ms | 11,997.80 ms |

---

### 3.3 `RooFit` Unbinned Mixture NLL — Event Scaling (`NPar = 19` Parameters)

| `NEvents` | Enzyme `HessJIT` | Clad `HessJIT` | Enzyme `AD_Grad` | Clad `AD_Grad` | **Grad vs Clad** | Enzyme `AD_Hess` | Clad `AD_Hess` | **Hess vs Clad** | Enzyme `AD_Fit` | Clad `AD_Fit` | AVX512 `CPU_Fit` |
| ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | :--- | ---: | ---: | ---: |
| **1,000** | 846.52 ms | 171.14 ms | **139.07 us** | 466.71 us | **3.36x** | **2.74 ms** | 37.68 ms | **13.77x** | **43.19 ms** | 47.81 ms | 53.14 ms |
| **10,000** | 864.46 ms | 469.49 ms | **1,378.56 us** | 4,638.31 us | **3.36x** | **27.28 ms** | 347.88 ms | **12.75x** | **205.98 ms** | 255.04 ms | 276.58 ms |
| **50,000** | **969.80 ms** | 2,005.91 ms | **6,891.05 us** | 23,141.20 us | **3.36x** | **136.62 ms** | 1,850.70 ms | **13.55x** | **1,469.12 ms** | 1,714.19 ms | 1,685.15 ms |

Clad's Hessian JIT compile time scales upward with dataset size (`171 ms` to `2,006 ms`), whereas Enzyme's Hessian JIT remains nearly constant (`847 ms` to `970 ms`, **2.07x faster JIT** at `50,000` events) while delivering **3.36x faster gradients** and **13.55x faster Hessians**.

---

### 3.4 `TMVA::SOFIE` ONNX Neural Network AD (`OpenBLAS`)

| Workload | Enzyme Runtime (`MaxErr`) | Clad Runtime (`MaxErr`) | Central FD Runtime | **Enzyme Speedup vs Clad (vs FD)** |
| :--- | :--- | :--- | :--- | :--- |
| **`Linear_16` (`1600 -> 160` 10-Layer MLP, Preallocated Shadow)** | **`65.50 us`** (`3.08e-05`, exact) | `134.74 us` (`2.31e+15`, diverges) | `64,210.57 us` | **2.06x faster and exact** (**980x vs FD**) |
| **`Linear_16` (`1600 -> 160` 10-Layer MLP, Per-Call Zeroed / `clad::gradient`)** | **`70.69 us`** (`3.08e-05`, exact) | `146.14 us` (`3.08e-05`, exact) | `64,210.57 us` | **2.07x faster** (**908x vs FD**) |
| **`LinearWithSigmoid` (`48 -> 24` Gemm+Sigmoid, Full `48 x 48` Hessian)** | **`162.96 us`** (`4.23e-07`, 0 warnings) | `701.25 us` (`3.74e-07`, 2 Clang errors) | `2,086.02 us` | **4.30x faster** (**12.8x vs FD**) |

---

## 4. Building and Running the Benchmark Suite

```bash
# 1. Build ROOT with Enzyme (enabled by default: -Denzyme=ON -Dclad=OFF)
cmake -B build-enzyme -S . -GNinja -DCMAKE_BUILD_TYPE=Release -Dtesting=ON -Droottest=OFF
ninja -C build-enzyme

# 2. Generate SOFIE ONNX test models and run the AD CTest suites
(cd build-enzyme/tmva/sofie/test && ROOTIGNOREPREFIX=1 ./emitFromONNX)
ctest --test-dir build-enzyme -j16 --output-on-failure \
  -R "TFormulaGradientTests|TFormulaHessianTests|CladDerivatorTests|EnzymeDerivatorTests|minuit2_testADMinim|testCodegen|testRooFuncWrapper|testHistFactory|test-stressroofit-codegen|roofit-create-onnx-model|testRooONNXFunc|emitFromONNX-build|SofieGenerateModels_ONNX|SofieCompileModels_ONNX|TestCustomModelsFromONNX|TestGemmDerivative|TestCladAutodiff|TestEnzymeAutodiff"

# 3. Compile and run the benchmark binary from inside build-enzyme
g++ -O3 -DNDEBUG -fPIC $(./build-enzyme/bin/root-config --cflags) \
    math/mathcore/test/EnzymeADBenchmark.cxx \
    $(./build-enzyme/bin/root-config --libs) -lRooFit -lRooFitCore -lMinuit2 \
    -Wl,-rpath,"$(pwd)/build-enzyme/lib" -o build-enzyme/bench_ad
(cd build-enzyme && ROOTSYS=$PWD LD_LIBRARY_PATH=$PWD/lib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./bench_ad)
```
