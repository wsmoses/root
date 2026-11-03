// Comprehensive Apples-to-Apples Benchmark Suite: Enzyme vs. Clad vs. Numerical Diff in ROOT
// Compiles identically against both Enzyme (-Denzyme=ON) and Clad (-Dclad=ON) builds.

#include <RConfigure.h>
#include <TApplication.h>
#include <TF1.h>
#include <TFormula.h>
#include <TH1D.h>
#include <TInterpreter.h>
#include <TRandom3.h>
#include <TROOT.h>

#include <RooAddPdf.h>
#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooEvaluatorWrapper.h>
#include <RooExponential.h>
#include <RooExtendPdf.h>
#include <RooFitResult.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
#include <RooMsgService.h>
#include <RooProdPdf.h>
#include <RooRandom.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>

#include <TMVA/RModel.hxx>
#include <TMVA/RModelParser_ONNX.hxx>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double elapsedMs(Clock::time_point t0, Clock::time_point t1)
{
   return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double elapsedUs(Clock::time_point t0, Clock::time_point t1)
{
   return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

static double elapsedNs(Clock::time_point t0, Clock::time_point t1)
{
   return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

static const char *backendName()
{
#if defined(R__HAS_ENZYME) && !defined(R__HAS_CLAD)
   return "Enzyme";
#elif defined(R__HAS_CLAD)
   return "Clad";
#else
   return "NoAD";
#endif
}

// -----------------------------------------------------------------------------
// 1. TFormula Gradient & Hessian Benchmark (P = 6, 15, 30 parameters)
// -----------------------------------------------------------------------------
static void benchTFormula()
{
   std::printf("\n=== [1] TFormula Gradient & Hessian Benchmark (%s) ===\n", backendName());
   std::printf("%-8s | %12s %12s | %12s %12s | %12s %12s | %10s %10s\n",
               "NPar", "GradJIT(ms)", "HessJIT(ms)", "AD_Grad(ns)", "FD_Grad(ns)",
               "AD_Hess(ns)", "FD_Hess(ns)", "MaxGradErr", "MaxHessErr");
   std::printf("--------------------------------------------------------------------------------------------------------------------\n");

   for (int nPeaks : {2, 5, 10}) {
      const int nPar = 3 * nPeaks;
      std::ostringstream expr;
      for (int k = 0; k < nPeaks; ++k) {
         if (k > 0)
            expr << " + ";
         int b = 3 * k;
         expr << "[" << b << "]*exp(-0.5*((x-[" << (b + 1) << "])/[" << (b + 2) << "])^2)";
      }

      std::string fname = "f_bench_" + std::to_string(nPar);
      TF1 tf(fname.c_str(), expr.str().c_str(), -10.0, 10.0);
      TFormula *form = tf.GetFormula();
      std::vector<double> params(nPar);
      for (int k = 0; k < nPeaks; ++k) {
         params[3 * k + 0] = 1.0 + 0.2 * k;
         params[3 * k + 1] = -4.0 + (8.0 * k) / std::max(1, nPeaks - 1);
         params[3 * k + 2] = 0.8 + 0.1 * (k % 3);
      }
      form->SetParameters(params.data());
      tf.SetParameters(params.data());

      double x = 0.35;
      // Warmup primal
      volatile double sink = form->EvalPar(&x, params.data());
      (void)sink;

      // Measure Gradient JIT time (including first execution to trigger machine code emission)
      std::vector<double> adGrad(nPar, 0.0);
      auto tg0 = Clock::now();
      bool okG = form->GenerateGradientPar();
      form->GradientPar(&x, adGrad);
      auto tg1 = Clock::now();
      double gradJitMs = elapsedMs(tg0, tg1);

      // Measure Hessian JIT time (including first execution)
      std::vector<double> adHess(nPar * nPar, 0.0);
      auto th0 = Clock::now();
      bool okH = form->GenerateHessianPar();
      form->HessianPar(&x, adHess);
      auto th1 = Clock::now();
      double hessJitMs = elapsedMs(th0, th1);

      if (!okG || !okH) {
         std::printf("ERROR: TFormula GenerateGradientPar/GenerateHessianPar failed for nPar=%d\n", nPar);
         continue;
      }

      // Numerical central-difference reference + timing
      std::vector<double> fdGrad(nPar, 0.0);
      std::vector<double> fdHess(nPar * nPar, 0.0);
      std::vector<double> pWork = params;
      const double epsG = 1e-6;
      const double epsH = 1e-4;

      auto computeFdGrad = [&]() {
         for (int i = 0; i < nPar; ++i) {
            double orig = pWork[i];
            pWork[i] = orig + epsG;
            double fp = form->EvalPar(&x, pWork.data());
            pWork[i] = orig - epsG;
            double fm = form->EvalPar(&x, pWork.data());
            pWork[i] = orig;
            fdGrad[i] = (fp - fm) / (2.0 * epsG);
         }
      };

      auto computeFdHess = [&]() {
         double f0 = form->EvalPar(&x, pWork.data());
         for (int i = 0; i < nPar; ++i) {
            double oi = pWork[i];
            pWork[i] = oi + epsH;
            double fip = form->EvalPar(&x, pWork.data());
            pWork[i] = oi - epsH;
            double fim = form->EvalPar(&x, pWork.data());
            pWork[i] = oi;
            fdHess[i * nPar + i] = (fip - 2.0 * f0 + fim) / (epsH * epsH);
            for (int j = i + 1; j < nPar; ++j) {
               double oj = pWork[j];
               pWork[i] = oi + epsH;
               pWork[j] = oj + epsH;
               double fpp = form->EvalPar(&x, pWork.data());
               pWork[j] = oj - epsH;
               double fpm = form->EvalPar(&x, pWork.data());
               pWork[i] = oi - epsH;
               pWork[j] = oj + epsH;
               double fmp = form->EvalPar(&x, pWork.data());
               pWork[j] = oj - epsH;
               double fmm = form->EvalPar(&x, pWork.data());
               pWork[i] = oi;
               pWork[j] = oj;
               double hij = (fpp - fpm - fmp + fmm) / (4.0 * epsH * epsH);
               fdHess[i * nPar + j] = hij;
               fdHess[j * nPar + i] = hij;
            }
         }
      };

      computeFdGrad();
      computeFdHess();

      double maxGradErr = 0.0;
      for (int i = 0; i < nPar; ++i)
         maxGradErr = std::max(maxGradErr, std::abs(adGrad[i] - fdGrad[i]));
      double maxHessErr = 0.0;
      for (int i = 0; i < nPar * nPar; ++i)
         maxHessErr = std::max(maxHessErr, std::abs(adHess[i] - fdHess[i]));

      // Throughput timing: AD Gradient
      const int nIterG = 200000;
      auto t0 = Clock::now();
      for (int it = 0; it < nIterG; ++it) {
         std::fill(adGrad.begin(), adGrad.end(), 0.0);
         form->GradientPar(&x, adGrad.data());
      }
      auto t1 = Clock::now();
      double adGradNs = elapsedNs(t0, t1) / nIterG;

      // Throughput timing: FD Gradient
      const int nIterFdG = 50000;
      auto t2 = Clock::now();
      for (int it = 0; it < nIterFdG; ++it) {
         computeFdGrad();
      }
      auto t3 = Clock::now();
      double fdGradNs = elapsedNs(t2, t3) / nIterFdG;

      // Throughput timing: AD Hessian
      const int nIterH = 50000;
      auto t4 = Clock::now();
      for (int it = 0; it < nIterH; ++it) {
         std::fill(adHess.begin(), adHess.end(), 0.0);
         form->HessianPar(&x, adHess.data());
      }
      auto t5 = Clock::now();
      double adHessNs = elapsedNs(t4, t5) / nIterH;

      // Throughput timing: FD Hessian
      const int nIterFdH = 5000;
      auto t6 = Clock::now();
      for (int it = 0; it < nIterFdH; ++it) {
         computeFdHess();
      }
      auto t7 = Clock::now();
      double fdHessNs = elapsedNs(t6, t7) / nIterFdH;

      std::printf("%-8d | %12.2f %12.2f | %12.1f %12.1f | %12.1f %12.1f | %10.2e %10.2e\n",
                  nPar, gradJitMs, hessJitMs, adGradNs, fdGradNs, adHessNs, fdHessNs, maxGradErr, maxHessErr);
      std::fflush(stdout);
   }
}

// -----------------------------------------------------------------------------
// 2 & 3. RooFit Unbinned Mixture Likelihood Benchmark (Parameter & Event Scaling)
// -----------------------------------------------------------------------------
struct RooBenchResult {
   int nParams;
   int nEvents;
   double gradJitMs;
   double hessJitMs;
   double adGradUs;
   double fdGradUs;
   double adHessUs;
   double fdHessUs;
   double adFitMs;
   double cpuFitMs;
   double maxGradRelErr;
   double maxHessRelErr;
};

static RooBenchResult runRooMixtureBench(int nGauss, int nEvents, int tagId)
{
   RooHelpers::LocalChangeMsgLevel chmsg{RooFit::WARNING, 0u, RooFit::Fitting, true};
   RooHelpers::LocalChangeMsgLevel chmsg2{RooFit::WARNING, 0u, RooFit::Minimization, true};
   RooHelpers::LocalChangeMsgLevel chmsg3{RooFit::WARNING, 0u, RooFit::Eval, true};

   std::string sfx = "_" + std::to_string(tagId);
   RooRealVar x(("x" + sfx).c_str(), "x", -10.0, 10.0);

   std::vector<std::unique_ptr<RooRealVar>> means, sigmas, fracs;
   std::vector<std::unique_ptr<RooGaussian>> gaussians;
   RooArgList pdfList, coefList;

   for (int k = 0; k < nGauss; ++k) {
      std::string ks = std::to_string(k) + sfx;
      double mu0 = -6.0 + (12.0 * k) / std::max(1, nGauss - 1);
      means.push_back(std::make_unique<RooRealVar>(("mu_" + ks).c_str(), "mu", mu0, -10.0, 10.0));
      sigmas.push_back(std::make_unique<RooRealVar>(("sig_" + ks).c_str(), "sigma", 0.7 + 0.05 * (k % 4), 0.15, 4.0));
      gaussians.push_back(
         std::make_unique<RooGaussian>(("g_" + ks).c_str(), "g", x, *means.back(), *sigmas.back()));
      pdfList.add(*gaussians.back());
      fracs.push_back(
         std::make_unique<RooRealVar>(("f_" + ks).c_str(), "frac", 0.75 / nGauss, 0.001, 0.99));
      coefList.add(*fracs.back());
   }

   RooRealVar slope(("slope" + sfx).c_str(), "slope", -0.08, -1.0, -0.001);
   RooExponential bkg(("bkg" + sfx).c_str(), "bkg", x, slope);
   pdfList.add(bkg);

   // Total free parameters: 3 * nGauss + 1 (mu_k, sig_k, f_k for k=0..nGauss-1, plus slope)
   RooAddPdf model(("model" + sfx).c_str(), "model", pdfList, coefList);

   RooRandom::randomGenerator()->SetSeed(42 + tagId);
   std::unique_ptr<RooDataSet> data{model.generate(RooArgSet(x), nEvents)};

   // Record true values, then perturb slightly while keeping sum(fracs) < 1 for all nGauss
   RooArgSet allVars;
   model.getParameters(data->get(), allVars);
   std::vector<RooRealVar *> freeVars;
   std::vector<double> initVals;
   for (auto *arg : allVars) {
      if (auto *rv = dynamic_cast<RooRealVar *>(arg)) {
         if (!rv->isConstant()) {
            freeVars.push_back(rv);
            double v = rv->getVal();
            std::string nm = rv->GetName();
            double perturbed = v;
            if (nm.rfind("f_", 0) == 0) {
               perturbed = v * 0.90;
            } else if (nm.rfind("mu_", 0) == 0) {
               perturbed = v + 0.08;
            } else if (nm.rfind("sig_", 0) == 0) {
               perturbed = v * 1.06;
            } else {
               perturbed = v * 1.05;
            }
            perturbed = std::clamp(perturbed, rv->getMin() + 1e-4, rv->getMax() - 1e-4);
            initVals.push_back(perturbed);
         }
      }
   }
   const int nPar = static_cast<int>(freeVars.size());
   auto resetParams = [&]() {
      for (int i = 0; i < nPar; ++i)
         freeVars[i]->setVal(initVals[i]);
   };
   resetParams();

   // Create Codegen NLL (which automatically generates the AD gradient) & run 1st gradient call
   auto tg0 = Clock::now();
   std::unique_ptr<RooAbsReal> nllCodegen{model.createNLL(*data, RooFit::EvalBackend::Codegen())};
   auto *evalWrap = dynamic_cast<RooFit::Experimental::RooEvaluatorWrapper *>(nllCodegen.get());
   std::vector<double> adGrad(nPar, 0.0);
   evalWrap->gradient(adGrad.data());
   auto tg1 = Clock::now();
   double gradJitMs = elapsedMs(tg0, tg1);

   // Full Minuit2 Migrad+Hesse Fit with AD (default EvalBackend::Codegen() setup, pre-JIT'd)
   resetParams();
   evalWrap->getVal(); // warmup primal batch evaluator
   auto tf0 = Clock::now();
   {
      RooMinimizer mAd(*nllCodegen);
      mAd.setPrintLevel(-1);
      mAd.setStrategy(1);
      mAd.migrad();
      mAd.hesse();
   }
   auto tf1 = Clock::now();
   double adFitMs = elapsedMs(tf0, tf1);

   // Full Minuit2 Migrad+Hesse Fit with CPU backend (numerical FD gradients)
   resetParams();
   std::unique_ptr<RooAbsReal> nllCpu{model.createNLL(*data, RooFit::EvalBackend::Cpu())};
   nllCpu->getVal(); // warmup primal batch evaluator
   auto tf2 = Clock::now();
   {
      RooMinimizer mCpu(*nllCpu);
      mCpu.setPrintLevel(-1);
      mCpu.setStrategy(1);
      mCpu.migrad();
      mCpu.hesse();
   }
   auto tf3 = Clock::now();
   double cpuFitMs = elapsedMs(tf2, tf3);

   // Reset to perturbed point for exact AD vs FD Gradient & Hessian microbenchmarks
   resetParams();
   evalWrap->gradient(adGrad.data());

   std::vector<double> adHess(nPar * nPar, 0.0);
   auto th0 = Clock::now();
   evalWrap->generateHessian();
   evalWrap->hessian(adHess.data());
   auto th1 = Clock::now();
   double hessJitMs = elapsedMs(th0, th1);

   evalWrap->setUseGeneratedFunctionCode(true);

   // Numerical FD gradient & Hessian on NLL evaluation
   std::vector<double> fdGrad(nPar, 0.0);
   std::vector<double> fdHess(nPar * nPar, 0.0);
   const double epsG = 1e-6;
   const double epsH = 1e-4;

   auto computeFdGrad = [&]() {
      for (int i = 0; i < nPar; ++i) {
         double orig = freeVars[i]->getVal();
         freeVars[i]->setVal(orig + epsG);
         double fp = evalWrap->getVal();
         freeVars[i]->setVal(orig - epsG);
         double fm = evalWrap->getVal();
         freeVars[i]->setVal(orig);
         fdGrad[i] = (fp - fm) / (2.0 * epsG);
      }
   };

   auto computeFdHess = [&]() {
      double f0 = evalWrap->getVal();
      for (int i = 0; i < nPar; ++i) {
         double oi = freeVars[i]->getVal();
         freeVars[i]->setVal(oi + epsH);
         double fip = evalWrap->getVal();
         freeVars[i]->setVal(oi - epsH);
         double fim = evalWrap->getVal();
         freeVars[i]->setVal(oi);
         fdHess[i * nPar + i] = (fip - 2.0 * f0 + fim) / (epsH * epsH);
         for (int j = i + 1; j < nPar; ++j) {
            double oj = freeVars[j]->getVal();
            freeVars[i]->setVal(oi + epsH);
            freeVars[j]->setVal(oj + epsH);
            double fpp = evalWrap->getVal();
            freeVars[j]->setVal(oj - epsH);
            double fpm = evalWrap->getVal();
            freeVars[i]->setVal(oi - epsH);
            freeVars[j]->setVal(oj + epsH);
            double fmp = evalWrap->getVal();
            freeVars[j]->setVal(oj - epsH);
            double fmm = evalWrap->getVal();
            freeVars[i]->setVal(oi);
            freeVars[j]->setVal(oj);
            double hij = (fpp - fpm - fmp + fmm) / (4.0 * epsH * epsH);
            fdHess[i * nPar + j] = hij;
            fdHess[j * nPar + i] = hij;
         }
      }
   };

   computeFdGrad();
   computeFdHess();

   double maxGradRel = 0.0;
   for (int i = 0; i < nPar; ++i) {
      double denom = std::max(1.0, std::abs(fdGrad[i]));
      maxGradRel = std::max(maxGradRel, std::abs(adGrad[i] - fdGrad[i]) / denom);
   }
   double maxHessRel = 0.0;
   for (int i = 0; i < nPar * nPar; ++i) {
      double denom = std::max(1.0, std::abs(fdHess[i]));
      maxHessRel = std::max(maxHessRel, std::abs(adHess[i] - fdHess[i]) / denom);
   }

   // Time AD Gradient
   const int nRepG = (nEvents <= 10000) ? 200 : 40;
   auto t0 = Clock::now();
   for (int r = 0; r < nRepG; ++r) {
      evalWrap->gradient(adGrad.data());
   }
   auto t1 = Clock::now();
   double adGradUs = elapsedUs(t0, t1) / nRepG;

   // Time FD Gradient
   const int nRepFdG = std::max(5, nRepG / 10);
   auto t2 = Clock::now();
   for (int r = 0; r < nRepFdG; ++r) {
      computeFdGrad();
   }
   auto t3 = Clock::now();
   double fdGradUs = elapsedUs(t2, t3) / nRepFdG;

   // Time AD Hessian
   const int nRepH = (nEvents <= 10000 && nPar <= 35) ? 50 : 10;
   auto t4 = Clock::now();
   for (int r = 0; r < nRepH; ++r) {
      evalWrap->hessian(adHess.data());
   }
   auto t5 = Clock::now();
   double adHessUs = elapsedUs(t4, t5) / nRepH;

   // Time FD Hessian
   const int nRepFdH = (nPar <= 20 && nEvents <= 10000) ? 3 : 1;
   auto t6 = Clock::now();
   for (int r = 0; r < nRepFdH; ++r) {
      computeFdHess();
   }
   auto t7 = Clock::now();
   double fdHessUs = elapsedUs(t6, t7) / nRepFdH;

   return {nPar, nEvents, gradJitMs, hessJitMs, adGradUs, fdGradUs, adHessUs, fdHessUs, adFitMs, cpuFitMs,
           maxGradRel, maxHessRel};
}

// -----------------------------------------------------------------------------
// 4. TMVA::SOFIE Neural Network AD Benchmark (Linear_16 + LinearWithSigmoid)
// -----------------------------------------------------------------------------
static void benchSOFIE()
{
   std::printf("\n=== [4] TMVA::SOFIE ONNX Neural Network AD Benchmark (%s) ===\n", backendName());

   // Build two ONNX models programmatically in memory / scratch or load from sofie test build dir:
   // Check if tmva/sofie/test models exist in the build directory.
   std::ifstream check16("tmva/sofie/test/Linear_16_FromONNX_unoptimized.hxx");
   std::ifstream checkSig("tmva/sofie/test/LinearWithSigmoid_FromONNX_unoptimized.hxx");
   if (!check16.good() || !checkSig.good()) {
      std::printf("  [Skipping SOFIE benchmark: generated ONNX headers not found in ./tmva/sofie/test/]\n");
      return;
   }

   gInterpreter->AddIncludePath("tmva/sofie/test");

   // --- 4A: Linear_16 (1600 -> 160 MLP) Reverse Gradient ---
   gInterpreter->Declare(R"cpp(
#include "Linear_16_FromONNX_unoptimized.hxx"
#include <Math/CladDerivator.h>

float bench_lin16(TMVA_SOFIE_Linear_16::Session const &session, float const *input)
{
   float out[160]{};
   TMVA_SOFIE_Linear_16::doInfer(session, input, out);
   float s = 0.0f;
   for (int i = 0; i < 160; ++i) s += out[i];
   return s;
}

float bench_lin16_outer(TMVA_SOFIE_Linear_16::Session const &session, float const *input)
{
   return bench_lin16(session, input);
}
   )cpp");

   gInterpreter->ProcessLine(
      "TMVA_SOFIE_Linear_16::Session g_bench_s16{\"tmva/sofie/test/Linear_16_FromONNX_unoptimized.dat\"};");
   gInterpreter->Declare(R"cpp(
std::vector<float> g_bench_in16(1600, 0.15f);
std::vector<float> g_bench_grad16(1600, 0.0f);
std::vector<float> g_bench_fd16(1600, 0.0f);
   )cpp");

   auto tj0 = Clock::now();
#if defined(R__HAS_ENZYME) && !defined(R__HAS_CLAD)
   gInterpreter->Declare(R"cpp(
auto g_bench_s16_shadow = g_bench_s16.CreateZeroShadow();
void run_bench_lin16_grad() {
   std::fill(g_bench_grad16.begin(), g_bench_grad16.end(), 0.0f);
   ROOT::Math::Enzyme::Detail::EnzymeFnThunkGrad2Shadow<float, TMVA_SOFIE_Linear_16::Session, float, true>(
      bench_lin16, &g_bench_s16, &g_bench_s16_shadow, g_bench_in16.data(), g_bench_grad16.data());
}
   )cpp");
   gInterpreter->ProcessLine("run_bench_lin16_grad();");
#else
   gInterpreter->ProcessLine("clad::gradient(bench_lin16_outer, \"input\");");
   gInterpreter->ProcessLine(
      "TMVA_SOFIE_Linear_16::Session g_bench_ds16 = g_bench_s16.CreateZeroShadow();");
   gInterpreter->Declare(R"cpp(
void run_bench_lin16_grad() {
   std::fill(g_bench_grad16.begin(), g_bench_grad16.end(), 0.0f);
   bench_lin16_pullback(g_bench_s16, g_bench_in16.data(), 1.0f, &g_bench_ds16, g_bench_grad16.data());
}
   )cpp");
   gInterpreter->ProcessLine("run_bench_lin16_grad();");
#endif
   auto tj1 = Clock::now();
   double lin16JitMs = elapsedMs(tj0, tj1);

   auto *runLin16Grad =
      reinterpret_cast<void (*)()>(gInterpreter->ProcessLine("static_cast<void (*)()>(run_bench_lin16_grad);"));
   gInterpreter->Declare(R"cpp(
double bench_lin16_num_eval(TMVA_SOFIE_Linear_16::Session const &session, float const *input) {
   float out[160]{};
   TMVA_SOFIE_Linear_16::doInfer(session, input, out);
   double s = 0.0;
   for (int i = 0; i < 160; ++i) s += out[i];
   return s;
}
void run_bench_lin16_fd() {
   const float eps = 1e-3f;
   for (int i = 0; i < 1600; ++i) {
      float orig = g_bench_in16[i];
      g_bench_in16[i] = orig + eps;
      double fp = bench_lin16_num_eval(g_bench_s16, g_bench_in16.data());
      g_bench_in16[i] = orig - eps;
      double fm = bench_lin16_num_eval(g_bench_s16, g_bench_in16.data());
      g_bench_in16[i] = orig;
      g_bench_fd16[i] = static_cast<float>((fp - fm) / (2.0 * eps));
   }
}
   )cpp");
   auto *runLin16Fd =
      reinterpret_cast<void (*)()>(gInterpreter->ProcessLine("static_cast<void (*)()>(run_bench_lin16_fd);"));
   runLin16Fd();

   const int nRep16 = 500;
   auto t0 = Clock::now();
   for (int r = 0; r < nRep16; ++r)
      runLin16Grad();
   auto t1 = Clock::now();
   double lin16AdUs = elapsedUs(t0, t1) / nRep16;

   const int nRep16Fd = 5;
   auto t2 = Clock::now();
   for (int r = 0; r < nRep16Fd; ++r)
      runLin16Fd();
   auto t3 = Clock::now();
   double lin16FdUs = elapsedUs(t2, t3) / nRep16Fd;

   auto *grad16Ptr = reinterpret_cast<const float *>(gInterpreter->ProcessLine("g_bench_grad16.data();"));
   auto *fd16Ptr = reinterpret_cast<const float *>(gInterpreter->ProcessLine("g_bench_fd16.data();"));
   float lin16MaxErr = 0.0f;
   for (int i = 0; i < 1600; ++i)
      lin16MaxErr = std::max(lin16MaxErr, std::abs(grad16Ptr[i] - fd16Ptr[i]));

   std::printf("  Linear_16 (1600->160 Pullback)   | GradJIT = %8.2f ms | AD_Grad = %8.2f us | FD_Grad = %10.2f us | MaxErr = %.2e\n",
               lin16JitMs, lin16AdUs, lin16FdUs, lin16MaxErr);
   std::fflush(stdout);

#if defined(R__HAS_ENZYME) && !defined(R__HAS_CLAD)
   auto tji0 = Clock::now();
   gInterpreter->Declare(R"cpp(
auto g_bench_lin16_in_grad = clad::gradient(bench_lin16, "input");
void run_bench_lin16_input_only() {
   std::fill(g_bench_grad16.begin(), g_bench_grad16.end(), 0.0f);
   g_bench_lin16_in_grad.execute(g_bench_s16, g_bench_in16.data(), g_bench_grad16.data());
}
   )cpp");
   gInterpreter->ProcessLine("run_bench_lin16_input_only();");
   auto tji1 = Clock::now();
   double lin16InJitMs = elapsedMs(tji0, tji1);
   auto *runLin16InOnly =
      reinterpret_cast<void (*)()>(gInterpreter->ProcessLine("static_cast<void (*)()>(run_bench_lin16_input_only);"));
   auto ti0 = Clock::now();
   for (int r = 0; r < nRep16; ++r)
      runLin16InOnly();
   auto ti1 = Clock::now();
   double lin16InAdUs = elapsedUs(ti0, ti1) / nRep16;
   float lin16InMaxErr = 0.0f;
   for (int i = 0; i < 1600; ++i)
      lin16InMaxErr = std::max(lin16InMaxErr, std::abs(grad16Ptr[i] - fd16Ptr[i]));
   std::printf("  Linear_16 (1600->160 Input-Only) | GradJIT = %8.2f ms | AD_Grad = %8.2f us | FD_Grad = %10.2f us | MaxErr = %.2e\n",
               lin16InJitMs, lin16InAdUs, lin16FdUs, lin16InMaxErr);
#else
   gInterpreter->Declare(R"cpp(
void run_bench_lin16_grad_zeroed() {
   auto ds = g_bench_s16.CreateZeroShadow();
   std::fill(g_bench_grad16.begin(), g_bench_grad16.end(), 0.0f);
   bench_lin16_pullback(g_bench_s16, g_bench_in16.data(), 1.0f, &ds, g_bench_grad16.data());
}
   )cpp");
   auto *runLin16Zeroed =
      reinterpret_cast<void (*)()>(gInterpreter->ProcessLine("static_cast<void (*)()>(run_bench_lin16_grad_zeroed);"));
   auto tz0 = Clock::now();
   for (int r = 0; r < nRep16; ++r)
      runLin16Zeroed();
   auto tz1 = Clock::now();
   double lin16ZeroedAdUs = elapsedUs(tz0, tz1) / nRep16;
   float lin16ZeroedMaxErr = 0.0f;
   for (int i = 0; i < 1600; ++i)
      lin16ZeroedMaxErr = std::max(lin16ZeroedMaxErr, std::abs(grad16Ptr[i] - fd16Ptr[i]));
   std::printf("  Linear_16 (Zeroed-Shadow Valid)  | GradJIT = %8.2f ms | AD_Grad = %8.2f us | FD_Grad = %10.2f us | MaxErr = %.2e\n",
               lin16JitMs, lin16ZeroedAdUs, lin16FdUs, lin16ZeroedMaxErr);
#endif

   // --- 4B: LinearWithSigmoid (48 -> 24 Gemm+Sigmoid) Gradient & Full 48x48 Hessian ---
   gInterpreter->Declare(R"cpp(
#include "LinearWithSigmoid_FromONNX_unoptimized.hxx"

float bench_sig(TMVA_SOFIE_LinearWithSigmoid::Session const &session, float const *input)
{
   float out[24]{};
   TMVA_SOFIE_LinearWithSigmoid::doInfer(session, input, out);
   float s = 0.0f;
   for (int i = 0; i < 24; ++i) s += out[i];
   return s;
}
float bench_sig_outer(TMVA_SOFIE_LinearWithSigmoid::Session const &session, float const *input)
{
   return bench_sig(session, input);
}
   )cpp");

   gInterpreter->ProcessLine("TMVA_SOFIE_LinearWithSigmoid::Session "
                             "g_bench_ssig{\"tmva/sofie/test/LinearWithSigmoid_FromONNX_unoptimized.dat\"};");
   gInterpreter->Declare(R"cpp(
std::vector<float> g_bench_insig(48, 0.25f);
std::vector<float> g_bench_gradsig(48, 0.0f);
std::vector<float> g_bench_hesssig(48 * 48, 0.0f);
std::vector<float> g_bench_hesssig_fd(48 * 48, 0.0f);
std::vector<float> g_bench_hesssig_ref(48 * 48, 0.0f);
   )cpp");

   auto ths0 = Clock::now();
#if defined(R__HAS_ENZYME) && !defined(R__HAS_CLAD)
   gInterpreter->Declare(R"cpp(
void run_bench_sig_hess() {
   std::fill(g_bench_hesssig.begin(), g_bench_hesssig.end(), 0.0f);
   clad::hessian(bench_sig, "input[0:47]").execute(
      g_bench_ssig, g_bench_insig.data(), clad::array_ref<float>(g_bench_hesssig.data(), 48 * 48));
}
   )cpp");
   gInterpreter->ProcessLine("run_bench_sig_hess();");
#else
   gInterpreter->ProcessLine("clad::gradient(bench_sig_outer, \"input\");");
   gInterpreter->ProcessLine("clad::differentiate(bench_sig_outer, \"input[0]\");");
   gInterpreter->Declare(R"cpp(
float bench_sig_dir(TMVA_SOFIE_LinearWithSigmoid::Session const &session,
                    TMVA_SOFIE_LinearWithSigmoid::Session const &zeroSession, float const *input, float const *dinput)
{
   return bench_sig_pushforward(session, input, zeroSession, dinput).pushforward;
}
float bench_sig_dir_outer(TMVA_SOFIE_LinearWithSigmoid::Session const &session,
                          TMVA_SOFIE_LinearWithSigmoid::Session const &zeroSession, float const *input,
                          float const *dinput)
{
   return bench_sig_dir(session, zeroSession, input, dinput);
}
   )cpp");
   gInterpreter->ProcessLine("clad::gradient(bench_sig_dir_outer, \"input, dinput\");");
   gInterpreter->Declare(R"cpp(
TMVA_SOFIE_LinearWithSigmoid::Session g_bench_zsig = g_bench_ssig.CreateZeroShadow();
void run_bench_sig_hess() {
   float dinput[48]{};
   float hvp[48]{};
   float hvp_g[48]{};
   for (int j = 0; j < 48; ++j) {
      auto dS1 = g_bench_ssig.CreateZeroShadow();
      auto dS2 = g_bench_ssig.CreateZeroShadow();
      std::fill(dinput, dinput + 48, 0.0f);
      std::fill(hvp, hvp + 48, 0.0f);
      std::fill(hvp_g, hvp_g + 48, 0.0f);
      dinput[j] = 1.0f;
      bench_sig_dir_pullback(g_bench_ssig, g_bench_zsig, g_bench_insig.data(), dinput, 1.0f,
                             &dS1, &dS2, hvp, hvp_g);
      for (int i = 0; i < 48; ++i)
         g_bench_hesssig[i * 48 + j] = hvp[i];
   }
}
   )cpp");
   gInterpreter->ProcessLine("run_bench_sig_hess();");
#endif
   auto ths1 = Clock::now();
   double sigHessJitMs = elapsedMs(ths0, ths1);

   auto *runSigHess =
      reinterpret_cast<void (*)()>(gInterpreter->ProcessLine("static_cast<void (*)()>(run_bench_sig_hess);"));
   gInterpreter->Declare(R"cpp(
double bench_sig_num_eval(TMVA_SOFIE_LinearWithSigmoid::Session const &session, float const *input) {
   float out[24]{};
   TMVA_SOFIE_LinearWithSigmoid::doInfer(session, input, out);
   double s = 0.0;
   for (int i = 0; i < 24; ++i) s += out[i];
   return s;
}
void run_bench_sig_fd_hess() {
   const float eps = 2e-3f;
   double f0 = bench_sig_num_eval(g_bench_ssig, g_bench_insig.data());
   for (int i = 0; i < 48; ++i) {
      float oi = g_bench_insig[i];
      g_bench_insig[i] = oi + eps;
      double fip = bench_sig_num_eval(g_bench_ssig, g_bench_insig.data());
      g_bench_insig[i] = oi - eps;
      double fim = bench_sig_num_eval(g_bench_ssig, g_bench_insig.data());
      g_bench_insig[i] = oi;
      g_bench_hesssig_fd[i * 48 + i] = static_cast<float>((fip - 2.0 * f0 + fim) / (double(eps) * double(eps)));
      for (int j = i + 1; j < 48; ++j) {
         float oj = g_bench_insig[j];
         g_bench_insig[i] = oi + eps;
         g_bench_insig[j] = oj + eps;
         double fpp = bench_sig_num_eval(g_bench_ssig, g_bench_insig.data());
         g_bench_insig[j] = oj - eps;
         double fpm = bench_sig_num_eval(g_bench_ssig, g_bench_insig.data());
         g_bench_insig[i] = oi - eps;
         g_bench_insig[j] = oj + eps;
         double fmp = bench_sig_num_eval(g_bench_ssig, g_bench_insig.data());
         g_bench_insig[j] = oj - eps;
         double fmm = bench_sig_num_eval(g_bench_ssig, g_bench_insig.data());
         g_bench_insig[i] = oi;
         g_bench_insig[j] = oj;
         float hij = static_cast<float>((fpp - fpm - fmp + fmm) / (4.0 * double(eps) * double(eps)));
         g_bench_hesssig_fd[i * 48 + j] = hij;
         g_bench_hesssig_fd[j * 48 + i] = hij;
      }
   }
}
   )cpp");
   auto *runSigFdHess =
      reinterpret_cast<void (*)()>(gInterpreter->ProcessLine("static_cast<void (*)()>(run_bench_sig_fd_hess);"));
   runSigFdHess();

   const int nRepSigH = 200;
   auto ts0 = Clock::now();
   for (int r = 0; r < nRepSigH; ++r)
      runSigHess();
   auto ts1 = Clock::now();
   double sigHessUs = elapsedUs(ts0, ts1) / nRepSigH;

   const int nRepSigFdH = 10;
   auto ts2 = Clock::now();
   for (int r = 0; r < nRepSigFdH; ++r)
      runSigFdHess();
   auto ts3 = Clock::now();
   double sigFdHessUs = elapsedUs(ts2, ts3) / nRepSigFdH;

#if defined(R__HAS_ENZYME) && !defined(R__HAS_CLAD)
   gInterpreter->Declare(R"cpp(
void run_bench_sig_grad_ref_hess() {
   const float eps = 1e-2f;
   auto gradFn = clad::gradient(bench_sig, "input");
   std::vector<float> gp(48), gm(48);
   for (int j = 0; j < 48; ++j) {
      float orig = g_bench_insig[j];
      std::fill(gp.begin(), gp.end(), 0.0f);
      std::fill(gm.begin(), gm.end(), 0.0f);
      g_bench_insig[j] = orig + eps;
      gradFn.execute(g_bench_ssig, g_bench_insig.data(), gp.data());
      g_bench_insig[j] = orig - eps;
      gradFn.execute(g_bench_ssig, g_bench_insig.data(), gm.data());
      g_bench_insig[j] = orig;
      for (int i = 0; i < 48; ++i)
         g_bench_hesssig_ref[i * 48 + j] = (gp[i] - gm[i]) / (2.0f * eps);
   }
}
   )cpp");
#else
   gInterpreter->Declare(R"cpp(
void run_bench_sig_grad_ref_hess() {
   const float eps = 1e-2f;
   std::vector<float> gp(48), gm(48);
   for (int j = 0; j < 48; ++j) {
      float orig = g_bench_insig[j];
      auto dSp = g_bench_ssig.CreateZeroShadow();
      auto dSm = g_bench_ssig.CreateZeroShadow();
      std::fill(gp.begin(), gp.end(), 0.0f);
      std::fill(gm.begin(), gm.end(), 0.0f);
      g_bench_insig[j] = orig + eps;
      bench_sig_pullback(g_bench_ssig, g_bench_insig.data(), 1.0f, &dSp, gp.data());
      g_bench_insig[j] = orig - eps;
      bench_sig_pullback(g_bench_ssig, g_bench_insig.data(), 1.0f, &dSm, gm.data());
      g_bench_insig[j] = orig;
      for (int i = 0; i < 48; ++i)
         g_bench_hesssig_ref[i * 48 + j] = (gp[i] - gm[i]) / (2.0f * eps);
   }
}
   )cpp");
#endif
   gInterpreter->ProcessLine("run_bench_sig_grad_ref_hess();");

   auto *hessSigPtr = reinterpret_cast<const float *>(gInterpreter->ProcessLine("g_bench_hesssig.data();"));
   auto *refHessSigPtr = reinterpret_cast<const float *>(gInterpreter->ProcessLine("g_bench_hesssig_ref.data();"));
   float sigHessMaxErr = 0.0f;
   for (int i = 0; i < 48 * 48; ++i)
      sigHessMaxErr = std::max(sigHessMaxErr, std::abs(hessSigPtr[i] - refHessSigPtr[i]));

   std::printf("  LinearWithSigmoid (48x48 Hessian)| HessJIT = %8.2f ms | AD_Hess = %8.2f us | FD_Hess = %10.2f us | MaxErr = %.2e\n",
               sigHessJitMs, sigHessUs, sigFdHessUs, sigHessMaxErr);
   std::fflush(stdout);
}

int main(int argc, char **argv)
{
   RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);

   std::printf("====================================================================================================\n");
   std::printf("  ROOT Automatic Differentiation Benchmark Suite — Active Backend: %s\n", backendName());
   std::printf("====================================================================================================\n");

   benchTFormula();

   std::printf("\n=== [2] RooFit Unbinned Mixture NLL — Parameter Scaling (N = 10,000 events, %s) ===\n", backendName());
   std::printf("%-6s %-8s | %10s %10s | %11s %11s | %11s %12s | %10s %10s | %9s %9s\n",
               "NPar", "NEvents", "GradJIT(ms)", "HessJIT(ms)", "AD_Grad(us)", "FD_Grad(us)",
               "AD_Hess(us)", "FD_Hess(us)", "AD_Fit(ms)", "CPU_Fit(ms)", "GradRelEr", "HessRelEr");
   std::printf("--------------------------------------------------------------------------------------------------------------------------------------\n");
   int tag = 100;
   for (int nGauss : {2, 6, 10, 20}) {
      auto r = runRooMixtureBench(nGauss, 10000, tag++);
      std::printf("%-6d %-8d | %10.2f %10.2f | %11.2f %11.2f | %11.2f %12.1f | %10.2f %10.2f | %9.2e %9.2e\n",
                  r.nParams, r.nEvents, r.gradJitMs, r.hessJitMs, r.adGradUs, r.fdGradUs,
                  r.adHessUs, r.fdHessUs, r.adFitMs, r.cpuFitMs, r.maxGradRelErr, r.maxHessRelErr);
      std::fflush(stdout);
   }

   std::printf("\n=== [3] RooFit Unbinned Mixture NLL — Event Scaling (P = 19 params, %s) ===\n", backendName());
   std::printf("%-6s %-8s | %10s %10s | %11s %11s | %11s %12s | %10s %10s\n",
               "NPar", "NEvents", "GradJIT(ms)", "HessJIT(ms)", "AD_Grad(us)", "FD_Grad(us)",
               "AD_Hess(us)", "FD_Hess(us)", "AD_Fit(ms)", "CPU_Fit(ms)");
   std::printf("----------------------------------------------------------------------------------------------------------------\n");
   for (int nEv : {1000, 10000, 50000}) {
      auto r = runRooMixtureBench(6, nEv, tag++);
      std::printf("%-6d %-8d | %10.2f %10.2f | %11.2f %11.2f | %11.2f %12.1f | %10.2f %10.2f\n",
                  r.nParams, r.nEvents, r.gradJitMs, r.hessJitMs, r.adGradUs, r.fdGradUs,
                  r.adHessUs, r.fdHessUs, r.adFitMs, r.cpuFitMs);
      std::fflush(stdout);
   }

   benchSOFIE();
   std::printf("\n");
   return 0;
}
