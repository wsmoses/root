/// \file EnzymeDerivatorTests.cxx
///
/// \brief Unit tests for native Enzyme automatic differentiation in Cling and
/// ROOT::Math::Enzyme (forward, reverse, higher-order, batched Hessian, cross-transaction
/// JIT specialization, and MathCore/TMath special functions).
///
/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TInterpreter.h>
#include <TInterpreterValue.h>

#include "gtest/gtest.h"

#include <cmath>
#include <vector>

TEST(EnzymeDerivator, BundledHeaderZeroIncludeFlags)
{
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      #include <enzyme/enzyme>
      double ez_square(double x) { return x * x; }
      double ez_dsquare(double x) { return __enzyme_autodiff<double>((void *)ez_square, x); }
   )cpp"));
   auto val = gInterpreter->CreateTemporary();
   ASSERT_TRUE(gInterpreter->Evaluate("ez_dsquare(3.5)", *val));
   EXPECT_DOUBLE_EQ(7.0, val->GetAsDouble());
   delete val;
}

TEST(EnzymeDerivator, CrossTransactionAD)
{
   // Transaction 1: declare primal functions with non-trivial loops and branches.
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      double ez_poly_loop(double x, int n) {
         double acc = 1.0;
         for (int i = 0; i < n; ++i) {
            acc = acc * x + static_cast<double>(i + 1);
         }
         return acc;
      }
   )cpp"));
   // Force compilation of Transaction 1 before differentiating in Transaction 2.
   // For n=4 at x=2: (((1*x + 1)*x + 2)*x + 3)*x + 4 = x^4 + x^3 + 2x^2 + 3x + 4 = 42.
   auto v0 = gInterpreter->CreateTemporary();
   ASSERT_TRUE(gInterpreter->Evaluate("ez_poly_loop(2.0, 4)", *v0));
   EXPECT_DOUBLE_EQ(42.0, v0->GetAsDouble());
   delete v0;

   // Transaction 2: differentiate function from Transaction 1 in both reverse and forward modes.
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      #include <Math/EnzymeDerivator.h>
      double ez_dpoly_rev(double x, int n) {
         return __enzyme_autodiff<double>((void *)ez_poly_loop, enzyme_out, x, enzyme_const, n);
      }
      double ez_dpoly_fwd(double x, int n) {
         return __enzyme_fwddiff<double>((void *)ez_poly_loop, enzyme_dup, x, 1.0, enzyme_const, n);
      }
   )cpp"));
   // For n=4: (((x + 1)*x + 2)*x + 3)*x + 4 = x^4 + x^3 + 2x^2 + 3x + 4
   // Derivative at x=2: 4*8 + 3*4 + 4*2 + 3 = 32 + 12 + 8 + 3 = 55.
   auto v1 = gInterpreter->CreateTemporary();
   ASSERT_TRUE(gInterpreter->Evaluate("ez_dpoly_rev(2.0, 4)", *v1));
   EXPECT_DOUBLE_EQ(55.0, v1->GetAsDouble());
   ASSERT_TRUE(gInterpreter->Evaluate("ez_dpoly_fwd(2.0, 4)", *v1));
   EXPECT_DOUBLE_EQ(55.0, v1->GetAsDouble());
   delete v1;
}

TEST(EnzymeDerivator, HigherOrderDifferentiate)
{
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      #include <Math/EnzymeDerivator.h>
      double ez_sin_exp(double x, double y) { return std::sin(x) * std::exp(2.0 * y); }
   )cpp"));
   auto val = gInterpreter->CreateTemporary();
   // d/dx at (0.7, 0.3) = cos(0.7) * exp(0.6)
   ASSERT_TRUE(gInterpreter->Evaluate("ROOT::Math::Enzyme::differentiate<1>(ez_sin_exp, 0).execute(0.7, 0.3)", *val));
   EXPECT_NEAR(std::cos(0.7) * std::exp(0.6), val->GetAsDouble(), 1e-13);

   // d^2/dy^2 at (0.7, 0.3) = 4 * sin(0.7) * exp(0.6)
   ASSERT_TRUE(gInterpreter->Evaluate("ROOT::Math::Enzyme::differentiate<2>(ez_sin_exp, 1).execute(0.7, 0.3)", *val));
   EXPECT_NEAR(4.0 * std::sin(0.7) * std::exp(0.6), val->GetAsDouble(), 1e-13);

   // d^3/dx^3 at (0.7, 0.3) = -cos(0.7) * exp(0.6)
   ASSERT_TRUE(gInterpreter->Evaluate("ROOT::Math::Enzyme::differentiate<3>(ez_sin_exp, 0).execute(0.7, 0.3)", *val));
   EXPECT_NEAR(-std::cos(0.7) * std::exp(0.6), val->GetAsDouble(), 1e-12);
   delete val;
}

TEST(EnzymeDerivator, ScalarAndArrayGradientAndHessian)
{
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      #include <Math/EnzymeDerivator.h>
      double ez_rosenbrock_scalar(double x, double y, double z) {
         return 100.0 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x) +
                50.0 * (z - y * y) * (z - y * y) + (1.0 - y) * (1.0 - y);
      }
      double ez_rosenbrock_arr(const double *p) {
         double acc = 0.0;
         for (int i = 0; i < 40; ++i) {
            double d1 = p[i + 1] - p[i] * p[i];
            double d2 = 1.0 - p[i];
            acc += 100.0 * d1 * d1 + d2 * d2;
         }
         return acc;
      }
   )cpp"));

   // Scalar 3-var gradient & 3x3 Hessian (exercises sret >=3-double struct return too!)
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      double ez_sc_g[3]{};
      double ez_sc_h[9]{};
      void run_ez_sc() {
         ROOT::Math::Enzyme::gradient(ez_rosenbrock_scalar).execute(1.2, 1.5, 2.1, ez_sc_g);
         ROOT::Math::Enzyme::hessian(ez_rosenbrock_scalar).execute(1.2, 1.5, 2.1, ez_sc_h);
      }
   )cpp"));
   gInterpreter->ProcessLine("run_ez_sc();");
   auto *g3 = reinterpret_cast<double *>(gInterpreter->ProcessLine("ez_sc_g;"));
   auto *h3 = reinterpret_cast<double *>(gInterpreter->ProcessLine("ez_sc_h;"));
   const double x = 1.2, y = 1.5, z = 2.1;
   EXPECT_NEAR(g3[0], -400.0 * x * (y - x * x) - 2.0 * (1.0 - x), 1e-11);
   EXPECT_NEAR(g3[1], 200.0 * (y - x * x) - 200.0 * y * (z - y * y) - 2.0 * (1.0 - y), 1e-11);
   EXPECT_NEAR(g3[2], 100.0 * (z - y * y), 1e-11);
   EXPECT_NEAR(h3[0], -400.0 * (y - 3.0 * x * x) + 2.0, 1e-10);
   EXPECT_NEAR(h3[1], -400.0 * x, 1e-10);
   EXPECT_NEAR(h3[3], -400.0 * x, 1e-10);
   EXPECT_NEAR(h3[8], 100.0, 1e-10);

   // 41-D array gradient & 41x41 batched Hessian (exercises multi-chunk 32 + 8 + 1 batching!)
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      std::vector<double> ez_p41(41, 1.1);
      std::vector<double> ez_g41(41, 0.0);
      std::vector<double> ez_h41(41 * 41, 0.0);
      void run_ez_arr41() {
         ROOT::Math::Enzyme::gradient(ez_rosenbrock_arr, "p").execute(ez_p41.data(), ez_g41.data());
         ROOT::Math::Enzyme::hessian(ez_rosenbrock_arr, "p[0:40]").execute(
            ez_p41.data(), ROOT::Math::Enzyme::array_ref<double>(ez_h41.data(), 41 * 41));
      }
   )cpp"));
   gInterpreter->ProcessLine("run_ez_arr41();");
   auto *g41 = reinterpret_cast<double *>(gInterpreter->ProcessLine("ez_g41.data();"));
   auto *h41 = reinterpret_cast<double *>(gInterpreter->ProcessLine("ez_h41.data();"));
   const double u = 1.1;
   EXPECT_NEAR(g41[0], -400.0 * u * (u - u * u) - 2.0 * (1.0 - u), 1e-10);
   EXPECT_NEAR(g41[40], 200.0 * (u - u * u), 1e-10);
   // Symmetry & tridiagonal structure check across chunk boundary (indices 31, 32, 40)
   for (int idx : {0, 15, 31, 32, 39}) {
      EXPECT_NEAR(h41[idx * 41 + (idx + 1)], -400.0 * u, 1e-10);
      EXPECT_NEAR(h41[(idx + 1) * 41 + idx], -400.0 * u, 1e-10);
   }
   EXPECT_NEAR(h41[40 * 41 + 40], 200.0, 1e-10);
}

TEST(EnzymeDerivator, MathCoreAndTMathSpecialFunctions)
{
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      #include <Math/EnzymeDerivator.h>
      double ez_special(const double *p) {
         return ROOT::Math::normal_cdf(p[0], p[1], p[2]) +
                ROOT::Math::gamma_pdf(p[3], p[4], p[5], 0.0) +
                ROOT::Math::beta_pdf(p[6], p[7], p[8]) +
                TMath::Voigt(p[0] - p[2], p[1], 0.3) +
                TMath::DiLog(0.4 * p[6]) +
                ROOT::Math::Chebyshev4(p[0], 1.0, 0.5, -0.25, 0.1, 0.05);
      }
      std::vector<double> ez_sp_p = {0.8, 1.3, 0.2, 2.1, 2.5, 0.7, 0.35, 2.2, 3.1};
      std::vector<double> ez_sp_g(9, 0.0);
      std::vector<double> ez_sp_h(81, 0.0);
      void run_ez_special() {
         ROOT::Math::Enzyme::gradient(ez_special, "p").execute(ez_sp_p.data(), ez_sp_g.data());
         ROOT::Math::Enzyme::hessian(ez_special, "p[0:8]").execute(
            ez_sp_p.data(), ROOT::Math::Enzyme::array_ref<double>(ez_sp_h.data(), 81));
      }
   )cpp"));
   gInterpreter->ProcessLine("run_ez_special();");
   auto *g = reinterpret_cast<double *>(gInterpreter->ProcessLine("ez_sp_g.data();"));
   auto *h = reinterpret_cast<double *>(gInterpreter->ProcessLine("ez_sp_h.data();"));

   // Verify against central finite differences
   std::vector<double> p = {0.8, 1.3, 0.2, 2.1, 2.5, 0.7, 0.35, 2.2, 3.1};
   auto evalAt = [&](const std::vector<double> &v) {
      std::string cmd = "ez_special((const double[]){";
      for (std::size_t i = 0; i < v.size(); ++i) {
         if (i > 0)
            cmd += ",";
         char buf[64];
         std::snprintf(buf, sizeof(buf), "%.17g", v[i]);
         cmd += buf;
      }
      cmd += "})";
      auto tv = gInterpreter->CreateTemporary();
      gInterpreter->Evaluate(cmd.c_str(), *tv);
      double res = tv->GetAsDouble();
      delete tv;
      return res;
   };

   const double eps = 1e-6;
   for (std::size_t i = 0; i < p.size(); ++i) {
      auto pp = p, pm = p;
      pp[i] += eps;
      pm[i] -= eps;
      double fd = (evalAt(pp) - evalAt(pm)) / (2.0 * eps);
      EXPECT_NEAR(g[i], fd, 1e-6) << "special-function gradient mismatch at index " << i;
   }
   for (std::size_t i = 0; i < p.size(); ++i) {
      for (std::size_t j = i + 1; j < p.size(); ++j) {
         EXPECT_NEAR(h[i * 9 + j], h[j * 9 + i], 1e-9) << "Hessian symmetry at (" << i << "," << j << ")";
      }
   }
}

TEST(EnzymeDerivator, GlobalAndCrossTransactionEvaluators)
{
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      #include <Math/EnzymeDerivator.h>
      double ez_glob_fn(double x, double y) {
         return x * x * y + std::sin(y);
      }
   )cpp"));
   // Transaction 2: declare non-const global evaluator objects at prompt/global scope
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      auto g_ez_diff_y = ROOT::Math::Enzyme::differentiate(ez_glob_fn, 1);
      auto g_ez_grad   = ROOT::Math::Enzyme::gradient(ez_glob_fn);
      auto g_ez_hess   = ROOT::Math::Enzyme::hessian(ez_glob_fn);
   )cpp"));
   // Transaction 3: execute those global evaluators in a separate transaction
   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
      double g_ez_glob_dy = 0.0;
      double g_ez_glob_g[2]{};
      double g_ez_glob_h[4]{};
      void run_ez_glob_evals() {
         g_ez_glob_dy = g_ez_diff_y.execute(3.0, 0.5);
         g_ez_grad.execute(3.0, 0.5, g_ez_glob_g);
         g_ez_hess.execute(3.0, 0.5, g_ez_glob_h);
      }
   )cpp"));
   gInterpreter->ProcessLine("run_ez_glob_evals();");
   double dy = *reinterpret_cast<double *>(gInterpreter->ProcessLine("&g_ez_glob_dy;"));
   auto *g = reinterpret_cast<double *>(gInterpreter->ProcessLine("g_ez_glob_g;"));
   auto *h = reinterpret_cast<double *>(gInterpreter->ProcessLine("g_ez_glob_h;"));

   EXPECT_NEAR(dy, 9.0 + std::cos(0.5), 1e-12);
   EXPECT_NEAR(g[0], 3.0, 1e-12);
   EXPECT_NEAR(g[1], 9.0 + std::cos(0.5), 1e-12);
   EXPECT_NEAR(h[0], 1.0, 1e-12);
   EXPECT_NEAR(h[1], 6.0, 1e-12);
   EXPECT_NEAR(h[2], 6.0, 1e-12);
   EXPECT_NEAR(h[3], -std::sin(0.5), 1e-12);
}
