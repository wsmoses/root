/// \file EnzymeDerivator.h
///
/// \brief Bridge between ROOT/Cling and the Enzyme LLVM automatic differentiation
/// plugin, providing native ROOT::Math::Enzyme APIs, MathCore/TMath derivative
/// rules, and drop-in compatibility for clad::differentiate / gradient / hessian.

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_MATH_ENZYMEDERIVATOR_H
#define ROOT_MATH_ENZYMEDERIVATOR_H

#ifndef __CLING__
#error "This file must not be included by compiled programs."
#endif // __CLING__

#include <RConfigure.h>
#include <enzyme/enzyme>

#include "TMath.h"
#include "Math/ChebyshevPol.h"
#include "Math/Math.h"
#include "Math/PdfFuncMathCore.h"
#include "Math/ProbFuncMathCore.h"
#include "Math/SpecFuncMathCore.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

// Additional Enzyme vector-mode call markers not declared in <enzyme/utils>.
extern int enzyme_width;
extern int enzyme_dupv;
extern int enzyme_dupnoneedv;

// Smooth digamma / digammaf implementations required by Enzyme's built-in
// lgamma and tgamma differentiation rules (and pre-materialized by Cling's
// EnzymeMaterializePass for higher-order differentiation across transactions).
namespace ROOT::Math::Enzyme::Detail {

inline double digamma_positive(double x)
{
   double result = 0.0;
   for (int iter = 0; iter < 10 && x < 8.5; ++iter) {
      result -= 1.0 / x;
      x += 1.0;
   }
   const double inv = 1.0 / x;
   const double inv2 = inv * inv;
   result += std::log(x) - 0.5 * inv;
   // Bernoulli asymptotic series: -sum_{k=1}^{7} B_{2k} / (2k * x^{2k})
   result -= inv2 * ((1.0 / 12.0) +
                     inv2 * ((-1.0 / 120.0) +
                             inv2 * ((1.0 / 252.0) +
                                     inv2 * ((-1.0 / 240.0) +
                                             inv2 * ((1.0 / 132.0) +
                                                     inv2 * ((-691.0 / 32760.0) + inv2 * (1.0 / 12.0)))))));
   return result;
}

} // namespace ROOT::Math::Enzyme::Detail

extern "C" __attribute__((used)) inline double digamma(double x)
{
   constexpr double kPi = 3.141592653589793238462643383279502884;
   if (x < 0.5) {
      const double pix = kPi * x;
      return ::ROOT::Math::Enzyme::Detail::digamma_positive(1.0 - x) - kPi * std::cos(pix) / std::sin(pix);
   }
   return ::ROOT::Math::Enzyme::Detail::digamma_positive(x);
}

extern "C" __attribute__((used)) inline float digammaf(float x)
{
   return static_cast<float>(digamma(static_cast<double>(x)));
}

namespace ROOT::Math::Enzyme::Detail {

// -----------------------------------------------------------------------------
// Differentiable bodies for out-of-line MathCore & TMath special functions
// -----------------------------------------------------------------------------

inline double cheb_T0(double) { return 1.0; }
inline double cheb_T1(double x) { return x; }
inline double cheb_T2(double x) { return 2.0 * x * x - 1.0; }
inline double cheb_T3(double x) { return 4.0 * x * x * x - 3.0 * x; }

inline double cheb_Eval0(double, const double *c) { return c[0]; }
inline double cheb_Eval1(double x, const double *c) { return c[1] * x + c[0]; }
inline double cheb_Eval2(double x, const double *c) { return c[2] * (2.0 * x * x - 1.0) + c[1] * x + c[0]; }
inline double cheb_Eval3(double x, const double *c)
{
   return c[3] * (4.0 * x * x * x - 3.0 * x) + c[2] * (2.0 * x * x - 1.0) + c[1] * x + c[0];
}

inline double landau_pdf_body(double x, double xi, double x0)
{
   constexpr double p1[5] = {0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253};
   constexpr double q1[5] = {1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063};
   constexpr double p2[5] = {0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211};
   constexpr double q2[5] = {1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714};
   constexpr double p3[5] = {0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101};
   constexpr double q3[5] = {1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675};
   constexpr double p4[5] = {0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186};
   constexpr double q4[5] = {1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511};
   constexpr double p5[5] = {1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910};
   constexpr double q5[5] = {1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357};
   constexpr double p6[5] = {1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109};
   constexpr double q6[5] = {1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939};
   constexpr double a1[3] = {0.04166666667, -0.01996527778, 0.02709538966};
   constexpr double a2[2] = {-1.845568670, -4.284640743};

   if (xi <= 0.0)
      return 0.0;
   const double v = (x - x0) / xi;
   double u = 0.0;
   double denlan = 0.0;
   if (v < -5.5) {
      u = std::exp(v + 1.0);
      if (u < 1e-10)
         return 0.0;
      denlan = 0.3989422803 * std::exp(-1.0 / u - 0.5 * (v + 1.0)) * (1.0 + (a1[0] + (a1[1] + a1[2] * u) * u) * u);
   } else if (v < -1.0) {
      u = std::exp(-v - 1.0);
      denlan = std::exp(-u - 0.5 * (v + 1.0)) *
               (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) /
               (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * v);
   } else if (v < 1.0) {
      denlan = (p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4] * v) * v) * v) * v) /
               (q2[0] + (q2[1] + (q2[2] + (q2[3] + q2[4] * v) * v) * v) * v);
   } else if (v < 5.0) {
      denlan = (p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4] * v) * v) * v) * v) /
               (q3[0] + (q3[1] + (q3[2] + (q3[3] + q3[4] * v) * v) * v) * v);
   } else if (v < 12.0) {
      u = 1.0 / v;
      denlan = u * u * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u) /
               (q4[0] + (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * u);
   } else if (v < 50.0) {
      u = 1.0 / v;
      denlan = u * u * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u) /
               (q5[0] + (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * u);
   } else if (v < 300.0) {
      u = 1.0 / v;
      denlan = u * u * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u) /
               (q6[0] + (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * u);
   } else {
      u = 1.0 / (v - v * std::log(v) / (v + 1.0));
      denlan = u * u * (1.0 + (a2[0] + a2[1] * u) * u);
   }
   return denlan / xi;
}

inline double landau_cdf_body(double x, double xi, double x0)
{
   constexpr double p1[5] = {0.2514091491e+0, -0.6250580444e-1, 0.1458381230e-1, -0.2108817737e-2, 0.7411247290e-3};
   constexpr double q1[5] = {1.0, -0.5571175625e-2, 0.6225310236e-1, -0.3137378427e-2, 0.1931496439e-2};
   constexpr double p2[4] = {0.2868328584e+0, 0.3564363231e+0, 0.1523518695e+0, 0.2251304883e-1};
   constexpr double q2[4] = {1.0, 0.6191136137e+0, 0.1720721448e+0, 0.2278594771e-1};
   constexpr double p3[4] = {0.2868329066e+0, 0.3003828436e+0, 0.9950951941e-1, 0.8733827185e-2};
   constexpr double q3[4] = {1.0, 0.4237190502e+0, 0.1095631512e+0, 0.8693851567e-2};
   constexpr double p4[4] = {0.1000351630e+1, 0.4503592498e+1, 0.1085883880e+2, 0.7536052269e+1};
   constexpr double q4[4] = {1.0, 0.5539969678e+1, 0.1933581111e+2, 0.2721321508e+2};
   constexpr double p5[4] = {0.1000006517e+1, 0.4909414111e+2, 0.8505544753e+2, 0.1532153455e+3};
   constexpr double q5[4] = {1.0, 0.5009928881e+2, 0.1399819104e+3, 0.4200002909e+3};
   constexpr double p6[4] = {0.1000000983e+1, 0.1329868456e+3, 0.9162149244e+3, -0.9605054274e+3};
   constexpr double q6[4] = {1.0, 0.1339887843e+3, 0.1055990413e+4, 0.5532224619e+3};
   constexpr double a1[4] = {0.0, -0.4583333333e+0, 0.6675347222e+0, -0.1641741416e+1};
   constexpr double a2[4] = {0.0, 1.0, -0.4227843351e+0, -0.2043403138e+1};

   const double v = (x - x0) / xi;
   double u = 0.0;
   double lan = 0.0;
   if (v < -5.5) {
      u = std::exp(v + 1.0);
      lan = 0.3989422803 * std::exp(-1.0 / u) * std::sqrt(u) * (1.0 + (a1[1] + (a1[2] + a1[3] * u) * u) * u);
   } else if (v < -1.0) {
      u = std::exp(-v - 1.0);
      lan = (std::exp(-u) / std::sqrt(u)) *
            (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) /
            (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * v);
   } else if (v < 1.0) {
      lan = (p2[0] + (p2[1] + (p2[2] + p2[3] * v) * v) * v) / (q2[0] + (q2[1] + (q2[2] + q2[3] * v) * v) * v);
   } else if (v < 4.0) {
      lan = (p3[0] + (p3[1] + (p3[2] + p3[3] * v) * v) * v) / (q3[0] + (q3[1] + (q3[2] + q3[3] * v) * v) * v);
   } else if (v < 12.0) {
      u = 1.0 / v;
      lan = (p4[0] + (p4[1] + (p4[2] + p4[3] * u) * u) * u) / (q4[0] + (q4[1] + (q4[2] + q4[3] * u) * u) * u);
   } else if (v < 50.0) {
      u = 1.0 / v;
      lan = (p5[0] + (p5[1] + (p5[2] + p5[3] * u) * u) * u) / (q5[0] + (q5[1] + (q5[2] + q5[3] * u) * u) * u);
   } else if (v < 300.0) {
      u = 1.0 / v;
      lan = (p6[0] + (p6[1] + (p6[2] + p6[3] * u) * u) * u) / (q6[0] + (q6[1] + (q6[2] + q6[3] * u) * u) * u);
   } else {
      u = 1.0 / (v - v * std::log(v) / (v + 1.0));
      lan = 1.0 - (a2[1] + (a2[2] + a2[3] * u) * u) * u;
   }
   return lan;
}

inline double inc_gamma_series(double a, double x)
{
   constexpr double kMACHEP = 1.11022302462515654042363166809e-16;
   constexpr double kMAXLOG = 709.782712893383973096206318587;
   if (a <= 0.0 || x <= 0.0)
      return 0.0;
   const double ax = a * std::log(x) - x - std::lgamma(a);
   if (ax < -kMAXLOG)
      return 0.0;
   double r = a;
   double c = 1.0;
   double ans = 1.0;
   for (int iter = 0; iter < 2000; ++iter) {
      r += 1.0;
      c *= x / r;
      ans += c;
      if (c <= ans * kMACHEP)
         break;
   }
   return ans * std::exp(ax) / a;
}

inline double inc_gamma_cf(double a, double x)
{
   constexpr double kMACHEP = 1.11022302462515654042363166809e-16;
   constexpr double kMAXLOG = 709.782712893383973096206318587;
   constexpr double kBig = 4.503599627370496e15;
   constexpr double kBiginv = 2.22044604925031308085e-16;
   if (a <= 0.0 || x <= 0.0)
      return 1.0;
   const double ax = a * std::log(x) - x - std::lgamma(a);
   if (ax < -kMAXLOG)
      return 0.0;
   double y = 1.0 - a;
   double z = x + y + 1.0;
   double c = 0.0;
   double pkm2 = 1.0;
   double qkm2 = x;
   double pkm1 = x + 1.0;
   double qkm1 = z * x;
   double ans = pkm1 / qkm1;
   for (int iter = 0; iter < 2000; ++iter) {
      c += 1.0;
      y += 1.0;
      z += 2.0;
      const double yc = y * c;
      const double pk = pkm1 * z - pkm2 * yc;
      const double qk = qkm1 * z - qkm2 * yc;
      double t = 1.0;
      if (qk != 0.0) {
         const double r = pk / qk;
         t = std::abs((ans - r) / r);
         ans = r;
      }
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;
      if (std::abs(pk) > kBig) {
         pkm2 *= kBiginv;
         pkm1 *= kBiginv;
         qkm2 *= kBiginv;
         qkm1 *= kBiginv;
      }
      if (t <= kMACHEP)
         break;
   }
   return ans * std::exp(ax);
}

inline double inc_gamma_body(double a, double x)
{
   if (a <= 0.0 || x <= 0.0)
      return 0.0;
   if (x > 1.0 && x > a)
      return 1.0 - inc_gamma_cf(a, x);
   return inc_gamma_series(a, x);
}

inline double inc_gamma_c_body(double a, double x)
{
   if (a <= 0.0 || x <= 0.0)
      return 1.0;
   if (x < 1.0 || x < a)
      return 1.0 - inc_gamma_series(a, x);
   return inc_gamma_cf(a, x);
}

inline double beta_body(double x, double y)
{
   return std::exp(std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y));
}

inline double inc_beta_cf(double x, double a, double b)
{
   constexpr double eps = 3.0e-14;
   constexpr double fpmin = 1.0e-30;
   const double qab = a + b;
   const double qap = a + 1.0;
   const double qam = a - 1.0;
   double c = 1.0;
   double d = 1.0 - qab * x / qap;
   if (std::abs(d) < fpmin)
      d = fpmin;
   d = 1.0 / d;
   double h = d;
   for (int m = 1; m <= 500; ++m) {
      const double dm = static_cast<double>(m);
      const double m2 = 2.0 * dm;
      double aa = dm * (b - dm) * x / ((qam + m2) * (a + m2));
      d = 1.0 + aa * d;
      if (std::abs(d) < fpmin)
         d = fpmin;
      c = 1.0 + aa / c;
      if (std::abs(c) < fpmin)
         c = fpmin;
      d = 1.0 / d;
      h *= d * c;
      aa = -(a + dm) * (qab + dm) * x / ((a + m2) * (qap + m2));
      d = 1.0 + aa * d;
      if (std::abs(d) < fpmin)
         d = fpmin;
      c = 1.0 + aa / c;
      if (std::abs(c) < fpmin)
         c = fpmin;
      d = 1.0 / d;
      const double del = d * c;
      h *= del;
      if (std::abs(del - 1.0) <= eps)
         break;
   }
   return h;
}

inline double inc_beta_body(double x, double a, double b)
{
   if (x <= 0.0)
      return 0.0;
   if (x >= 1.0)
      return 1.0;
   const double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) +
                              b * std::log(1.0 - x));
   if (x < (a + 1.0) / (a + b + 2.0))
      return bt * inc_beta_cf(x, a, b) / a;
   return 1.0 - bt * inc_beta_cf(1.0 - x, b, a) / b;
}

// ProbFuncMathCore CDF bodies
inline double beta_cdf_body(double x, double a, double b) { return inc_beta_body(x, a, b); }
inline double beta_cdf_c_body(double x, double a, double b) { return inc_beta_body(1.0 - x, b, a); }

inline double breitwigner_cdf_body(double x, double gamma, double x0)
{
   return 0.5 + std::atan(2.0 * (x - x0) / gamma) / 3.141592653589793238462643383279502884;
}
inline double breitwigner_cdf_c_body(double x, double gamma, double x0)
{
   return 0.5 - std::atan(2.0 * (x - x0) / gamma) / 3.141592653589793238462643383279502884;
}

inline double cauchy_cdf_body(double x, double b, double x0)
{
   return 0.5 + std::atan((x - x0) / b) / 3.141592653589793238462643383279502884;
}
inline double cauchy_cdf_c_body(double x, double b, double x0)
{
   return 0.5 - std::atan((x - x0) / b) / 3.141592653589793238462643383279502884;
}

inline double chisquared_cdf_body(double x, double r, double x0) { return inc_gamma_body(0.5 * r, 0.5 * (x - x0)); }
inline double chisquared_cdf_c_body(double x, double r, double x0)
{
   return inc_gamma_c_body(0.5 * r, 0.5 * (x - x0));
}

inline double normal_cdf_body(double x, double sigma, double x0)
{
   constexpr double kSqrt2 = 1.41421356237309515;
   const double z = (x - x0) / (sigma * kSqrt2);
   if (z < -1.0)
      return 0.5 * std::erfc(-z);
   return 0.5 * (1.0 + std::erf(z));
}
inline double normal_cdf_c_body(double x, double sigma, double x0)
{
   constexpr double kSqrt2 = 1.41421356237309515;
   const double z = (x - x0) / (sigma * kSqrt2);
   if (z > 1.0)
      return 0.5 * std::erfc(z);
   return 0.5 * (1.0 - std::erf(z));
}

inline double crystalball_integral_body(double x, double alpha, double n, double sigma, double mean)
{
   if (sigma == 0.0 || alpha == 0.0)
      return 0.0;
   const bool useLog = (std::abs(n - 1.0) < 1.0e-5);
   double z = (x - mean) / sigma;
   if (alpha < 0.0)
      z = -z;
   const double abs_alpha = std::abs(alpha);
   double intgaus = 0.0;
   double intpow = 0.0;
   const double sqrtpiover2 = std::sqrt(3.14159265358979323846 / 2.0);
   const double sqrt2pi = std::sqrt(2.0 * 3.14159265358979323846);
   const double oneoversqrt2 = 1.0 / std::sqrt(2.0);
   if (z <= -abs_alpha) {
      const double r = n / abs_alpha;
      const double A = r * std::exp(-0.5 * alpha * alpha);
      const double B = r - abs_alpha;
      if (!useLog) {
         intpow = A * (1.0 - std::pow(r / (B - z), n - 1.0)) / (n - 1.0);
      } else {
         const double log_B_z = std::log(B - z);
         const double log_r = std::log(r);
         intpow = A * std::pow(r, n - 1.0) * (log_B_z - log_r + 0.5 * (1.0 - n) * (log_B_z * log_B_z - log_r * log_r));
      }
      intgaus = sqrtpiover2 * (1.0 + std::erf(abs_alpha * oneoversqrt2));
   } else {
      intgaus = normal_cdf_c_body(z, 1.0, 0.0) * sqrt2pi;
   }
   return sigma * (intgaus + intpow);
}

inline double crystalball_cdf_body(double x, double alpha, double n, double sigma, double mean)
{
   if (n <= 1.0)
      return std::numeric_limits<double>::quiet_NaN();
   const double abs_alpha = std::abs(alpha);
   const double C = n / abs_alpha * (1.0 / (n - 1.0)) * std::exp(-0.5 * alpha * alpha);
   const double D = std::sqrt(3.14159265358979323846 / 2.0) * (1.0 + std::erf(abs_alpha / std::sqrt(2.0)));
   const double totIntegral = sigma * (C + D);
   const double integral = crystalball_integral_body(x, alpha, n, sigma, mean);
   return (alpha > 0.0) ? (1.0 - integral / totIntegral) : (integral / totIntegral);
}

inline double crystalball_cdf_c_body(double x, double alpha, double n, double sigma, double mean)
{
   if (n <= 1.0)
      return std::numeric_limits<double>::quiet_NaN();
   const double abs_alpha = std::abs(alpha);
   const double C = n / abs_alpha * (1.0 / (n - 1.0)) * std::exp(-0.5 * alpha * alpha);
   const double D = std::sqrt(3.14159265358979323846 / 2.0) * (1.0 + std::erf(abs_alpha / std::sqrt(2.0)));
   const double totIntegral = sigma * (C + D);
   const double integral = crystalball_integral_body(x, alpha, n, sigma, mean);
   return (alpha > 0.0) ? (integral / totIntegral) : (1.0 - integral / totIntegral);
}

inline double exponential_cdf_body(double x, double lambda, double x0)
{
   return ((x - x0) < 0.0) ? 0.0 : -std::expm1(-lambda * (x - x0));
}
inline double exponential_cdf_c_body(double x, double lambda, double x0)
{
   return ((x - x0) < 0.0) ? 1.0 : std::exp(-lambda * (x - x0));
}

inline double fdistribution_cdf_body(double x, double n, double m, double x0)
{
   if (n < 0.0 || m < 0.0)
      return std::numeric_limits<double>::quiet_NaN();
   const double z = n * (x - x0) / (m + n * (x - x0));
   return inc_beta_body(z, 0.5 * n, 0.5 * m);
}
inline double fdistribution_cdf_c_body(double x, double n, double m, double x0)
{
   if (n < 0.0 || m < 0.0)
      return std::numeric_limits<double>::quiet_NaN();
   const double z = m / (m + n * (x - x0));
   return inc_beta_body(z, 0.5 * m, 0.5 * n);
}

inline double gamma_cdf_body(double x, double alpha, double theta, double x0)
{
   return inc_gamma_body(alpha, (x - x0) / theta);
}
inline double gamma_cdf_c_body(double x, double alpha, double theta, double x0)
{
   return inc_gamma_c_body(alpha, (x - x0) / theta);
}

inline double lognormal_cdf_body(double x, double m, double s, double x0)
{
   constexpr double kSqrt2 = 1.41421356237309515;
   const double z = (std::log(x - x0) - m) / (s * kSqrt2);
   if (z < -1.0)
      return 0.5 * std::erfc(-z);
   return 0.5 * (1.0 + std::erf(z));
}
inline double lognormal_cdf_c_body(double x, double m, double s, double x0)
{
   constexpr double kSqrt2 = 1.41421356237309515;
   const double z = (std::log(x - x0) - m) / (s * kSqrt2);
   if (z > 1.0)
      return 0.5 * std::erfc(z);
   return 0.5 * (1.0 - std::erf(z));
}

inline double tdistribution_cdf_body(double x, double r, double x0)
{
   const double p = x - x0;
   const double sign = (p > 0.0) ? 1.0 : -1.0;
   return 0.5 + 0.5 * inc_beta_body(p * p / (r + p * p), 0.5, 0.5 * r) * sign;
}
inline double tdistribution_cdf_c_body(double x, double r, double x0)
{
   const double p = x - x0;
   const double sign = (p > 0.0) ? 1.0 : -1.0;
   return 0.5 - 0.5 * inc_beta_body(p * p / (r + p * p), 0.5, 0.5 * r) * sign;
}

inline double uniform_cdf_body(double x, double a, double b, double x0)
{
   if ((x - x0) < a)
      return 0.0;
   if ((x - x0) >= b)
      return 1.0;
   return ((x - x0) - a) / (b - a);
}
inline double uniform_cdf_c_body(double x, double a, double b, double x0)
{
   if ((x - x0) < a)
      return 1.0;
   if ((x - x0) >= b)
      return 0.0;
   return (b - (x - x0)) / (b - a);
}

inline double poisson_cdf_body(unsigned int n, double mu)
{
   return inc_gamma_c_body(static_cast<double>(n) + 1.0, mu);
}
inline double poisson_cdf_c_body(unsigned int n, double mu)
{
   return inc_gamma_body(static_cast<double>(n) + 1.0, mu);
}

// TMath out-of-line bodies
inline Double_t tmath_gaus_body(Double_t x, Double_t mean, Double_t sigma, Bool_t norm)
{
   if (sigma == 0.0)
      return 1.0e30;
   const Double_t arg = (x - mean) / sigma;
   if (arg < -39.0 || arg > 39.0)
      return 0.0;
   const Double_t res = std::exp(-0.5 * arg * arg);
   return norm ? (res / (2.50662827463100024 * sigma)) : res;
}

inline Double_t tmath_landau_body(Double_t x, Double_t mu, Double_t sigma, Bool_t norm)
{
   if (sigma <= 0.0)
      return 0.0;
   const Double_t den = landau_pdf_body((x - mu) / sigma, 1.0, 0.0);
   return norm ? (den / sigma) : den;
}

inline Double_t tmath_landaui_body(Double_t x) { return landau_cdf_body(x, 1.0, 0.0); }

inline Double_t tmath_breitwigner_body(Double_t x, Double_t mean, Double_t gamma)
{
   const Double_t bw = gamma / ((x - mean) * (x - mean) + 0.25 * gamma * gamma);
   return bw / (2.0 * 3.141592653589793238462643383279502884);
}

inline Double_t tmath_breitwigner_rel_body(Double_t x, Double_t median, Double_t gamma)
{
   const Double_t mm = median * median;
   const Double_t gg = gamma * gamma;
   const Double_t mg = median * gamma;
   const Double_t xxMinusmm = x * x - mm;
   const Double_t y = std::sqrt(mm * (mm + gg));
   const Double_t k = (0.90031631615710606 * mg * y) / std::sqrt(mm + y);
   return k / (xxMinusmm * xxMinusmm + mg * mg);
}

inline Double_t tmath_cauchy_body(Double_t x, Double_t t, Double_t s)
{
   const Double_t temp = (x - t) * (x - t) / (s * s);
   return 1.0 / (s * 3.141592653589793238462643383279502884 * (1.0 + temp));
}

inline Double_t tmath_gamma2_body(Double_t a, Double_t x) { return inc_gamma_body(a, x); }

inline Double_t tmath_gammadist_body(Double_t x, Double_t gamma, Double_t mu, Double_t beta)
{
   if (x < mu || gamma <= 0.0 || beta <= 0.0)
      return 0.0;
   if (x == mu)
      return (gamma == 1.0) ? (1.0 / beta) : 0.0;
   if (gamma == 1.0)
      return std::exp(-(x - mu) / beta) / beta;
   return std::exp((gamma - 1.0) * std::log((x - mu) / beta) - (x - mu) / beta - std::lgamma(gamma)) / beta;
}

inline Double_t tmath_laplace_body(Double_t x, Double_t alpha, Double_t beta)
{
   return std::exp(-std::abs((x - alpha) / beta)) / (2.0 * beta);
}

inline Double_t tmath_laplacei_body(Double_t x, Double_t alpha, Double_t beta)
{
   if (x <= alpha)
      return 0.5 * std::exp(-std::abs((x - alpha) / beta));
   return 1.0 - 0.5 * std::exp(-std::abs((x - alpha) / beta));
}

inline Double_t tmath_lognormal_body(Double_t x, Double_t sigma, Double_t theta, Double_t m)
{
   if (x <= theta || sigma <= 0.0 || m <= 0.0)
      return 0.0;
   const Double_t tmp = (std::log(x - theta) - std::log(m)) / sigma;
   return 1.0 / ((x - theta) * std::abs(sigma) * std::sqrt(2.0 * 3.141592653589793238462643383279502884)) *
          std::exp(-0.5 * tmp * tmp);
}

inline Double_t tmath_poisson_body(Double_t x, Double_t par)
{
   if (par < 0.0)
      return std::numeric_limits<double>::quiet_NaN();
   if (x < 0.0)
      return 0.0;
   if (x == 0.0)
      return std::exp(-par);
   return std::exp(x * std::log(par) - std::lgamma(x + 1.0) - par);
}

inline Double_t tmath_poissoni_body(Double_t x, Double_t par)
{
   const Int_t ix = Int_t(x);
   return tmath_poisson_body(static_cast<Double_t>(ix), par);
}

inline Double_t tmath_prob_body(Double_t chi2, Int_t ndf)
{
   if (ndf <= 0)
      return 0.0;
   if (chi2 <= 0.0)
      return (chi2 < 0.0) ? 0.0 : 1.0;
   return chisquared_cdf_c_body(chi2, static_cast<double>(ndf), 0.0);
}

inline Double_t tmath_freq_body(Double_t x)
{
   return normal_cdf_body(x, 1.0, 0.0);
}

inline Double_t tmath_student_body(Double_t T, Double_t ndf)
{
   if (ndf < 1.0)
      return 0.0;
   const Double_t rh = 0.5 * ndf;
   const Double_t rh1 = rh + 0.5;
   return std::exp(std::lgamma(rh1) - std::lgamma(rh)) /
          (std::sqrt(ndf * 3.141592653589793238462643383279502884) * std::pow(1.0 + T * T / ndf, rh1));
}

inline Double_t tmath_studenti_body(Double_t T, Double_t ndf)
{
   return tdistribution_cdf_body(T, ndf, 0.0);
}

inline Double_t tmath_beta_body(Double_t p, Double_t q) { return beta_body(p, q); }

inline Double_t tmath_betadist_body(Double_t x, Double_t p, Double_t q)
{
   if (x < 0.0 || x > 1.0 || p <= 0.0 || q <= 0.0)
      return 0.0;
   return std::pow(x, p - 1.0) * std::pow(1.0 - x, q - 1.0) / beta_body(p, q);
}

inline Double_t tmath_betadisti_body(Double_t x, Double_t p, Double_t q)
{
   if (x < 0.0 || x > 1.0 || p <= 0.0 || q <= 0.0)
      return 0.0;
   return inc_beta_body(x, p, q);
}

inline Double_t tmath_fdist_body(Double_t F, Double_t N, Double_t M)
{
   if (N < 0.0 || M < 0.0)
      return std::numeric_limits<double>::quiet_NaN();
   if (F < 0.0)
      return 0.0;
   return std::exp(0.5 * N * std::log(N) + 0.5 * M * std::log(M) + std::lgamma(0.5 * (N + M)) -
                   std::lgamma(0.5 * N) - std::lgamma(0.5 * M) + (0.5 * N - 1.0) * std::log(F) -
                   0.5 * (N + M) * std::log(M + N * F));
}

inline Double_t tmath_fdisti_body(Double_t F, Double_t N, Double_t M)
{
   return fdistribution_cdf_body(F, N, M, 0.0);
}

inline Double_t tmath_gamma1_body(Double_t x)
{
   return std::tgamma(x);
}

inline Double_t tmath_lngamma_body(Double_t x)
{
   return std::lgamma(x);
}

inline Double_t tmath_erfcinv_body(Double_t x)
{
   return ::TMath::ErfInverse(1.0 - x);
}

inline Double_t tmath_dilog_body(Double_t x)
{
   constexpr Double_t hf = 0.5;
   constexpr Double_t pi = 3.141592653589793238462643383279502884;
   constexpr Double_t pi2 = pi * pi;
   constexpr Double_t pi3 = pi2 / 3.0;
   constexpr Double_t pi6 = pi2 / 6.0;
   constexpr Double_t pi12 = pi2 / 12.0;
   constexpr Double_t c[20] = {0.42996693560813697,  0.40975987533077106,  -0.01858843665014592, 0.00145751084062268,
                               -0.00014304184442340, 0.00001588415541880,  -0.00000190784959387, 0.00000024195180854,
                               -0.00000003193341274, 0.00000000434545063,  -0.00000000060578480, 0.00000000008612098,
                               -0.00000000001244332, 0.00000000000182256,  -0.00000000000027007, 0.00000000000004042,
                               -0.00000000000000610, 0.00000000000000093,  -0.00000000000000014, 0.00000000000000002};

   if (x == 1.0)
      return pi6;
   if (x == -1.0)
      return -pi12;

   const Double_t t = -x;
   Double_t y = 0.0;
   Double_t s = 0.0;
   Double_t a = 0.0;
   if (t <= -2.0) {
      y = -1.0 / (1.0 + t);
      s = 1.0;
      const Double_t b1 = std::log(-t);
      const Double_t b2 = std::log(1.0 + 1.0 / t);
      a = -pi3 + hf * (b1 * b1 - b2 * b2);
   } else if (t < -1.0) {
      y = -1.0 - t;
      s = -1.0;
      const Double_t lt = std::log(-t);
      a = -pi6 + lt * (lt + std::log(1.0 + 1.0 / t));
   } else if (t <= -0.5) {
      y = -(1.0 + t) / t;
      s = 1.0;
      const Double_t lt = std::log(-t);
      a = -pi6 + lt * (-hf * lt + std::log(1.0 + t));
   } else if (t < 0.0) {
      y = -t / (1.0 + t);
      s = -1.0;
      const Double_t b1 = std::log(1.0 + t);
      a = hf * b1 * b1;
   } else if (t <= 1.0) {
      y = t;
      s = 1.0;
      a = 0.0;
   } else {
      y = 1.0 / t;
      s = -1.0;
      const Double_t b1 = std::log(t);
      a = pi6 + hf * b1 * b1;
   }
   const Double_t h = y + y - 1.0;
   const Double_t alfa = h + h;
   Double_t b1 = 0.0;
   Double_t b2 = 0.0;
   Double_t b0 = 0.0;
   for (Int_t i = 19; i >= 0; --i) {
      b0 = c[i] + alfa * b1 - b2;
      b2 = b1;
      b1 = b0;
   }
   return -(s * (b0 - h * b2) + a);
}

inline Double_t tmath_voigt_body(Double_t xx, Double_t sigma, Double_t lg, Int_t r)
{
   if ((sigma < 0.0 || lg < 0.0) || (sigma == 0.0 && lg == 0.0))
      return 0.0;
   if (sigma == 0.0)
      return lg * 0.159154943 / (xx * xx + lg * lg / 4.0);
   if (lg == 0.0)
      return 0.39894228 / sigma * std::exp(-xx * xx / (2.0 * sigma * sigma));

   const Double_t x = xx / sigma / 1.41421356;
   const Double_t y = lg / 2.0 / sigma / 1.41421356;

   if (r < 2)
      r = 2;
   if (r > 5)
      r = 5;

   const Double_t r0 = 1.51 * std::exp(1.144 * static_cast<Double_t>(r));
   const Double_t r1 = 1.60 * std::exp(0.554 * static_cast<Double_t>(r));
   constexpr Double_t rrtpi = 0.56418958;
   constexpr Double_t y0 = 1.5;
   constexpr Double_t y0py0 = y0 + y0;
   constexpr Double_t y0q = y0 * y0;
   constexpr Double_t c[6] = {1.0117281, -0.75197147, 0.012557727, 0.010022008, -0.00024206814, 0.00000050084806};
   constexpr Double_t s[6] = {1.393237, 0.23115241, -0.15535147, 0.0062183662, 0.000091908299, -0.00000062752596};
   constexpr Double_t t[6] = {0.31424038, 0.94778839, 1.5976826, 2.2795071, 3.0206370, 3.8897249};

   const Double_t yq = y * y;
   const Double_t yrrtpi = y * rrtpi;
   const Double_t xlim0 = r0 - y;
   Double_t xlim1 = r1 - y;
   const Double_t xlim3 = 3.097 * y - 0.45;
   Double_t xlim2 = 6.8 - y;
   const Double_t xlim4 = 18.1 * y + 1.65;
   if (y <= 1e-6) {
      xlim1 = xlim0;
      xlim2 = xlim0;
   }

   const Double_t abx = std::abs(x);
   const Double_t xq = abx * abx;
   Double_t k = 0.0;
   if (abx > xlim0) {
      k = yrrtpi / (xq + yq);
   } else if (abx > xlim1) {
      const Double_t a0 = yq + 0.5;
      const Double_t d0 = a0 * a0;
      const Double_t d2 = yq + yq - 1.0;
      const Double_t d = rrtpi / (d0 + xq * (d2 + xq));
      k = d * y * (a0 + xq);
   } else if (abx > xlim2) {
      const Double_t h0 = 0.5625 + yq * (4.5 + yq * (10.5 + yq * (6.0 + yq)));
      const Double_t h2 = -4.5 + yq * (9.0 + yq * (6.0 + yq * 4.0));
      const Double_t h4 = 10.5 - yq * (6.0 - yq * 6.0);
      const Double_t h6 = -6.0 + yq * 4.0;
      const Double_t e0 = 1.875 + yq * (8.25 + yq * (5.5 + yq));
      const Double_t e2 = 5.25 + yq * (1.0 + yq * 3.0);
      const Double_t e4 = 0.75 * h6;
      const Double_t d = rrtpi / (h0 + xq * (h2 + xq * (h4 + xq * (h6 + xq))));
      k = d * y * (e0 + xq * (e2 + xq * (e4 + xq)));
   } else if (abx < xlim3) {
      const Double_t z0 =
         272.1014 +
         y * (1280.829 +
              y * (2802.870 +
                   y * (3764.966 +
                        y * (3447.629 +
                             y * (2256.981 + y * (1074.409 + y * (369.1989 + y * (88.26741 + y * (13.39880 + y)))))))));
      const Double_t z2 =
         211.678 +
         y * (902.3066 +
              y * (1758.336 +
                   y * (2037.310 + y * (1549.675 + y * (793.4273 + y * (266.2987 + y * (53.59518 + y * 5.0)))))));
      const Double_t z4 =
         78.86585 + y * (308.1852 + y * (497.3014 + y * (479.2576 + y * (269.2916 + y * (80.39278 + y * 10.0)))));
      const Double_t z6 = 22.03523 + y * (55.02933 + y * (92.75679 + y * (53.59518 + y * 10.0)));
      const Double_t z8 = 1.496460 + y * (13.39880 + y * 5.0);
      const Double_t p0 =
         153.5168 +
         y * (549.3954 +
              y * (919.4955 +
                   y * (946.8970 +
                        y * (662.8097 +
                             y * (328.2151 + y * (115.3772 + y * (27.93941 + y * (4.264678 + y * 0.3183291))))))));
      const Double_t p2 =
         -34.16955 +
         y * (-1.322256 +
              y * (124.5975 + y * (189.7730 + y * (139.4665 + y * (56.81652 + y * (12.79458 + y * 1.2733163))))));
      const Double_t p4 = 2.584042 + y * (10.46332 + y * (24.01655 + y * (29.81482 + y * (12.79568 + y * 1.9099744))));
      const Double_t p6 = -0.07272979 + y * (0.9377051 + y * (4.266322 + y * 1.273316));
      const Double_t p8 = 0.0005480304 + y * 0.3183291;
      const Double_t d = 1.7724538 / (z0 + xq * (z2 + xq * (z4 + xq * (z6 + xq * (z8 + xq)))));
      k = d * (p0 + xq * (p2 + xq * (p4 + xq * (p6 + xq * p8))));
   } else {
      const Double_t ypy0 = y + y0;
      const Double_t ypy0q = ypy0 * ypy0;
      Double_t xp[6], xm[6], yp[6], ym[6], mq[6], pq[6], mf[6], pf[6];
      for (int j = 0; j <= 5; ++j) {
         const Double_t dm = x - t[j];
         mq[j] = dm * dm;
         mf[j] = 1.0 / (mq[j] + ypy0q);
         xm[j] = mf[j] * dm;
         ym[j] = mf[j] * ypy0;
         const Double_t dp = x + t[j];
         pq[j] = dp * dp;
         pf[j] = 1.0 / (pq[j] + ypy0q);
         xp[j] = pf[j] * dp;
         yp[j] = pf[j] * ypy0;
      }
      if (abx <= xlim4) {
         for (int j = 0; j <= 5; ++j)
            k = k + c[j] * (ym[j] + yp[j]) - s[j] * (xm[j] - xp[j]);
      } else {
         const Double_t yf = y + y0py0;
         for (int j = 0; j <= 5; ++j) {
            k = k + (c[j] * (mq[j] * mf[j] - y0 * ym[j]) + s[j] * yf * xm[j]) / (mq[j] + y0q) +
                (c[j] * (pq[j] * pf[j] - y0 * yp[j]) - s[j] * yf * xp[j]) / (pq[j] + y0q);
         }
         k = y * k + std::exp(-xq);
      }
   }
   return k / 2.506628 / sigma;
}

// -----------------------------------------------------------------------------
// Enzyme registrations (captured into Cling's cross-transaction Enzyme archive)
// -----------------------------------------------------------------------------

#define ROOT_ENZYME_REGISTER_FN_LIKE(ID, FN_EXPR, MATH_NAME)                                   \
   __attribute__((used)) inline const void *__enzyme_function_like_##ID[2] = {                 \
      reinterpret_cast<const void *>(FN_EXPR), MATH_NAME}

#define ROOT_ENZYME_REGISTER_BODY(ID, TARGET_EXPR, BODY_EXPR)                                  \
   __attribute__((used)) inline const void *__enzyme_provide_body_##ID[2] = {                  \
      reinterpret_cast<const void *>(TARGET_EXPR), reinterpret_cast<const void *>(BODY_EXPR)}

// Standard math library mappings for out-of-line TMath & ROOT::Math wrappers
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_acosh, static_cast<Double_t (*)(Double_t)>(&::TMath::ACosH), "acosh");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_asinh, static_cast<Double_t (*)(Double_t)>(&::TMath::ASinH), "asinh");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_atanh, static_cast<Double_t (*)(Double_t)>(&::TMath::ATanH), "atanh");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_hypot, static_cast<Double_t (*)(Double_t, Double_t)>(&::TMath::Hypot), "hypot");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_log2, static_cast<Double_t (*)(Double_t)>(&::TMath::Log2), "log2");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_erf, static_cast<Double_t (*)(Double_t)>(&::TMath::Erf), "erf");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_erfc, static_cast<Double_t (*)(Double_t)>(&::TMath::Erfc), "erfc");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_erfinv, static_cast<Double_t (*)(Double_t)>(&::TMath::ErfInverse), "erfinv");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_j0, static_cast<Double_t (*)(Double_t)>(&::TMath::BesselJ0), "j0");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_j1, static_cast<Double_t (*)(Double_t)>(&::TMath::BesselJ1), "j1");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_y0, static_cast<Double_t (*)(Double_t)>(&::TMath::BesselY0), "y0");
ROOT_ENZYME_REGISTER_FN_LIKE(tmath_y1, static_cast<Double_t (*)(Double_t)>(&::TMath::BesselY1), "y1");
ROOT_ENZYME_REGISTER_BODY(tmath_gamma1, static_cast<Double_t (*)(Double_t)>(&::TMath::Gamma), &tmath_gamma1_body);
ROOT_ENZYME_REGISTER_BODY(tmath_lngamma, static_cast<Double_t (*)(Double_t)>(&::TMath::LnGamma), &tmath_lngamma_body);
ROOT_ENZYME_REGISTER_BODY(tmath_erfcinv, static_cast<Double_t (*)(Double_t)>(&::TMath::ErfcInverse),
                          &tmath_erfcinv_body);
ROOT_ENZYME_REGISTER_BODY(tmath_dilog, static_cast<Double_t (*)(Double_t)>(&::TMath::DiLog), &tmath_dilog_body);
ROOT_ENZYME_REGISTER_BODY(tmath_voigt,
                          static_cast<Double_t (*)(Double_t, Double_t, Double_t, Int_t)>(&::TMath::Voigt),
                          &tmath_voigt_body);

ROOT_ENZYME_REGISTER_FN_LIKE(rm_erf, static_cast<double (*)(double)>(&::ROOT::Math::erf), "erf");
ROOT_ENZYME_REGISTER_FN_LIKE(rm_erfc, static_cast<double (*)(double)>(&::ROOT::Math::erfc), "erfc");
ROOT_ENZYME_REGISTER_BODY(rm_tgamma, static_cast<double (*)(double)>(&::ROOT::Math::tgamma), &tmath_gamma1_body);
ROOT_ENZYME_REGISTER_BODY(rm_lgamma, static_cast<double (*)(double)>(&::ROOT::Math::lgamma), &tmath_lngamma_body);

// Chebyshev specializations (enables TFormula cheb0..cheb10)
ROOT_ENZYME_REGISTER_BODY(cheb_t0, static_cast<double (*)(double)>(&::ROOT::Math::Chebyshev::T<0>), &cheb_T0);
ROOT_ENZYME_REGISTER_BODY(cheb_t1, static_cast<double (*)(double)>(&::ROOT::Math::Chebyshev::T<1>), &cheb_T1);
ROOT_ENZYME_REGISTER_BODY(cheb_t2, static_cast<double (*)(double)>(&::ROOT::Math::Chebyshev::T<2>), &cheb_T2);
ROOT_ENZYME_REGISTER_BODY(cheb_t3, static_cast<double (*)(double)>(&::ROOT::Math::Chebyshev::T<3>), &cheb_T3);
ROOT_ENZYME_REGISTER_BODY(cheb_e0, static_cast<double (*)(double, const double *)>(&::ROOT::Math::Chebyshev::Eval<0>),
                          &cheb_Eval0);
ROOT_ENZYME_REGISTER_BODY(cheb_e1, static_cast<double (*)(double, const double *)>(&::ROOT::Math::Chebyshev::Eval<1>),
                          &cheb_Eval1);
ROOT_ENZYME_REGISTER_BODY(cheb_e2, static_cast<double (*)(double, const double *)>(&::ROOT::Math::Chebyshev::Eval<2>),
                          &cheb_Eval2);
ROOT_ENZYME_REGISTER_BODY(cheb_e3, static_cast<double (*)(double, const double *)>(&::ROOT::Math::Chebyshev::Eval<3>),
                          &cheb_Eval3);

// MathCore special, PDF, and CDF functions
ROOT_ENZYME_REGISTER_BODY(rm_landau_pdf, static_cast<double (*)(double, double, double)>(&::ROOT::Math::landau_pdf),
                          &landau_pdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_landau_cdf, static_cast<double (*)(double, double, double)>(&::ROOT::Math::landau_cdf),
                          &landau_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_inc_gamma, static_cast<double (*)(double, double)>(&::ROOT::Math::inc_gamma),
                          &inc_gamma_body);
ROOT_ENZYME_REGISTER_BODY(rm_inc_gamma_c, static_cast<double (*)(double, double)>(&::ROOT::Math::inc_gamma_c),
                          &inc_gamma_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_beta, static_cast<double (*)(double, double)>(&::ROOT::Math::beta), &beta_body);
ROOT_ENZYME_REGISTER_BODY(rm_inc_beta, static_cast<double (*)(double, double, double)>(&::ROOT::Math::inc_beta),
                          &inc_beta_body);

ROOT_ENZYME_REGISTER_BODY(rm_beta_cdf, static_cast<double (*)(double, double, double)>(&::ROOT::Math::beta_cdf),
                          &beta_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_beta_cdf_c, static_cast<double (*)(double, double, double)>(&::ROOT::Math::beta_cdf_c),
                          &beta_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_bw_cdf, static_cast<double (*)(double, double, double)>(&::ROOT::Math::breitwigner_cdf),
                          &breitwigner_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_bw_cdf_c,
                          static_cast<double (*)(double, double, double)>(&::ROOT::Math::breitwigner_cdf_c),
                          &breitwigner_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_cauchy_cdf, static_cast<double (*)(double, double, double)>(&::ROOT::Math::cauchy_cdf),
                          &cauchy_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_cauchy_cdf_c,
                          static_cast<double (*)(double, double, double)>(&::ROOT::Math::cauchy_cdf_c),
                          &cauchy_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_chisq_cdf, static_cast<double (*)(double, double, double)>(&::ROOT::Math::chisquared_cdf),
                          &chisquared_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_chisq_cdf_c,
                          static_cast<double (*)(double, double, double)>(&::ROOT::Math::chisquared_cdf_c),
                          &chisquared_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(
   rm_cb_int, static_cast<double (*)(double, double, double, double, double)>(&::ROOT::Math::crystalball_integral),
   &crystalball_integral_body);
ROOT_ENZYME_REGISTER_BODY(
   rm_cb_cdf, static_cast<double (*)(double, double, double, double, double)>(&::ROOT::Math::crystalball_cdf),
   &crystalball_cdf_body);
ROOT_ENZYME_REGISTER_BODY(
   rm_cb_cdf_c, static_cast<double (*)(double, double, double, double, double)>(&::ROOT::Math::crystalball_cdf_c),
   &crystalball_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_exp_cdf, static_cast<double (*)(double, double, double)>(&::ROOT::Math::exponential_cdf),
                          &exponential_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_exp_cdf_c,
                          static_cast<double (*)(double, double, double)>(&::ROOT::Math::exponential_cdf_c),
                          &exponential_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_fdist_cdf,
                          static_cast<double (*)(double, double, double, double)>(&::ROOT::Math::fdistribution_cdf),
                          &fdistribution_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_fdist_cdf_c,
                          static_cast<double (*)(double, double, double, double)>(&::ROOT::Math::fdistribution_cdf_c),
                          &fdistribution_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_gamma_cdf,
                          static_cast<double (*)(double, double, double, double)>(&::ROOT::Math::gamma_cdf),
                          &gamma_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_gamma_cdf_c,
                          static_cast<double (*)(double, double, double, double)>(&::ROOT::Math::gamma_cdf_c),
                          &gamma_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_lognorm_cdf,
                          static_cast<double (*)(double, double, double, double)>(&::ROOT::Math::lognormal_cdf),
                          &lognormal_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_lognorm_cdf_c,
                          static_cast<double (*)(double, double, double, double)>(&::ROOT::Math::lognormal_cdf_c),
                          &lognormal_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_norm_cdf, static_cast<double (*)(double, double, double)>(&::ROOT::Math::normal_cdf),
                          &normal_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_norm_cdf_c, static_cast<double (*)(double, double, double)>(&::ROOT::Math::normal_cdf_c),
                          &normal_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_tdist_cdf,
                          static_cast<double (*)(double, double, double)>(&::ROOT::Math::tdistribution_cdf),
                          &tdistribution_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_tdist_cdf_c,
                          static_cast<double (*)(double, double, double)>(&::ROOT::Math::tdistribution_cdf_c),
                          &tdistribution_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_unif_cdf,
                          static_cast<double (*)(double, double, double, double)>(&::ROOT::Math::uniform_cdf),
                          &uniform_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_unif_cdf_c,
                          static_cast<double (*)(double, double, double, double)>(&::ROOT::Math::uniform_cdf_c),
                          &uniform_cdf_c_body);
ROOT_ENZYME_REGISTER_BODY(rm_pois_cdf, static_cast<double (*)(unsigned int, double)>(&::ROOT::Math::poisson_cdf),
                          &poisson_cdf_body);
ROOT_ENZYME_REGISTER_BODY(rm_pois_cdf_c, static_cast<double (*)(unsigned int, double)>(&::ROOT::Math::poisson_cdf_c),
                          &poisson_cdf_c_body);

// TMath distributions & special functions
ROOT_ENZYME_REGISTER_BODY(tm_gaus, static_cast<Double_t (*)(Double_t, Double_t, Double_t, Bool_t)>(&::TMath::Gaus),
                          &tmath_gaus_body);
ROOT_ENZYME_REGISTER_BODY(tm_landau, static_cast<Double_t (*)(Double_t, Double_t, Double_t, Bool_t)>(&::TMath::Landau),
                          &tmath_landau_body);
ROOT_ENZYME_REGISTER_BODY(tm_landaui, static_cast<Double_t (*)(Double_t)>(&::TMath::LandauI), &tmath_landaui_body);
ROOT_ENZYME_REGISTER_BODY(tm_bw, static_cast<Double_t (*)(Double_t, Double_t, Double_t)>(&::TMath::BreitWigner),
                          &tmath_breitwigner_body);
ROOT_ENZYME_REGISTER_BODY(tm_bw_rel,
                          static_cast<Double_t (*)(Double_t, Double_t, Double_t)>(&::TMath::BreitWignerRelativistic),
                          &tmath_breitwigner_rel_body);
ROOT_ENZYME_REGISTER_BODY(tm_cauchy, static_cast<Double_t (*)(Double_t, Double_t, Double_t)>(&::TMath::CauchyDist),
                          &tmath_cauchy_body);
ROOT_ENZYME_REGISTER_BODY(tm_gamma2, static_cast<Double_t (*)(Double_t, Double_t)>(&::TMath::Gamma),
                          &tmath_gamma2_body);
ROOT_ENZYME_REGISTER_BODY(tm_gammadist,
                          static_cast<Double_t (*)(Double_t, Double_t, Double_t, Double_t)>(&::TMath::GammaDist),
                          &tmath_gammadist_body);
ROOT_ENZYME_REGISTER_BODY(tm_laplace, static_cast<Double_t (*)(Double_t, Double_t, Double_t)>(&::TMath::LaplaceDist),
                          &tmath_laplace_body);
ROOT_ENZYME_REGISTER_BODY(tm_laplacei, static_cast<Double_t (*)(Double_t, Double_t, Double_t)>(&::TMath::LaplaceDistI),
                          &tmath_laplacei_body);
ROOT_ENZYME_REGISTER_BODY(tm_lognormal,
                          static_cast<Double_t (*)(Double_t, Double_t, Double_t, Double_t)>(&::TMath::LogNormal),
                          &tmath_lognormal_body);
ROOT_ENZYME_REGISTER_BODY(tm_poisson, static_cast<Double_t (*)(Double_t, Double_t)>(&::TMath::Poisson),
                          &tmath_poisson_body);
ROOT_ENZYME_REGISTER_BODY(tm_poissoni, static_cast<Double_t (*)(Double_t, Double_t)>(&::TMath::PoissonI),
                          &tmath_poissoni_body);
ROOT_ENZYME_REGISTER_BODY(tm_prob, static_cast<Double_t (*)(Double_t, Int_t)>(&::TMath::Prob), &tmath_prob_body);
ROOT_ENZYME_REGISTER_BODY(tm_freq, static_cast<Double_t (*)(Double_t)>(&::TMath::Freq), &tmath_freq_body);
ROOT_ENZYME_REGISTER_BODY(tm_student, static_cast<Double_t (*)(Double_t, Double_t)>(&::TMath::Student),
                          &tmath_student_body);
ROOT_ENZYME_REGISTER_BODY(tm_studenti, static_cast<Double_t (*)(Double_t, Double_t)>(&::TMath::StudentI),
                          &tmath_studenti_body);
ROOT_ENZYME_REGISTER_BODY(tm_beta, static_cast<Double_t (*)(Double_t, Double_t)>(&::TMath::Beta), &tmath_beta_body);
ROOT_ENZYME_REGISTER_BODY(tm_betadist, static_cast<Double_t (*)(Double_t, Double_t, Double_t)>(&::TMath::BetaDist),
                          &tmath_betadist_body);
ROOT_ENZYME_REGISTER_BODY(tm_betainc,
                          static_cast<Double_t (*)(Double_t, Double_t, Double_t)>(&::TMath::BetaIncomplete),
                          &tmath_betadisti_body);
ROOT_ENZYME_REGISTER_BODY(tm_betadisti, static_cast<Double_t (*)(Double_t, Double_t, Double_t)>(&::TMath::BetaDistI),
                          &tmath_betadisti_body);
ROOT_ENZYME_REGISTER_BODY(tm_fdist, static_cast<Double_t (*)(Double_t, Double_t, Double_t)>(&::TMath::FDist),
                          &tmath_fdist_body);
ROOT_ENZYME_REGISTER_BODY(tm_fdisti, static_cast<Double_t (*)(Double_t, Double_t, Double_t)>(&::TMath::FDistI),
                          &tmath_fdisti_body);

#undef ROOT_ENZYME_REGISTER_FN_LIKE
#undef ROOT_ENZYME_REGISTER_BODY

// -----------------------------------------------------------------------------
// Thunk & specialization helpers for arbitrary-order forward/reverse AD
// -----------------------------------------------------------------------------

template <typename T>
using ThunkParam_t = std::conditional_t<std::is_reference_v<T>, std::add_pointer_t<std::remove_reference_t<T>>, T>;

template <typename T>
__attribute__((always_inline)) inline ThunkParam_t<T> toThunkArg(T val)
{
   if constexpr (std::is_reference_v<T>)
      return std::addressof(val);
   else
      return val;
}

template <typename T>
__attribute__((always_inline)) inline T fromThunkArg(ThunkParam_t<T> val)
{
   if constexpr (std::is_reference_v<T>)
      return *val;
   else
      return val;
}

template <typename Ret, typename... Args>
inline Ret EnzymeFnThunk(Ret (*fn)(Args...), ThunkParam_t<Args>... args)
{
   return fn(fromThunkArg<Args>(args)...);
}

template <typename Arg>
__attribute__((always_inline)) inline auto makeFwdArgTuple(ThunkParam_t<Arg> val, bool isTarget)
{
   if constexpr (std::is_floating_point_v<Arg>) {
      return std::make_tuple(enzyme_dup, val, isTarget ? static_cast<Arg>(1) : static_cast<Arg>(0));
   } else {
      return std::make_tuple(enzyme_const, val);
   }
}

template <typename Ret, typename TargetFn, typename... Args, std::size_t... Is>
__attribute__((always_inline)) inline Ret
callFwdDiffImpl(TargetFn targetFn, Ret (*innerFn)(Args...), int targetIdx, std::index_sequence<Is...>,
                ThunkParam_t<Args>... args)
{
   auto packed = std::tuple_cat(std::make_tuple(reinterpret_cast<void *>(targetFn), enzyme_const, innerFn),
                                makeFwdArgTuple<Args>(args, static_cast<int>(Is) == targetIdx)...);
   return std::apply([](auto... xs) { return __enzyme_fwddiff<Ret>(xs...); }, packed);
}

template <typename Ret, typename TargetFn, typename... Args, std::size_t... Is>
__attribute__((always_inline)) inline Ret
callFwdDiffNthStepImpl(TargetFn targetFn, Ret (*innerFn)(Args...), int targetIdx, std::index_sequence<Is...>,
                       ThunkParam_t<Args>... args)
{
   auto packed = std::tuple_cat(
      std::make_tuple(reinterpret_cast<void *>(targetFn), enzyme_const, innerFn, enzyme_const, targetIdx),
      makeFwdArgTuple<Args>(args, static_cast<int>(Is) == targetIdx)...);
   return std::apply([](auto... xs) { return __enzyme_fwddiff<Ret>(xs...); }, packed);
}

template <unsigned Order, typename Ret, typename... Args>
inline Ret FwdDiffNth(Ret (*fn)(Args...), int targetIdx, ThunkParam_t<Args>... args)
{
   static_assert(Order >= 1, "Differentiation order must be >= 1");
   if constexpr (Order == 1) {
      return callFwdDiffImpl<Ret>(&EnzymeFnThunk<Ret, Args...>, fn, targetIdx, std::index_sequence_for<Args...>{},
                                  args...);
   } else {
      return callFwdDiffNthStepImpl<Ret>(&FwdDiffNth<Order - 1, Ret, Args...>, fn, targetIdx,
                                         std::index_sequence_for<Args...>{}, args...);
   }
}

// Parse "[idx]" or "[lo:hi]" from a Clad-style parameter spec string.
constexpr bool hasBracketSpec(const char *spec)
{
   if (!spec)
      return false;
   for (const char *p = spec; *p != '\0'; ++p) {
      if (*p == '[')
         return true;
   }
   return false;
}

constexpr int parseSpecSingleIndex(const char *spec)
{
   if (!spec)
      return 0;
   const char *lb = nullptr;
   for (const char *p = spec; *p != '\0'; ++p) {
      if (*p == '[') {
         lb = p;
         break;
      }
   }
   if (!lb)
      return 0;
   int idx = 0;
   for (const char *p = lb + 1; *p >= '0' && *p <= '9'; ++p)
      idx = idx * 10 + (*p - '0');
   return idx;
}

constexpr std::size_t parseSpecExtent(const char *spec)
{
   if (!spec)
      return 0;
   const char *lb = nullptr;
   for (const char *p = spec; *p != '\0'; ++p) {
      if (*p == '[') {
         lb = p;
         break;
      }
   }
   if (!lb)
      return 0;
   std::size_t lo = 0;
   const char *p = lb + 1;
   for (; *p >= '0' && *p <= '9'; ++p)
      lo = lo * 10 + static_cast<std::size_t>(*p - '0');
   if (*p != ':')
      return 1;
   ++p;
   std::size_t hi = 0;
   for (; *p >= '0' && *p <= '9'; ++p)
      hi = hi * 10 + static_cast<std::size_t>(*p - '0');
   return (hi >= lo) ? (hi - lo + 1) : 0;
}

inline std::size_t integerSqrt(std::size_t area)
{
   std::size_t r = 0;
   while ((r + 1) * (r + 1) <= area)
      ++r;
   return r;
}

template <typename T, typename = void>
struct HasCreateZeroShadow : std::false_type {};

template <typename T>
struct HasCreateZeroShadow<
   T, std::void_t<decltype(std::declval<const std::remove_cv_t<std::remove_reference_t<T>> &>().CreateZeroShadow())>>
   : std::true_type {};

template <typename T>
inline constexpr bool HasCreateZeroShadow_v = HasCreateZeroShadow<T>::value;

template <typename SessionT, std::size_t B>
struct ShadowSessionBatch {
   alignas(SessionT) unsigned char fStorage[B * sizeof(SessionT)];

   explicit ShadowSessionBatch(const SessionT &proto)
   {
      for (std::size_t v = 0; v < B; ++v)
         ::new (static_cast<void *>(data() + v)) SessionT(proto.CreateZeroShadow());
   }

   ~ShadowSessionBatch()
   {
      for (std::size_t v = B; v > 0; --v)
         (data() + (v - 1))->~SessionT();
   }

   ShadowSessionBatch(const ShadowSessionBatch &) = delete;
   ShadowSessionBatch &operator=(const ShadowSessionBatch &) = delete;

   SessionT *data() { return reinterpret_cast<SessionT *>(fStorage); }
};

template <typename Ret, typename T, bool ConstP>
inline void EnzymeFnThunkGrad1(Ret (*fn)(std::conditional_t<ConstP, const T *, T *>), T *p, T *grad)
{
   using PType = std::conditional_t<ConstP, const T *, T *>;
   __enzyme_autodiff<void>(reinterpret_cast<void *>(&EnzymeFnThunk<Ret, PType>), enzyme_const, fn, enzyme_dup, p, grad);
}

template <typename Ret, typename Arg0, typename T, bool ConstP>
inline void EnzymeFnThunkGrad2(Ret (*fn)(Arg0, std::conditional_t<ConstP, const T *, T *>), ThunkParam_t<Arg0> a0, T *p,
                               T *grad)
{
   using PType = std::conditional_t<ConstP, const T *, T *>;
   __enzyme_autodiff<void>(reinterpret_cast<void *>(&EnzymeFnThunk<Ret, Arg0, PType>), enzyme_const, fn, enzyme_const,
                           a0, enzyme_dup, p, grad);
}

template <typename Ret, typename SessionT, typename T, bool ConstP>
inline void EnzymeFnThunkGrad2Shadow(Ret (*fn)(const SessionT &, std::conditional_t<ConstP, const T *, T *>),
                                     const SessionT *s, SessionT *d_s, T *p, T *grad)
{
   using PType = std::conditional_t<ConstP, const T *, T *>;
   __enzyme_autodiff<void>(reinterpret_cast<void *>(&EnzymeFnThunk<Ret, const SessionT &, PType>), enzyme_const, fn,
                           enzyme_dup, s, d_s, enzyme_dup, p, grad);
}

template <std::size_t B, typename Ret, typename T, bool ConstP>
__attribute__((always_inline)) inline void
runHessianBatch1(Ret (*fn)(std::conditional_t<ConstP, const T *, T *>), T *p_mut, T *seed, T *g_dummy, T *hess_row,
                 std::size_t n, std::size_t base)
{
   for (std::size_t v = 0; v < B; ++v)
      seed[v * n + base + v] = static_cast<T>(1);
   if constexpr (B == 1) {
      __enzyme_fwddiff<void>(reinterpret_cast<void *>(&EnzymeFnThunkGrad1<Ret, T, ConstP>), enzyme_const, fn,
                             enzyme_dup, p_mut, seed, enzyme_dupnoneed, g_dummy, hess_row);
   } else {
      const std::size_t stride = n * sizeof(T);
      __enzyme_fwddiff<void>(reinterpret_cast<void *>(&EnzymeFnThunkGrad1<Ret, T, ConstP>), enzyme_width, B,
                             enzyme_const, fn, enzyme_dupv, stride, p_mut, seed, enzyme_dupnoneedv, stride, g_dummy,
                             hess_row);
   }
   for (std::size_t v = 0; v < B; ++v)
      seed[v * n + base + v] = static_cast<T>(0);
}

template <std::size_t B, typename Ret, typename Arg0, typename T, bool ConstP>
__attribute__((always_inline)) inline void
runHessianBatch2(Ret (*fn)(Arg0, std::conditional_t<ConstP, const T *, T *>), ThunkParam_t<Arg0> a0, T *p_mut, T *seed,
                 T *g_dummy, T *hess_row, std::size_t n, std::size_t base)
{
   for (std::size_t v = 0; v < B; ++v)
      seed[v * n + base + v] = static_cast<T>(1);
   if constexpr (HasCreateZeroShadow_v<Arg0>) {
      using SessionT = std::remove_cv_t<std::remove_reference_t<Arg0>>;
      auto d_s_primal = a0->CreateZeroShadow();
      ShadowSessionBatch<SessionT, B> zero_s(*a0);
      ShadowSessionBatch<SessionT, B> dd_s(*a0);
      if constexpr (B == 1) {
         __enzyme_fwddiff<void>(reinterpret_cast<void *>(&EnzymeFnThunkGrad2Shadow<Ret, SessionT, T, ConstP>),
                                enzyme_const, fn, enzyme_dup, a0, zero_s.data(), enzyme_dup,
                                std::addressof(d_s_primal), dd_s.data(), enzyme_dup, p_mut, seed, enzyme_dupnoneed,
                                g_dummy, hess_row);
      } else {
         const std::size_t stride = n * sizeof(T);
         const std::size_t s_stride = sizeof(SessionT);
         __enzyme_fwddiff<void>(reinterpret_cast<void *>(&EnzymeFnThunkGrad2Shadow<Ret, SessionT, T, ConstP>),
                                enzyme_width, B, enzyme_const, fn, enzyme_dupv, s_stride, a0, zero_s.data(),
                                enzyme_dupv, s_stride, std::addressof(d_s_primal), dd_s.data(), enzyme_dupv, stride,
                                p_mut, seed, enzyme_dupnoneedv, stride, g_dummy, hess_row);
      }
   } else {
      if constexpr (B == 1) {
         __enzyme_fwddiff<void>(reinterpret_cast<void *>(&EnzymeFnThunkGrad2<Ret, Arg0, T, ConstP>), enzyme_const, fn,
                                enzyme_const, a0, enzyme_dup, p_mut, seed, enzyme_dupnoneed, g_dummy, hess_row);
      } else {
         const std::size_t stride = n * sizeof(T);
         __enzyme_fwddiff<void>(reinterpret_cast<void *>(&EnzymeFnThunkGrad2<Ret, Arg0, T, ConstP>), enzyme_width, B,
                                enzyme_const, fn, enzyme_const, a0, enzyme_dupv, stride, p_mut, seed, enzyme_dupnoneedv,
                                stride, g_dummy, hess_row);
      }
   }
   for (std::size_t v = 0; v < B; ++v)
      seed[v * n + base + v] = static_cast<T>(0);
}

template <typename Ret, typename... ScalarArgs, std::size_t... Is>
__attribute__((always_inline)) inline Ret
callScalarFromArray(Ret (*fn)(ScalarArgs...), const double *p, std::index_sequence<Is...>)
{
   return fn(static_cast<ScalarArgs>(p[Is])...);
}

template <typename Ret, typename... ScalarArgs>
inline Ret EnzymeFnThunkScalarArr(Ret (*fn)(ScalarArgs...), const double *p)
{
   return callScalarFromArray(fn, p, std::index_sequence_for<ScalarArgs...>{});
}

template <typename Ret, typename... ScalarArgs>
inline void EnzymeFnThunkScalarArrGrad(Ret (*fn)(ScalarArgs...), double *p, double *grad)
{
   __enzyme_autodiff<void>(reinterpret_cast<void *>(&EnzymeFnThunkScalarArr<Ret, ScalarArgs...>), enzyme_const, fn,
                           enzyme_dup, p, grad);
}

} // namespace ROOT::Math::Enzyme::Detail

namespace ROOT::Math::Enzyme {

/// Lightweight non-owning array view compatible with clad::array_ref<T>.
template <typename T>
class array_ref {
   T *fData = nullptr;
   std::size_t fSize = 0;

public:
   constexpr array_ref() = default;
   constexpr array_ref(T *data, std::size_t size = 0) : fData(data), fSize(size) {}
   template <std::size_t N>
   constexpr array_ref(T (&arr)[N]) : fData(arr), fSize(N)
   {
   }

   constexpr T *data() const { return fData; }
   constexpr std::size_t size() const { return fSize; }
   constexpr T &operator[](std::size_t i) const { return fData[i]; }
   constexpr operator T *() const { return fData; }
};

/// Forward-mode arbitrary-order differentiator functor compatible with clad::differentiate.
template <unsigned Order, typename Ret, typename... Args>
class Differentiator {
   Ret (*fFn)(Args...);
   int fTargetIdx;
   bool fTargetIsArray;

public:
   constexpr Differentiator(Ret (*fn)(Args...), int targetIdx, bool targetIsArray = false)
      : fFn(fn), fTargetIdx(targetIdx), fTargetIsArray(targetIsArray)
   {
   }

   __attribute__((always_inline)) Ret execute(Args... args) const
   {
      if constexpr (Order == 1 && sizeof...(Args) == 1 && (std::is_pointer_v<Args> && ...)) {
         auto tup = std::forward_as_tuple(args...);
         auto p = std::get<0>(tup);
         using PType = std::tuple_element_t<0, std::tuple<Args...>>;
         using ValT = std::remove_const_t<std::remove_pointer_t<PType>>;
         if constexpr (std::is_floating_point_v<ValT>) {
            const std::size_t bufLen = static_cast<std::size_t>(fTargetIdx >= 0 ? fTargetIdx : 0) + 4096;
            ValT stackBuf[4160]{};
            ValT *d_p = (bufLen <= 4160) ? stackBuf : new ValT[bufLen]();
            if (fTargetIdx >= 0)
               d_p[fTargetIdx] = static_cast<ValT>(1);
            Ret res = __enzyme_fwddiff<Ret>(reinterpret_cast<void *>(&Detail::EnzymeFnThunk<Ret, PType>), enzyme_const,
                                            fFn, enzyme_dup, p, d_p);
            if (bufLen > 4160)
               delete[] d_p;
            return res;
         }
      } else if constexpr (Order == 1 && sizeof...(Args) == 2) {
         using Arg0 = std::tuple_element_t<0, std::tuple<Args...>>;
         using Arg1 = std::tuple_element_t<1, std::tuple<Args...>>;
         if constexpr (std::is_pointer_v<Arg1> &&
                       std::is_floating_point_v<std::remove_const_t<std::remove_pointer_t<Arg1>>>) {
            if (fTargetIsArray || !std::is_floating_point_v<Arg0>) {
               auto tup = std::forward_as_tuple(args...);
               decltype(auto) a0 = std::get<0>(tup);
               auto p = std::get<1>(tup);
               using ValT = std::remove_const_t<std::remove_pointer_t<Arg1>>;
               const std::size_t bufLen = static_cast<std::size_t>(fTargetIdx >= 0 ? fTargetIdx : 0) + 4096;
               ValT stackBuf[4160]{};
               ValT *d_p = (bufLen <= 4160) ? stackBuf : new ValT[bufLen]();
               if (fTargetIdx >= 0)
                  d_p[fTargetIdx] = static_cast<ValT>(1);
               Ret res{};
               if constexpr (Detail::HasCreateZeroShadow_v<Arg0>) {
                  auto zero_s = a0.CreateZeroShadow();
                  res = __enzyme_fwddiff<Ret>(reinterpret_cast<void *>(&Detail::EnzymeFnThunk<Ret, Arg0, Arg1>),
                                              enzyme_const, fFn, enzyme_dup, Detail::toThunkArg<Arg0>(a0),
                                              std::addressof(zero_s), enzyme_dup, p, d_p);
               } else {
                  res = __enzyme_fwddiff<Ret>(reinterpret_cast<void *>(&Detail::EnzymeFnThunk<Ret, Arg0, Arg1>),
                                              enzyme_const, fFn, enzyme_const, Detail::toThunkArg<Arg0>(a0), enzyme_dup,
                                              p, d_p);
               }
               if (bufLen > 4160)
                  delete[] d_p;
               return res;
            }
         }
      }
      return Detail::FwdDiffNth<Order, Ret, Args...>(fFn, fTargetIdx, Detail::toThunkArg<Args>(args)...);
   }

   __attribute__((always_inline)) Ret operator()(Args... args) const { return execute(args...); }
};

template <unsigned Order = 1, typename Ret, typename... Args, typename Spec = int,
          std::enable_if_t<std::is_integral_v<Spec>, int> = 0>
__attribute__((always_inline)) constexpr Differentiator<Order, Ret, Args...>
differentiate(Ret (*fn)(Args...), Spec targetIdx = 0)
{
   return Differentiator<Order, Ret, Args...>(fn, static_cast<int>(targetIdx), false);
}

template <unsigned Order = 1, typename Ret, typename... Args>
__attribute__((always_inline)) constexpr Differentiator<Order, Ret, Args...>
differentiate(Ret (*fn)(Args...), const char *spec)
{
   const bool isArr = Detail::hasBracketSpec(spec);
   return Differentiator<Order, Ret, Args...>(fn, Detail::parseSpecSingleIndex(spec), isArr);
}

/// Reverse-mode gradient evaluator compatible with clad::gradient.
template <typename FnPtr>
class GradientEvaluator;

// Case 1: Scalar floating-point arguments Ret(Args...)
template <typename Ret, typename... Args>
class GradientEvaluator<Ret (*)(Args...)> {
   Ret (*fFn)(Args...);

public:
   explicit constexpr GradientEvaluator(Ret (*fn)(Args...)) : fFn(fn) {}

   // Scalar signature: execute(args..., grad_out)
   template <typename OutPtr, typename = std::enable_if_t<(std::is_floating_point_v<Args> && ...)>>
   __attribute__((always_inline)) void execute(Args... args, OutPtr grad_out) const
   {
      constexpr std::size_t K = sizeof...(Args);
      double p[K] = {static_cast<double>(args)...};
      double tmp[K]{};
      __enzyme_autodiff<void>(reinterpret_cast<void *>(&Detail::EnzymeFnThunkScalarArr<Ret, Args...>), enzyme_const,
                              fFn, enzyme_dup, p, tmp);
      for (std::size_t i = 0; i < K; ++i)
         grad_out[i] += tmp[i];
   }
};

// Case 2: Single array pointer Ret(T *) or Ret(const T *)
template <typename Ret, typename T>
class GradientEvaluator<Ret (*)(T *)> {
   Ret (*fFn)(T *);

public:
   explicit constexpr GradientEvaluator(Ret (*fn)(T *)) : fFn(fn) {}

   __attribute__((always_inline)) void execute(T *p, std::remove_const_t<T> *d_p) const
   {
      using ValT = std::remove_const_t<T>;
      using PType = std::conditional_t<std::is_const_v<T>, const ValT *, ValT *>;
      __enzyme_autodiff<void>(reinterpret_cast<void *>(&Detail::EnzymeFnThunk<Ret, PType>), enzyme_const, fFn,
                              enzyme_dup, const_cast<ValT *>(p), d_p);
   }
};

// Case 3: Inactive context + array pointer Ret(Arg0, T *) or Ret(Arg0, const T *)
// Covers TFormula (const double *x, double *p) and TMVA::SOFIE (const Session &s, const float *input).
template <typename Ret, typename Arg0, typename T>
class GradientEvaluator<Ret (*)(Arg0, T *)> {
   Ret (*fFn)(Arg0, T *);

public:
   explicit constexpr GradientEvaluator(Ret (*fn)(Arg0, T *)) : fFn(fn) {}

   __attribute__((always_inline)) void execute(Arg0 a0, T *p, std::remove_const_t<T> *d_p) const
   {
      using ValT = std::remove_const_t<T>;
      using PType = std::conditional_t<std::is_const_v<T>, const ValT *, ValT *>;
      if constexpr (Detail::HasCreateZeroShadow_v<Arg0>) {
         using SessionT = std::remove_cv_t<std::remove_reference_t<Arg0>>;
         auto d_s = a0.CreateZeroShadow();
         __enzyme_autodiff<void>(reinterpret_cast<void *>(&Detail::EnzymeFnThunk<Ret, const SessionT &, PType>),
                                 enzyme_const, fFn, enzyme_dup, Detail::toThunkArg<Arg0>(a0), std::addressof(d_s),
                                 enzyme_dup, const_cast<ValT *>(p), d_p);
      } else {
         __enzyme_autodiff<void>(reinterpret_cast<void *>(&Detail::EnzymeFnThunk<Ret, Arg0, PType>), enzyme_const, fFn,
                                 enzyme_const, Detail::toThunkArg<Arg0>(a0), enzyme_dup, const_cast<ValT *>(p), d_p);
      }
   }
};

template <typename Ret, typename... Args>
__attribute__((always_inline)) constexpr GradientEvaluator<Ret (*)(Args...)>
gradient(Ret (*fn)(Args...), const char * /*spec*/ = nullptr)
{
   return GradientEvaluator<Ret (*)(Args...)>(fn);
}

/// Batched forward-over-reverse Hessian evaluator compatible with clad::hessian.
template <typename FnPtr>
class HessianEvaluator;

// Case 1: Scalar floating-point arguments Ret(Args...)
template <typename Ret, typename... Args>
class HessianEvaluator<Ret (*)(Args...)> {
   Ret (*fFn)(Args...);

public:
   explicit constexpr HessianEvaluator(Ret (*fn)(Args...), std::size_t = 0) : fFn(fn) {}

   template <typename OutPtr, typename = std::enable_if_t<(std::is_floating_point_v<Args> && ...)>>
   __attribute__((always_inline)) void execute(Args... args, OutPtr hess_out) const
   {
      constexpr std::size_t K = sizeof...(Args);
      double p[K] = {static_cast<double>(args)...};
      double g_dummy[K]{};
      double h_buf[K * K]{};
      double seed[K * K]{};
      for (std::size_t v = 0; v < K; ++v)
         seed[v * K + v] = 1.0;
      if constexpr (K == 1) {
         __enzyme_fwddiff<void>(reinterpret_cast<void *>(&Detail::EnzymeFnThunkScalarArrGrad<Ret, Args...>),
                                enzyme_const, fFn, enzyme_dup, p, seed, enzyme_dupnoneed, g_dummy, h_buf);
      } else {
         constexpr std::size_t stride = K * sizeof(double);
         __enzyme_fwddiff<void>(reinterpret_cast<void *>(&Detail::EnzymeFnThunkScalarArrGrad<Ret, Args...>),
                                enzyme_width, K, enzyme_const, fFn, enzyme_dupv, stride, p, seed, enzyme_dupnoneedv,
                                stride, g_dummy, h_buf);
      }
      for (std::size_t i = 0; i < K * K; ++i)
         hess_out[i] += h_buf[i];
   }
};

// Case 2: Single array pointer Ret(T *) or Ret(const T *)
template <typename Ret, typename T>
class HessianEvaluator<Ret (*)(T *)> {
   Ret (*fFn)(T *);
   std::size_t fDim = 0;

public:
   explicit constexpr HessianEvaluator(Ret (*fn)(T *), std::size_t dim = 0) : fFn(fn), fDim(dim) {}

   __attribute__((always_inline)) void execute(T *p, array_ref<std::remove_const_t<T>> hess_out) const
   {
      using ValT = std::remove_const_t<T>;
      const std::size_t n = (hess_out.size() > 0) ? Detail::integerSqrt(hess_out.size()) : fDim;
      if (n == 0)
         return;
      ValT *p_mut = const_cast<ValT *>(p);
      ValT *h_ptr = hess_out.data();
      constexpr bool kConst = std::is_const_v<T>;
      constexpr std::size_t kStackMax = 128;
      ValT g_stack[kStackMax]{};
      ValT seed_stack[32 * kStackMax]{};
      ValT *g_dummy = (n <= kStackMax) ? g_stack : new ValT[n]();
      ValT *seed = (n <= kStackMax) ? seed_stack : new ValT[32 * n]();

      std::size_t base = 0;
      while (base + 32 <= n) {
         Detail::runHessianBatch1<32, Ret, ValT, kConst>(fFn, p_mut, seed, g_dummy, h_ptr + base * n, n, base);
         base += 32;
      }
      if (base + 16 <= n) {
         Detail::runHessianBatch1<16, Ret, ValT, kConst>(fFn, p_mut, seed, g_dummy, h_ptr + base * n, n, base);
         base += 16;
      }
      if (base + 8 <= n) {
         Detail::runHessianBatch1<8, Ret, ValT, kConst>(fFn, p_mut, seed, g_dummy, h_ptr + base * n, n, base);
         base += 8;
      }
      if (base + 4 <= n) {
         Detail::runHessianBatch1<4, Ret, ValT, kConst>(fFn, p_mut, seed, g_dummy, h_ptr + base * n, n, base);
         base += 4;
      }
      while (base < n) {
         Detail::runHessianBatch1<1, Ret, ValT, kConst>(fFn, p_mut, seed, g_dummy, h_ptr + base * n, n, base);
         ++base;
      }

      if (n > kStackMax) {
         delete[] seed;
         delete[] g_dummy;
      }
   }
};

// Case 3: Inactive context + array pointer Ret(Arg0, T *) or Ret(Arg0, const T *)
template <typename Ret, typename Arg0, typename T>
class HessianEvaluator<Ret (*)(Arg0, T *)> {
   Ret (*fFn)(Arg0, T *);
   std::size_t fDim = 0;

public:
   explicit constexpr HessianEvaluator(Ret (*fn)(Arg0, T *), std::size_t dim = 0) : fFn(fn), fDim(dim) {}

   __attribute__((always_inline)) void execute(Arg0 a0, T *p, array_ref<std::remove_const_t<T>> hess_out) const
   {
      using ValT = std::remove_const_t<T>;
      const std::size_t n = (hess_out.size() > 0) ? Detail::integerSqrt(hess_out.size()) : fDim;
      if (n == 0)
         return;
      ValT *p_mut = const_cast<ValT *>(p);
      ValT *h_ptr = hess_out.data();
      auto thunkA0 = Detail::toThunkArg<Arg0>(a0);
      constexpr bool kConst = std::is_const_v<T>;
      constexpr std::size_t kStackMax = 128;
      ValT g_stack[kStackMax]{};
      ValT seed_stack[32 * kStackMax]{};
      ValT *g_dummy = (n <= kStackMax) ? g_stack : new ValT[n]();
      ValT *seed = (n <= kStackMax) ? seed_stack : new ValT[32 * n]();

      std::size_t base = 0;
      while (base + 32 <= n) {
         Detail::runHessianBatch2<32, Ret, Arg0, ValT, kConst>(fFn, thunkA0, p_mut, seed, g_dummy, h_ptr + base * n, n,
                                                               base);
         base += 32;
      }
      if (base + 16 <= n) {
         Detail::runHessianBatch2<16, Ret, Arg0, ValT, kConst>(fFn, thunkA0, p_mut, seed, g_dummy, h_ptr + base * n, n,
                                                               base);
         base += 16;
      }
      if (base + 8 <= n) {
         Detail::runHessianBatch2<8, Ret, Arg0, ValT, kConst>(fFn, thunkA0, p_mut, seed, g_dummy, h_ptr + base * n, n,
                                                              base);
         base += 8;
      }
      if (base + 4 <= n) {
         Detail::runHessianBatch2<4, Ret, Arg0, ValT, kConst>(fFn, thunkA0, p_mut, seed, g_dummy, h_ptr + base * n, n,
                                                              base);
         base += 4;
      }
      while (base < n) {
         Detail::runHessianBatch2<1, Ret, Arg0, ValT, kConst>(fFn, thunkA0, p_mut, seed, g_dummy, h_ptr + base * n, n,
                                                              base);
         ++base;
      }

      if (n > kStackMax) {
         delete[] seed;
         delete[] g_dummy;
      }
   }
};

template <typename Ret, typename... Args>
__attribute__((always_inline)) constexpr HessianEvaluator<Ret (*)(Args...)>
hessian(Ret (*fn)(Args...), const char *spec = nullptr)
{
   return HessianEvaluator<Ret (*)(Args...)>(fn, Detail::parseSpecExtent(spec));
}

} // namespace ROOT::Math::Enzyme

// Provide drop-in clad:: compatibility when ROOT is built with Enzyme instead of Clad.
#ifndef R__HAS_CLAD
namespace clad {
using ROOT::Math::Enzyme::array_ref;
using ROOT::Math::Enzyme::differentiate;
using ROOT::Math::Enzyme::gradient;
using ROOT::Math::Enzyme::hessian;
} // namespace clad
#endif // !R__HAS_CLAD

#endif // ROOT_MATH_ENZYMEDERIVATOR_H
