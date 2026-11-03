constexpr auto modelHeaderSuffix = "_FromONNX_unoptimized.hxx";
constexpr auto modelDataSuffix = "_FromONNX_unoptimized.dat";
#include "test_helpers.h"

#include "gtest/gtest.h"

// Native Enzyme automatic differentiation tests on TMVA::SOFIE ONNX models:
// 1. Reverse-mode input + weight gradients on Linear_16
// 2. Forward-mode, reverse-mode, and batched SIMD 48x48 Hessian on LinearWithSigmoid
TEST(ONNXEnzyme, Linear16InputAndWeightGradients)
{
   SofieReference ref = readReference("Linear_16");
   std::vector<float> input = ref.f32("input0");
   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Linear_16", input);
   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);

   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
#include <Math/EnzymeDerivator.h>

float ez_lin16(TMVA_SOFIE_Linear_16::Session const &session, float const *input)
{
   float out[160]{};
   TMVA_SOFIE_Linear_16::doInfer(session, input, out);
   float sum = 0.0f;
   for (float v : out) sum += v;
   return sum;
}

double ez_lin16_f64(TMVA_SOFIE_Linear_16::Session const &session, float const *input)
{
   float out[160]{};
   TMVA_SOFIE_Linear_16::doInfer(session, input, out);
   double sum = 0.0;
   for (float v : out) sum += v;
   return sum;
}
   )cpp"));

   auto inputInterp = toInterpreter(input, "std::vector<float>", true);
   gInterpreter->ProcessLine("TMVA_SOFIE_Linear_16::Session ez_s16{\"Linear_16_FromONNX.dat\"};");
   gInterpreter->ProcessLine("std::vector<float> ez_g16(1600, 0.0f);");
   gInterpreter->ProcessLine(
      ("ROOT::Math::Enzyme::gradient(ez_lin16, \"input\").execute(ez_s16, " + inputInterp + ", ez_g16.data());")
         .c_str());

   // Verify first 32 and last 32 entries against finite differences
   gInterpreter->ProcessLine((R"cpp(
   std::vector<float> ez_fd16(1600, 0.0f);
   for (int idx : {0, 1, 2, 15, 31, 512, 1024, 1598, 1599}) {
      float *inp = )cpp" + inputInterp +
                              R"cpp(;
      const float orig = inp[idx];
      const float eps = 1e-3f;
      inp[idx] = orig - eps;
      double fm = ez_lin16_f64(ez_s16, inp);
      inp[idx] = orig + eps;
      double fp = ez_lin16_f64(ez_s16, inp);
      inp[idx] = orig;
      ez_fd16[idx] = static_cast<float>((fp - fm) / (2.0 * eps));
   }
   )cpp")
                                .c_str());

   auto *g = reinterpret_cast<float *>(gInterpreter->ProcessLine("ez_g16.data();"));
   auto *fd = reinterpret_cast<float *>(gInterpreter->ProcessLine("ez_fd16.data();"));
   for (int idx : {0, 1, 2, 15, 31, 512, 1024, 1598, 1599}) {
      EXPECT_NEAR(g[idx], fd[idx], 2.5e-3f) << "at index " << idx;
   }
}

TEST(ONNXEnzyme, LinearWithSigmoidBatchedHessian)
{
   constexpr int nInputs = 48;
   SofieReference ref = readReference("LinearWithSigmoid");
   std::vector<float> input = ref.f32("input0");
   ASSERT_EQ(input.size(), nInputs);

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "LinearWithSigmoid", input);
   expectNear(output, ref.f32("output0"), DEFAULT_TOLERANCE);

   ASSERT_TRUE(gInterpreter->Declare(R"cpp(
#include <Math/EnzymeDerivator.h>

float ez_sig(TMVA_SOFIE_LinearWithSigmoid::Session const &session, float const *input)
{
   float out[24]{};
   TMVA_SOFIE_LinearWithSigmoid::doInfer(session, input, out);
   float sum = 0.0f;
   for (float v : out) sum += v;
   return sum;
}
   )cpp"));

   auto inputInterp = toInterpreter(input, "std::vector<float>", true);
   gInterpreter->Declare(R"cpp(
TMVA_SOFIE_LinearWithSigmoid::Session ez_sig_s{"LinearWithSigmoid_FromONNX_unoptimized.dat"};
std::vector<float> ez_sig_g(48, 0.0f);
std::vector<float> ez_sig_h(48 * 48, 0.0f);
   )cpp");

   gInterpreter->ProcessLine(
      ("ROOT::Math::Enzyme::gradient(ez_sig, \"input\").execute(ez_sig_s, " + inputInterp + ", ez_sig_g.data());")
         .c_str());
   gInterpreter->ProcessLine(
      ("ROOT::Math::Enzyme::hessian(ez_sig, \"input[0:47]\").execute(ez_sig_s, " + inputInterp +
       ", ROOT::Math::Enzyme::array_ref<float>(ez_sig_h.data(), 48 * 48));")
         .c_str());

   auto *h = reinterpret_cast<float *>(gInterpreter->ProcessLine("ez_sig_h.data();"));
   float maxAbs = 0.0f;
   for (int i = 0; i < nInputs; ++i) {
      for (int j = 0; j < nInputs; ++j) {
         maxAbs = std::max(maxAbs, std::abs(h[i * nInputs + j]));
         EXPECT_NEAR(h[i * nInputs + j], h[j * nInputs + i], 1e-5f)
            << "Hessian symmetry mismatch at (" << i << "," << j << ")";
      }
   }
   EXPECT_GT(maxAbs, 1e-4f);
}
