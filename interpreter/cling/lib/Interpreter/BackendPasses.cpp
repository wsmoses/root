//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "BackendPasses.h"

#include "IncrementalJIT.h"

#include "cling/Utils/Platform.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm-c/Core.h"

//#include "clang/Basic/LangOptions.h"
//#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/CodeGenOptions.h"

#include <optional>

extern "C" __attribute__((weak)) LLVMValueRef (*CustomAllocator)(
    LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef, uint8_t,
    LLVMValueRef *);

#ifdef CLING_WITH_ADAPTIVECPP
llvm::PassPluginLibraryInfo getAdaptiveCppPluginInfo();
#endif

using namespace cling;
using namespace clang;
using namespace llvm;

namespace {
  class UniqueInitFunctionNamePass
      : public PassInfoMixin<UniqueInitFunctionNamePass> {
    // append a suffix to a symbol to make it unique
    // the suffix is "_cling_module_<module number>"
    llvm::SmallString<128> add_module_suffix(const StringRef OriginalName,
                                             const StringRef ModuleName) {
      llvm::SmallString<128> NewFunctionName;
      NewFunctionName.append(ModuleName);
      NewFunctionName.append("_");

      for (size_t i = 0; i < NewFunctionName.size(); ++i) {
        // Replace everything that is not [a-zA-Z0-9._] with a _. This set
        // happens to be the set of C preprocessing numbers.
        if (!isPreprocessingNumberBody(NewFunctionName[i]))
          NewFunctionName[i] = '_';
      }

      return NewFunctionName;
    }

    // make static initialization function names (__cxx_global_var_init) unique
    bool runOnFunction(Function& F, const StringRef ModuleName) {
      if (F.hasName() && F.getName().starts_with("__cxx_global_var_init")) {
        F.setName(add_module_suffix(F.getName(), ModuleName));
        return true;
      }

      return false;
    }

  public:
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      const StringRef ModuleName = M.getName();
      for (auto&& F : M)
        changed |= runOnFunction(F, ModuleName);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
} // namespace

namespace {
  class WorkAroundConstructorPriorityBugPass
      : public PassInfoMixin<WorkAroundConstructorPriorityBugPass> {
  public:
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      llvm::GlobalVariable* GlobalCtors = M.getNamedGlobal("llvm.global_ctors");
      if (!GlobalCtors)
        return PreservedAnalyses::all();

      auto* OldCtors = llvm::dyn_cast_or_null<llvm::ConstantArray>(
          GlobalCtors->getInitializer());
      if (!OldCtors)
        return PreservedAnalyses::all();

      // LLVM had a bug where constructors with the same priority would not be
      // stably sorted. This has been fixed upstream by
      // https://github.com/llvm/llvm-project/pull/95532, but to avoid relying
      // on a backport this pass works around the issue: The idea is that we
      // lower the default priority of concerned constructors to make them sort
      // correctly.
      static constexpr uint64_t DefaultPriority = 65535;

      unsigned NumCtors = OldCtors->getNumOperands();
      const uint64_t NewDefaultPriorityStart = DefaultPriority - NumCtors;
      uint64_t NewDefaultPriority = NewDefaultPriorityStart;

      llvm::SmallVector<Constant*> NewCtors;
      for (unsigned I = 0; I < NumCtors; I++) {
        auto* Ctor =
            llvm::dyn_cast<llvm::ConstantStruct>(OldCtors->getOperand(I));
        auto* PriorityC = llvm::cast<llvm::ConstantInt>(Ctor->getOperand(0));
        uint64_t Priority = PriorityC->getZExtValue();
        if (Priority >= NewDefaultPriorityStart && Priority < DefaultPriority) {
          llvm::errs() << "Found priority " << Priority
                       << ", not changing anything\n";
          return PreservedAnalyses::all();
        }

        if (Priority == DefaultPriority) {
          Priority = NewDefaultPriority;
          NewDefaultPriority++;
        }

        llvm::SmallVector<Constant*> NewCtorArgs;
        NewCtorArgs.push_back(
            llvm::ConstantInt::get(PriorityC->getIntegerType(), Priority));
        // Copy the function and data Constant, if present.
        NewCtorArgs.push_back(Ctor->getOperand(1));
        if (Ctor->getNumOperands() >= 3) {
          NewCtorArgs.push_back(Ctor->getOperand(2));
        }

        NewCtors.push_back(
            llvm::ConstantStruct::get(Ctor->getType(), NewCtorArgs));
      }

      GlobalCtors->setInitializer(
          llvm::ConstantArray::get(OldCtors->getType(), NewCtors));

      return PreservedAnalyses::none();
    }
  };
} // namespace

namespace {
  class KeepLocalGVPass : public PassInfoMixin<KeepLocalGVPass> {
    bool runOnGlobal(GlobalValue& GV) {
      if (GV.isDeclaration())
        return false; // no change.

      // GV is a definition.

      // It doesn't make sense to keep unnamed constants, we wouldn't know how
      // to reference them anyway.
      if (!GV.hasName())
        return false;

      if (GV.getName().starts_with(".str"))
        return false;

      llvm::GlobalValue::LinkageTypes LT = GV.getLinkage();
      if (!GV.isDiscardableIfUnused(LT))
        return false;

      if (LT == llvm::GlobalValue::InternalLinkage) {
        // We want to keep this GlobalValue around, but have to tell the JIT
        // linker that it should not error on duplicate symbols.
        // FIXME: Ideally the frontend would never emit duplicate symbols and
        // we could just use the old version of saying:
        // GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
        GV.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
        return true; // a change!
      }
      return false;
    }

  public:
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      for (auto &&F: M)
        changed |= runOnGlobal(F);
      for (auto &&G: M.globals())
        changed |= runOnGlobal(G);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

namespace {
  class PreventLocalOptPass : public PassInfoMixin<PreventLocalOptPass> {
    bool runOnGlobal(GlobalValue& GV) {
      bool changed = false;

      // Prevent any optimization that tries to take advantage of the actual
      // definition being "local" because we have no influence on the memory
      // layout of sections and how "close" they are.

      if (GV.hasLocalLinkage()) {
        if (GV.isDeclaration()) {
          // For declarations with no definition, we can simply adjust the
          // linkage.
          GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
          changed = true;
        } else {
          // FIXME: Not clear what would be the right linkage. We also cannot
          // continue because "GlobalValue with local linkage [...] must be
          // dso_local!"
          return false;
        }
      }

      if (!GV.hasDefaultVisibility()) {
        GV.setVisibility(llvm::GlobalValue::DefaultVisibility);
        changed = true;
      }

      // Set DSO locality last because setLinkage() and setVisibility() check
      // isImplicitDSOLocal() and then might call setDSOLocal(true).
      if (GV.isDSOLocal()) {
        GV.setDSOLocal(false);
        changed = true;
      }

      return changed;
    }

  public:
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      for (auto &&F: M)
        changed |= runOnGlobal(F);
      for (auto &&G: M.globals())
        changed |= runOnGlobal(G);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

namespace {
  class WeakTypeinfoVTablePass : public PassInfoMixin<WeakTypeinfoVTablePass> {
    bool runOnGlobalVariable(GlobalVariable& GV) {
      // Only need to consider symbols with external linkage because only
      // these could be reported as duplicate.
      if (GV.getLinkage() != llvm::GlobalValue::ExternalLinkage)
        return false;

      if (GV.getName().starts_with("_ZT")) {
        // Currently, if Cling sees the "key function" of a virtual class, it
        // emits typeinfo and vtable variables in every transaction llvm::Module
        // that reference them. Turn them into weak linkage to avoid duplicate
        // symbol errors from the JIT linker.
        // FIXME: This is a hack, we should teach the frontend to emit these
        // only once, or mark all duplicates as available_externally (if that
        // improves performance due to optimizations).
        GV.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
        return true; // a change!
      }

      return false;
    }

  public:
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      for (auto &&GV : M.globals())
        changed |= runOnGlobalVariable(GV);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

namespace {

  // Add a suffix to the CUDA module ctor/dtor, CUDA specific functions and
  // variables to generate a unique name. This is necessary for lazy
  // compilation. Without suffix, cling cannot distinguish ctor/dtor, register
  // function and and ptx code string of subsequent modules.
  class UniqueCUDAStructorName : public PassInfoMixin<UniqueCUDAStructorName> {
    // append a suffix to a symbol to make it unique
    // the suffix is "_cling_module_<module number>"
    llvm::SmallString<128> add_module_suffix(const StringRef SymbolName,
                                             const StringRef ModuleName) {
      llvm::SmallString<128> NewFunctionName;
      NewFunctionName.append(SymbolName);
      NewFunctionName.append("_");
      NewFunctionName.append(ModuleName);

      for (size_t i = 0; i < NewFunctionName.size(); ++i) {
        // Replace everything that is not [a-zA-Z0-9._] with a _. This set
        // happens to be the set of C preprocessing numbers.
        if (!isPreprocessingNumberBody(NewFunctionName[i]))
          NewFunctionName[i] = '_';
      }

      return NewFunctionName;
    }

    // make CUDA specific variables unique
    bool runOnGlobal(GlobalValue& GV, const StringRef ModuleName) {
      if (GV.isDeclaration())
        return false; // no change.

      if (!GV.hasName())
        return false;

      if (GV.getName() == "__cuda_fatbin_wrapper" ||
          GV.getName() == "__cuda_gpubin_handle") {
        GV.setName(add_module_suffix(GV.getName(), ModuleName));
        return true;
      }

      return false;
    }

    // make CUDA specific functions unique
    bool runOnFunction(Function& F, const StringRef ModuleName) {
      if (F.hasName() && (F.getName() == "__cuda_module_ctor" ||
                          F.getName() == "__cuda_module_dtor" ||
                          F.getName() == "__cuda_register_globals")) {
        F.setName(add_module_suffix(F.getName(), ModuleName));
        return true;
      }

      return false;
    }

  public:
    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      const StringRef ModuleName = M.getName();
      for (auto&& F : M)
        changed |= runOnFunction(F, ModuleName);
      for (auto&& G : M.globals())
        changed |= runOnGlobal(G, ModuleName);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
} // namespace

namespace {

  // Replace definitions of weak symbols for which symbols already exist by
  // declarations. This reduces the amount of emitted symbols.
  class ReuseExistingWeakSymbols
      : public PassInfoMixin<ReuseExistingWeakSymbols> {
    cling::IncrementalJIT &m_JIT;
    bool m_KeepAsAvailableExternally;

    bool shouldRemoveGlobalDefinition(GlobalValue& GV) {
      // Existing *weak* symbols can be re-used thanks to ODR.
      llvm::GlobalValue::LinkageTypes LT = GV.getLinkage();
      if (!GV.isDiscardableIfUnused(LT) || !GV.isWeakForLinker(LT))
        return false;

      // Find the symbol as existing, previously compiled symbol in the JIT...
      if (m_JIT.doesSymbolAlreadyExist(GV.getName()))
        return true;

      // ...or in shared libraries (without auto-loading).
      std::string Name = GV.getName().str();
#if !defined(_WIN32)
        return llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(Name);
#else
        return platform::DLSym(Name);
#endif
    }

    bool runOnVar(GlobalVariable& GV) {
#if !defined(_WIN32)
      // Heuristically, Windows cannot handle cross-library variables; they
      // must be library-local.

      if (GV.isDeclaration())
        return false; // no change.
      if (shouldRemoveGlobalDefinition(GV)) {
        if (m_KeepAsAvailableExternally && GV.isConstant() &&
            GV.hasInitializer() && !GV.getName().starts_with("_ZT")) {
          GV.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);
          GV.setVisibility(llvm::GlobalValue::DefaultVisibility);
          GV.setDSOLocal(false);
          return true;
        }
        GV.setInitializer(nullptr); // make this a declaration
        return true; // a change!
      }
#endif
      return false; // no change.
    }

    bool runOnFunc(Function& Func) {
      if (Func.isDeclaration())
        return false; // no change.
#ifndef _WIN32
      // MSVC's stdlib gets symbol issues; i.e. apparently: JIT all or none.
      if (Func.getInstructionCount() < 50) {
        // This is a small function. Keep its definition to retain it for
        // inlining: the cost for JITting it is small, and the likelihood
        // that the call will be inlined is high.
        return false;
      }
#endif
      if (shouldRemoveGlobalDefinition(Func)) {
        if (m_KeepAsAvailableExternally) {
          Func.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);
          Func.setVisibility(llvm::GlobalValue::DefaultVisibility);
          Func.setDSOLocal(false);
          return true;
        }
        Func.deleteBody(); // make this a declaration
        return true; // a change!
      }
      return false; // no change.
    }

  public:
    ReuseExistingWeakSymbols(IncrementalJIT& JIT,
                             bool KeepAsAvailableExternally = false)
        : m_JIT(JIT), m_KeepAsAvailableExternally(KeepAsAvailableExternally) {}

    PreservedAnalyses run(llvm::Module& M, ModuleAnalysisManager& AM) {
      bool changed = false;
      // FIXME: use SymbolLookupSet, rather than looking up symbol by symbol.
      for (auto &&F: M)
        changed |= runOnFunc(F);
      for (auto &&G: M.globals())
        changed |= runOnVar(G);
      return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

namespace {
  struct EnzymeModuleInfo {
    bool HasCalls = false;
    bool HasRegistrations = false;
    bool needsPipeline() const { return HasCalls || HasRegistrations; }
  };

  static constexpr StringLiteral kEnzymeFnStrAttrs[] = {
      "enzyme_math",
      "enzyme_inactive",
      "enzyme_shouldrecompute",
      "enzyme_ta_norecur",
      "enzyme_allocator",
      "enzyme_no_escaping_allocation",
  };

  static constexpr StringLiteral kEnzymeCustomMDKinds[] = {
      "enzyme_augment",
      "enzyme_gradient",
      "enzyme_derivative",
      "enzyme_splitderivative",
  };

  static bool isEnzymeCallName(StringRef Name) {
    return Name.contains("__enzyme_autodiff") ||
           Name.contains("__enzyme_fwddiff") ||
           Name.contains("__enzyme_augmentfwd") ||
           Name.contains("__enzyme_reverse") ||
           Name.contains("__enzyme_virtualreverse") ||
           Name.contains("__enzyme_batch");
  }

  static bool hasEnzymeRuleOrAttr(const Function &F) {
    for (StringRef AttrName : kEnzymeFnStrAttrs) {
      if (F.hasFnAttribute(AttrName))
        return true;
    }
    for (StringRef MDName : kEnzymeCustomMDKinds) {
      if (F.getMetadata(MDName))
        return true;
    }
    return false;
  }

  static bool hasValidCustomDerivativeMD(const Function &F, StringRef MDName) {
    MDNode *MD = F.getMetadata(MDName);
    if (!MD || MD->getNumOperands() == 0)
      return false;
    for (const MDOperand &Op : MD->operands()) {
      Metadata *MDOp = Op.get();
      if (!MDOp)
        return false;
      if (const auto *VMD = dyn_cast<ValueAsMetadata>(MDOp)) {
        if (!VMD->getValue())
          return false;
      }
    }
    return true;
  }

  static bool blocksEnzymeBodyMaterialization(const Function &F) {
    if (F.hasFnAttribute("enzyme_math") || F.hasFnAttribute("enzyme_allocator"))
      return true;
    for (StringRef MDName : kEnzymeCustomMDKinds) {
      if (hasValidCustomDerivativeMD(F, MDName))
        return true;
    }
    return false;
  }

  static EnzymeModuleInfo inspectEnzymeModule(const Module &M) {
    EnzymeModuleInfo Info;
    for (const Function &F : M) {
      if (isEnzymeCallName(F.getName()))
        Info.HasCalls = true;
      if (hasEnzymeRuleOrAttr(F))
        Info.HasRegistrations = true;
    }
    for (const GlobalVariable &GV : M.globals()) {
      StringRef Name = GV.getName();
      if (Name.contains("__enzyme_register_") ||
          Name.contains("__enzyme_function_like") ||
          Name.contains("__enzyme_inactivefn") ||
          Name.contains("__enzyme_nofree") ||
          Name.contains("__enzyme_allocation_like") ||
          Name.contains("__enzyme_sparse_accumulate") ||
          Name.contains("__enzyme_provide_body")) {
        Info.HasRegistrations = true;
      } else if (Name == "llvm.global.annotations" && GV.hasInitializer()) {
        if (const auto *CA = dyn_cast<ConstantArray>(GV.getInitializer())) {
          for (const Use &Op : CA->operands()) {
            const auto *CS = dyn_cast<ConstantStruct>(Op.get());
            if (!CS || CS->getNumOperands() < 2)
              continue;
            const Value *StrV = CS->getOperand(1)->stripPointerCasts();
            if (const auto *StrGV = dyn_cast<GlobalVariable>(StrV)) {
              if (StrGV->hasInitializer()) {
                if (const auto *CDA =
                        dyn_cast<ConstantDataArray>(StrGV->getInitializer())) {
                  if (CDA->isString() &&
                      CDA->getAsString().contains("enzyme_")) {
                    Info.HasRegistrations = true;
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }
    return Info;
  }

  static bool shouldSkipArchivingFunction(const Function &F) {
    if (F.isDeclaration() || F.isIntrinsic() || !F.hasName())
      return true;
    StringRef Name = F.getName();
    return Name.starts_with("__cxx_global_var_init") ||
           Name.starts_with("_GLOBAL__sub_I_") ||
           Name == "__clang_call_terminate" ||
           Name.starts_with("__cling_Un1Qu3") ||
           Name.starts_with("_Z41__static_initialization_and_destruction_0") ||
           isEnzymeCallName(Name);
  }

  static void removeGlobalFromUsedLists(Module &M, GlobalValue *Target) {
    for (StringRef UsedName : {"llvm.used", "llvm.compiler.used"}) {
      GlobalVariable *UsedGV = M.getGlobalVariable(UsedName);
      if (!UsedGV || !UsedGV->hasInitializer())
        continue;
      auto *CA = dyn_cast<ConstantArray>(UsedGV->getInitializer());
      if (!CA)
        continue;
      SmallVector<Constant *, 4> Kept;
      bool Found = false;
      for (unsigned I = 0, E = CA->getNumOperands(); I != E; ++I) {
        if (CA->getOperand(I)->stripPointerCasts() == Target)
          Found = true;
        else
          Kept.push_back(CA->getOperand(I));
      }
      if (!Found)
        continue;
      if (Kept.empty()) {
        UsedGV->eraseFromParent();
      } else {
        auto *NewArrTy =
            ArrayType::get(CA->getType()->getElementType(), Kept.size());
        auto *NewInit = ConstantArray::get(NewArrTy, Kept);
        auto *NewUsedGV = new GlobalVariable(
            M, NewArrTy, UsedGV->isConstant(), UsedGV->getLinkage(), NewInit,
            "", UsedGV, UsedGV->getThreadLocalMode());
        NewUsedGV->takeName(UsedGV);
        UsedGV->eraseFromParent();
      }
    }
  }

  static void mapConstantGlobalsForClone(
      const Constant *C, const Function *RootSrcFn, Module &DstM,
      ValueToValueMapTy &VMap, SmallPtrSetImpl<const Constant *> &Visited,
      SmallVectorImpl<std::pair<const Function *, Function *>> &LocalFns,
      bool MaterializingIntoLiveModule) {
    if (!C || !Visited.insert(C).second)
      return;

    if (const auto *BA = dyn_cast<BlockAddress>(C)) {
      if (BA->getFunction() != RootSrcFn) {
        mapConstantGlobalsForClone(BA->getFunction(), RootSrcFn, DstM, VMap,
                                   Visited, LocalFns,
                                   MaterializingIntoLiveModule);
      }
      return;
    }

    if (const auto *Fn = dyn_cast<Function>(C)) {
      if (VMap.count(Fn))
        return;
      if (Fn->hasLocalLinkage() && !Fn->isDeclaration()) {
        Function *LocalDst = Function::Create(
            Fn->getFunctionType(), GlobalValue::InternalLinkage,
            Fn->getAddressSpace(),
            Fn->hasName() ? Fn->getName() : "__cling_enzyme_local_fn", &DstM);
        LocalDst->copyAttributesFrom(Fn);
        VMap[Fn] = LocalDst;
        LocalFns.emplace_back(Fn, LocalDst);
        return;
      }
      Function *DstFn = DstM.getFunction(Fn->getName());
      if (!DstFn) {
        DstFn = Function::Create(Fn->getFunctionType(),
                                 GlobalValue::ExternalLinkage,
                                 Fn->getAddressSpace(), Fn->getName(), &DstM);
        DstFn->copyAttributesFrom(Fn);
        if (!Fn->isIntrinsic()) {
          DstFn->setVisibility(GlobalValue::DefaultVisibility);
          DstFn->setDSOLocal(false);
        }
      }
      VMap[Fn] = DstFn;
      return;
    }

    if (const auto *GVar = dyn_cast<GlobalVariable>(C)) {
      if (VMap.count(GVar))
        return;

      if ((GVar->hasLocalLinkage() || !GVar->hasName()) && GVar->isConstant() &&
          GVar->hasInitializer()) {
        mapConstantGlobalsForClone(GVar->getInitializer(), RootSrcFn, DstM,
                                   VMap, Visited, LocalFns,
                                   MaterializingIntoLiveModule);
        auto *NewGV = new GlobalVariable(
            DstM, GVar->getValueType(), /*isConstant=*/true,
            GlobalValue::PrivateLinkage, nullptr, GVar->getName(), nullptr,
            GVar->getThreadLocalMode(), GVar->getAddressSpace());
        NewGV->copyAttributesFrom(GVar);
        NewGV->setAlignment(GVar->getAlign());
        VMap[GVar] = NewGV;
        NewGV->setInitializer(MapValue(GVar->getInitializer(), VMap));
        return;
      }

      if (!GVar->hasName() || GVar->hasLocalLinkage()) {
        // Rare mutable unnamed/local global: externalize it in its parent
        // module so all transactions share the live JIT storage.
        auto *MutSrcGV = const_cast<GlobalVariable *>(GVar);
        if (!MutSrcGV->hasName()) {
          MutSrcGV->setName("__cling_enzyme_mut_gv_" +
                            MutSrcGV->getParent()->getName().str());
        }
        MutSrcGV->setLinkage(GlobalValue::WeakAnyLinkage);
        MutSrcGV->setVisibility(GlobalValue::DefaultVisibility);
        MutSrcGV->setDSOLocal(false);
      }

      GlobalVariable *DstGV = DstM.getNamedGlobal(GVar->getName());
      if (!DstGV) {
        bool CopyConstInit = GVar->isConstant() && GVar->hasInitializer() &&
                             !GVar->isExternallyInitialized() &&
                             !GVar->getName().starts_with("_ZT");
        if (CopyConstInit) {
          mapConstantGlobalsForClone(GVar->getInitializer(), RootSrcFn, DstM,
                                     VMap, Visited, LocalFns,
                                     MaterializingIntoLiveModule);
          DstGV = new GlobalVariable(
              DstM, GVar->getValueType(), /*isConstant=*/true,
              GlobalValue::AvailableExternallyLinkage, nullptr, GVar->getName(),
              nullptr, GVar->getThreadLocalMode(), GVar->getAddressSpace());
          DstGV->copyAttributesFrom(GVar);
          DstGV->setAlignment(GVar->getAlign());
          DstGV->setVisibility(GlobalValue::DefaultVisibility);
          DstGV->setDSOLocal(false);
          VMap[GVar] = DstGV;
          DstGV->setInitializer(MapValue(GVar->getInitializer(), VMap));
          return;
        }
        DstGV = new GlobalVariable(
            DstM, GVar->getValueType(), GVar->isConstant(),
            GlobalValue::ExternalLinkage, /*Initializer=*/nullptr,
            GVar->getName(), nullptr, GVar->getThreadLocalMode(),
            GVar->getAddressSpace());
        DstGV->copyAttributesFrom(GVar);
        DstGV->setAlignment(GVar->getAlign());
        DstGV->setVisibility(GlobalValue::DefaultVisibility);
        DstGV->setDSOLocal(false);
      }
      VMap[GVar] = DstGV;
      return;
    }

    if (const auto *GA = dyn_cast<GlobalAlias>(C)) {
      if (VMap.count(GA))
        return;
      if (auto *FnTy = dyn_cast<FunctionType>(GA->getValueType())) {
        Function *DstFn = DstM.getFunction(GA->getName());
        if (!DstFn) {
          DstFn = Function::Create(FnTy, GlobalValue::ExternalLinkage,
                                   GA->getAddressSpace(), GA->getName(), &DstM);
          DstFn->setVisibility(GlobalValue::DefaultVisibility);
          DstFn->setDSOLocal(false);
        }
        VMap[GA] = DstFn;
      } else {
        GlobalVariable *DstGV = DstM.getNamedGlobal(GA->getName());
        if (!DstGV) {
          DstGV = new GlobalVariable(DstM, GA->getValueType(), false,
                                     GlobalValue::ExternalLinkage, nullptr,
                                     GA->getName());
          DstGV->setVisibility(GlobalValue::DefaultVisibility);
          DstGV->setDSOLocal(false);
        }
        VMap[GA] = DstGV;
      }
      return;
    }

    for (const Use &Op : C->operands()) {
      if (const auto *OpC = dyn_cast<Constant>(Op.get())) {
        mapConstantGlobalsForClone(OpC, RootSrcFn, DstM, VMap, Visited,
                                   LocalFns, MaterializingIntoLiveModule);
      }
    }
  }

  static Function *cloneFunctionBetweenModules(
      const Function *SrcFn, Module &DstM, Function *DstFn = nullptr,
      StringRef OverrideDstName = "", bool MaterializingIntoLiveModule = false) {
    StringRef TargetName =
        OverrideDstName.empty() ? SrcFn->getName() : OverrideDstName;
    if (!DstFn) {
      DstFn = DstM.getFunction(TargetName);
      if (!DstFn) {
        DstFn = Function::Create(SrcFn->getFunctionType(),
                                 GlobalValue::ExternalLinkage,
                                 SrcFn->getAddressSpace(), TargetName, &DstM);
      }
    }
    if (DstFn->getFunctionType() != SrcFn->getFunctionType())
      return nullptr;

    // Preserve any existing Enzyme metadata on DstFn while replacing its body.
    SmallVector<std::pair<unsigned, MDNode *>, 4> SavedMDs;
    DstFn->getAllMetadata(SavedMDs);
    AttributeSet SavedFnAttrs = DstFn->getAttributes().getFnAttrs();

    if (!DstFn->empty())
      DstFn->deleteBody();

    DstFn->copyAttributesFrom(SrcFn);
    for (const Attribute &Attr : SavedFnAttrs) {
      if (Attr.isStringAttribute() &&
          Attr.getKindAsString().starts_with("enzyme_")) {
        DstFn->addFnAttr(Attr);
      }
    }
    for (const auto &KV : SavedMDs) {
      DstFn->setMetadata(KV.first, KV.second);
    }

    ValueToValueMapTy VMap;
    VMap[SrcFn] = DstFn;

    auto DstArgIt = DstFn->arg_begin();
    for (const Argument &SrcArg : SrcFn->args()) {
      DstArgIt->setName(SrcArg.getName());
      VMap[&SrcArg] = &*DstArgIt++;
    }

    SmallPtrSet<const Constant *, 32> Visited;
    SmallVector<std::pair<const Function *, Function *>, 4> LocalFns;

    if (SrcFn->hasPersonalityFn()) {
      mapConstantGlobalsForClone(SrcFn->getPersonalityFn(), SrcFn, DstM, VMap,
                                 Visited, LocalFns,
                                 MaterializingIntoLiveModule);
    }
    if (SrcFn->hasPrefixData()) {
      mapConstantGlobalsForClone(SrcFn->getPrefixData(), SrcFn, DstM, VMap,
                                 Visited, LocalFns,
                                 MaterializingIntoLiveModule);
    }
    if (SrcFn->hasPrologueData()) {
      mapConstantGlobalsForClone(SrcFn->getPrologueData(), SrcFn, DstM, VMap,
                                 Visited, LocalFns,
                                 MaterializingIntoLiveModule);
    }

    for (const BasicBlock &BB : *SrcFn) {
      for (const Instruction &I : BB) {
        for (const Use &Op : I.operands()) {
          if (const auto *C = dyn_cast<Constant>(Op.get())) {
            mapConstantGlobalsForClone(C, SrcFn, DstM, VMap, Visited, LocalFns,
                                       MaterializingIntoLiveModule);
          }
        }
      }
    }

    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(DstFn, SrcFn, VMap,
                      CloneFunctionChangeType::DifferentModule, Returns);

    // Clone any private/internal helper functions (e.g. local lambdas).
    for (size_t I = 0; I < LocalFns.size(); ++I) {
      const Function *LocalSrc = LocalFns[I].first;
      Function *LocalDst = LocalFns[I].second;
      cloneFunctionBetweenModules(LocalSrc, DstM, LocalDst, "",
                                  MaterializingIntoLiveModule);
      LocalDst->setLinkage(GlobalValue::InternalLinkage);
    }

    return DstFn;
  }

  static void stripSyntheticO0Attrs(Function &F) {
    if (F.hasFnAttribute(Attribute::OptimizeNone))
      F.removeFnAttr(Attribute::OptimizeNone);
    if (F.hasFnAttribute(Attribute::NoInline) &&
        !F.hasFnAttribute("prev_fixup")) {
      F.removeFnAttr(Attribute::NoInline);
    }
  }

  static void stripPreserveNVVMAttrs(Function &F) {
    for (StringRef Attr : {"prev_fixup", "prev_linkage", "prev_always_inline",
                           "prev_no_inline"}) {
      if (F.hasFnAttribute(Attr))
        F.removeFnAttr(Attr);
    }
  }

  static Metadata *remapCustomDerivativeMD(
      Metadata *MD, Module &DstM, Module &SrcM, bool MaterializingIntoLive,
      SmallVectorImpl<Function *> *NewlyMaterialized = nullptr) {
    if (!MD)
      return nullptr;
    if (auto *VMD = dyn_cast<ValueAsMetadata>(MD)) {
      if (!VMD->getValue())
        return nullptr;
      if (const auto *SrcFn =
              dyn_cast<Function>(VMD->getValue()->stripPointerCasts())) {
        Function *DstFn = DstM.getFunction(SrcFn->getName());
        const bool ShouldCloneBody =
            !SrcFn->isDeclaration() &&
            (!MaterializingIntoLive || !DstFn || DstFn->isDeclaration());
        if (ShouldCloneBody) {
          DstFn = cloneFunctionBetweenModules(SrcFn, DstM, DstFn, "",
                                              MaterializingIntoLive);
          if (DstFn) {
            stripPreserveNVVMAttrs(*DstFn);
            DstFn->setLinkage(GlobalValue::ExternalLinkage);
            if (MaterializingIntoLive)
              DstFn->addFnAttr("cling_enzyme_temp_external");
            DstFn->setVisibility(GlobalValue::DefaultVisibility);
            DstFn->setDSOLocal(false);
            if (NewlyMaterialized)
              NewlyMaterialized->push_back(DstFn);
          }
        } else if (DstFn && MaterializingIntoLive &&
                   DstFn->getLinkage() ==
                       GlobalValue::AvailableExternallyLinkage) {
          stripPreserveNVVMAttrs(*DstFn);
          DstFn->setLinkage(GlobalValue::ExternalLinkage);
          DstFn->addFnAttr("cling_enzyme_temp_external");
        }
        if (DstFn)
          return ValueAsMetadata::get(DstFn);
      }
      return MD;
    }
    if (auto *Tuple = dyn_cast<MDTuple>(MD)) {
      SmallVector<Metadata *, 4> Ops;
      Ops.reserve(Tuple->getNumOperands());
      for (const MDOperand &Op : Tuple->operands()) {
        Metadata *RemappedOp = remapCustomDerivativeMD(
            Op.get(), DstM, SrcM, MaterializingIntoLive, NewlyMaterialized);
        if (!RemappedOp)
          return nullptr;
        Ops.push_back(RemappedOp);
      }
      return MDTuple::get(DstM.getContext(), Ops);
    }
    return MD;
  }

  static const Function *extractEvaluatorFunction(const Constant *C) {
    if (!C)
      return nullptr;
    C = dyn_cast<Constant>(C->stripPointerCasts());
    if (!C)
      return nullptr;
    if (const auto *Fn = dyn_cast<Function>(C))
      return Fn;
    if (const auto *CA = dyn_cast<ConstantAggregate>(C)) {
      if (CA->getNumOperands() > 0)
        return extractEvaluatorFunction(CA->getOperand(0));
    }
    return nullptr;
  }

  static ConstantInt *extractEvaluatorTargetIdx(const Constant *C) {
    if (const auto *CA = dyn_cast_or_null<ConstantAggregate>(C)) {
      if (CA->getNumOperands() > 1)
        return dyn_cast<ConstantInt>(CA->getOperand(1));
    }
    return nullptr;
  }

  static bool hasRuntimeStoresToGlobal(const GlobalVariable &GV) {
    SmallVector<const User *, 8> Worklist(GV.users().begin(), GV.users().end());
    SmallPtrSet<const User *, 8> Visited;
    while (!Worklist.empty()) {
      const User *U = Worklist.pop_back_val();
      if (!Visited.insert(U).second)
        continue;
      if (const auto *SI = dyn_cast<StoreInst>(U)) {
        const Function *ParentFn = SI->getFunction();
        if (!ParentFn ||
            (!ParentFn->getName().starts_with("__cxx_global_var_init") &&
             !ParentFn->getName().starts_with("_GLOBAL__sub_I_")))
          return true;
        continue;
      }
      if (const auto *CB = dyn_cast<CallBase>(U)) {
        if (!CB->onlyReadsMemory())
          return true;
        continue;
      }
      if (isa<ConstantExpr>(U) || isa<GetElementPtrInst>(U) ||
          isa<BitCastInst>(U)) {
        Worklist.append(U->users().begin(), U->users().end());
      }
    }
    return false;
  }

  class EnzymePreArchivePass : public PassInfoMixin<EnzymePreArchivePass> {
    std::unique_ptr<llvm::Module> &m_Archive;
    int m_OrigOptLevel;
    bool m_NeedsEnzymePipeline;

  public:
    EnzymePreArchivePass(std::unique_ptr<llvm::Module> &Archive,
                         int OrigOptLevel, bool NeedsEnzymePipeline)
        : m_Archive(Archive), m_OrigOptLevel(OrigOptLevel),
          m_NeedsEnzymePipeline(NeedsEnzymePipeline) {}

    PreservedAnalyses run(llvm::Module &M, ModuleAnalysisManager &) {
      if (!m_Archive) {
        m_Archive =
            std::make_unique<llvm::Module>("cling_enzyme_archive", M.getContext());
        m_Archive->setDataLayout(M.getDataLayout());
        m_Archive->setTargetTriple(M.getTargetTriple());
      }

      bool Changed = false;

      // 1. Handle __enzyme_provide_body* globals: { &externalDecl, &implFn }
      SmallVector<GlobalVariable *, 4> ProvideBodyGlobals;
      SmallPtrSet<const Function *, 4> ProvidedTargets;
      for (GlobalVariable &GV : M.globals()) {
        if (!GV.getName().contains("__enzyme_provide_body") ||
            !GV.hasInitializer())
          continue;
        const auto *CA = dyn_cast<ConstantAggregate>(GV.getInitializer());
        if (!CA || CA->getNumOperands() < 2)
          continue;
        const auto *TargetFn =
            dyn_cast<Function>(CA->getOperand(0)->stripPointerCasts());
        const auto *ImplFn =
            dyn_cast<Function>(CA->getOperand(1)->stripPointerCasts());
        if (TargetFn && ImplFn && !ImplFn->isDeclaration()) {
          if (Function *ArchFn = cloneFunctionBetweenModules(
                  ImplFn, *m_Archive, nullptr, TargetFn->getName(),
                  /*MaterializingIntoLiveModule=*/false)) {
            ArchFn->setLinkage(GlobalValue::ExternalLinkage);
            ArchFn->setVisibility(GlobalValue::DefaultVisibility);
            ArchFn->setDSOLocal(false);
            ArchFn->removeFnAttr(Attribute::NoInline);
            ArchFn->addFnAttr("cling_enzyme_provided_body");
            if (m_OrigOptLevel == 0)
              stripSyntheticO0Attrs(*ArchFn);
            ProvidedTargets.insert(TargetFn);
          }
          ProvideBodyGlobals.push_back(&GV);
        }
      }
      for (GlobalVariable *GV : ProvideBodyGlobals) {
        removeGlobalFromUsedLists(M, GV);
        GV->eraseFromParent();
        Changed = true;
      }

      // 2. Snapshot user function definitions into m_Archive before
      // ReuseExistingWeakSymbols can delete weak bodies.
      for (Function &F : M) {
        if (m_NeedsEnzymePipeline && m_OrigOptLevel == 0 && !F.isDeclaration()) {
          stripSyntheticO0Attrs(F);
          Changed = true;
        }
        if (shouldSkipArchivingFunction(F) || F.hasLocalLinkage() ||
            ProvidedTargets.contains(&F))
          continue;
        if (Function *ExistingArchFn = m_Archive->getFunction(F.getName())) {
          if (ExistingArchFn->hasFnAttribute("cling_enzyme_provided_body"))
            continue;
        }
        if (Function *ArchFn = cloneFunctionBetweenModules(
                &F, *m_Archive, nullptr, "",
                /*MaterializingIntoLiveModule=*/false)) {
          ArchFn->setLinkage(GlobalValue::ExternalLinkage);
          ArchFn->setVisibility(GlobalValue::DefaultVisibility);
          ArchFn->setDSOLocal(false);
          if (m_OrigOptLevel == 0)
            stripSyntheticO0Attrs(*ArchFn);
        }
      }

      // 3. Snapshot global evaluator objects (e.g. `auto df = clad::gradient(f);`)
      // whose constant initializer holds a target Function* in field 0, so
      // cross-transaction `df.execute(...)` calls can specialize the thunk.
      for (const GlobalVariable &GV : M.globals()) {
        if (!GV.hasName() || GV.hasLocalLinkage() || !GV.hasInitializer())
          continue;
        const Function *TargetFn = extractEvaluatorFunction(GV.getInitializer());
        if (!TargetFn || !TargetFn->hasName())
          continue;
        Function *ArchFn = m_Archive->getFunction(TargetFn->getName());
        if (!ArchFn) {
          ArchFn = Function::Create(TargetFn->getFunctionType(),
                                    GlobalValue::ExternalLinkage,
                                    TargetFn->getAddressSpace(),
                                    TargetFn->getName(), m_Archive.get());
          ArchFn->setVisibility(GlobalValue::DefaultVisibility);
          ArchFn->setDSOLocal(false);
        }
        LLVMContext &Ctx = m_Archive->getContext();
        SmallVector<Metadata *, 3> Ops;
        Ops.push_back(MDString::get(Ctx, GV.getName()));
        Ops.push_back(ValueAsMetadata::get(ArchFn));
        if (ConstantInt *Idx = extractEvaluatorTargetIdx(GV.getInitializer()))
          Ops.push_back(ConstantAsMetadata::get(Idx));
        NamedMDNode *EvalMD =
            m_Archive->getOrInsertNamedMetadata("cling.enzyme.evaluators");
        EvalMD->addOperand(MDTuple::get(Ctx, Ops));
      }

      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };

  static bool materializeDifferentiableGammaFamily(llvm::Module &M) {
    LLVMContext &Ctx = M.getContext();
    Type *DblTy = Type::getDoubleTy(Ctx);
    Type *FltTy = Type::getFloatTy(Ctx);
    bool Changed = false;

    auto stripCallSiteEnzymeMath = [](Function *Fn) {
      for (User *U : Fn->users()) {
        if (auto *CB = dyn_cast<CallBase>(U)) {
          if (CB->getCalledFunction() == Fn &&
              CB->hasFnAttr("enzyme_math")) {
            CB->removeFnAttr("enzyme_math");
          }
        }
      }
    };

    auto emitLanczosLgammaPos = [DblTy](IRBuilder<> &B, Value *Z) -> Value * {
      static constexpr double kLanczosCoeffs[9] = {
          0.99999999999980993227684700473478,
          676.520368121885098567009190444019,
          -1259.13921672240287047156078755283,
          771.3234287776530788486528258894,
          -176.61502916214059906584551354,
          12.507343278686904814458936853,
          -0.13857109526572011689554707,
          9.984369578019570859563e-6,
          1.50563273514931155834e-7,
      };
      Value *Xm1 = B.CreateFSub(Z, ConstantFP::get(DblTy, 1.0));
      Value *Acc = ConstantFP::get(DblTy, kLanczosCoeffs[0]);
      for (unsigned I = 1; I < 9; ++I) {
        Value *Den =
            B.CreateFAdd(Xm1, ConstantFP::get(DblTy, static_cast<double>(I)));
        Acc = B.CreateFAdd(
            Acc, B.CreateFDiv(ConstantFP::get(DblTy, kLanczosCoeffs[I]), Den));
      }
      Value *T = B.CreateFAdd(Xm1, ConstantFP::get(DblTy, 7.5));
      Value *LogT = B.CreateIntrinsic(Intrinsic::log, {DblTy}, {T});
      Value *LogAcc = B.CreateIntrinsic(Intrinsic::log, {DblTy}, {Acc});
      Value *XpHalf = B.CreateFAdd(Xm1, ConstantFP::get(DblTy, 0.5));
      Value *Res = B.CreateFAdd(
          ConstantFP::get(DblTy, 0.9189385332046727417803297364056176398614),
          B.CreateFMul(XpHalf, LogT));
      Res = B.CreateFSub(Res, T);
      return B.CreateFAdd(Res, LogAcc);
    };

    // Ensure double-precision counterparts exist if float wrappers are present.
    if (Function *LF = M.getFunction("lgammaf")) {
      if (LF->isDeclaration() && !M.getFunction("lgamma")) {
        Function *Decl = Function::Create(
            FunctionType::get(DblTy, {DblTy}, /*isVarArg=*/false),
            GlobalValue::ExternalLinkage, "lgamma", &M);
        Decl->setVisibility(GlobalValue::DefaultVisibility);
        Decl->setDSOLocal(false);
        Changed = true;
      }
    }
    if (Function *TF = M.getFunction("tgammaf")) {
      if (TF->isDeclaration() && !M.getFunction("tgamma")) {
        Function *Decl = Function::Create(
            FunctionType::get(DblTy, {DblTy}, /*isVarArg=*/false),
            GlobalValue::ExternalLinkage, "tgamma", &M);
        Decl->setVisibility(GlobalValue::DefaultVisibility);
        Decl->setDSOLocal(false);
        Changed = true;
      }
    }

    for (bool IsLogGamma : {true, false}) {
      StringRef Name = IsLogGamma ? "lgamma" : "tgamma";
      Function *Fn = M.getFunction(Name);
      if (!Fn || Fn->arg_size() != 1 || !Fn->getReturnType()->isDoubleTy() ||
          !Fn->getArg(0)->getType()->isDoubleTy())
        continue;

      stripCallSiteEnzymeMath(Fn);
      std::string CustomTag = ("__cling_lanczos_" + Name).str();
      if (!Fn->hasFnAttribute("enzyme_math") ||
          Fn->getFnAttribute("enzyme_math").getValueAsString() != CustomTag) {
        Fn->addFnAttr("enzyme_math", CustomTag);
        Changed = true;
      }
      if (!Fn->hasFnAttribute("enzyme_shouldrecompute")) {
        Fn->addFnAttr("enzyme_shouldrecompute");
        Changed = true;
      }
      if (!Fn->isDeclaration())
        continue;

      Fn->setLinkage(GlobalValue::AvailableExternallyLinkage);
      Fn->setVisibility(GlobalValue::DefaultVisibility);
      Fn->setDSOLocal(false);
      Fn->removeFnAttr(Attribute::OptimizeNone);
      Fn->removeFnAttr(Attribute::NoInline);
      Fn->addFnAttr(Attribute::NoUnwind);
      Fn->addFnAttr(Attribute::WillReturn);
      Fn->setMemoryEffects(MemoryEffects::none());

      BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", Fn);
      BasicBlock *ReflBB = BasicBlock::Create(Ctx, "refl", Fn);
      BasicBlock *PosBB = BasicBlock::Create(Ctx, "pos", Fn);

      Value *X = Fn->getArg(0);
      IRBuilder<> B(EntryBB);
      Value *IsLtHalf = B.CreateFCmpOLT(X, ConstantFP::get(DblTy, 0.5));
      B.CreateCondBr(IsLtHalf, ReflBB, PosBB);

      B.SetInsertPoint(ReflBB);
      Value *OneMinusX = B.CreateFSub(ConstantFP::get(DblTy, 1.0), X);
      Value *LgOneMinusX = emitLanczosLgammaPos(B, OneMinusX);
      Value *Pi = ConstantFP::get(DblTy, 3.141592653589793238462643383279502884);
      Value *SinPiX =
          B.CreateIntrinsic(Intrinsic::sin, {DblTy}, {B.CreateFMul(Pi, X)});
      if (IsLogGamma) {
        // log(pi) - 0.5 * log(sin(pi * x)^2) - lgamma_pos(1 - x)
        Value *SinSq = B.CreateFMul(SinPiX, SinPiX);
        Value *HalfLogSinSq = B.CreateFMul(
            ConstantFP::get(DblTy, 0.5),
            B.CreateIntrinsic(Intrinsic::log, {DblTy}, {SinSq}));
        Value *ReflRes = B.CreateFSub(
            ConstantFP::get(DblTy, 1.1447298858494001741434273513530587116473),
            HalfLogSinSq);
        B.CreateRet(B.CreateFSub(ReflRes, LgOneMinusX));
      } else {
        Value *ExpNegLg = B.CreateIntrinsic(
            Intrinsic::exp, {DblTy}, {B.CreateFNeg(LgOneMinusX)});
        B.CreateRet(B.CreateFMul(B.CreateFDiv(Pi, SinPiX), ExpNegLg));
      }

      B.SetInsertPoint(PosBB);
      Value *LgPos = emitLanczosLgammaPos(B, X);
      if (IsLogGamma) {
        B.CreateRet(LgPos);
      } else {
        B.CreateRet(B.CreateIntrinsic(Intrinsic::exp, {DblTy}, {LgPos}));
      }
      Changed = true;
    }

    for (bool IsLogGamma : {true, false}) {
      StringRef Name = IsLogGamma ? "lgammaf" : "tgammaf";
      StringRef DblName = IsLogGamma ? "lgamma" : "tgamma";
      Function *Fn = M.getFunction(Name);
      Function *DblFn = M.getFunction(DblName);
      if (!Fn || !DblFn || Fn->arg_size() != 1 ||
          !Fn->getReturnType()->isFloatTy() ||
          !Fn->getArg(0)->getType()->isFloatTy())
        continue;

      stripCallSiteEnzymeMath(Fn);
      std::string CustomTag = ("__cling_lanczos_" + Name).str();
      if (!Fn->hasFnAttribute("enzyme_math") ||
          Fn->getFnAttribute("enzyme_math").getValueAsString() != CustomTag) {
        Fn->addFnAttr("enzyme_math", CustomTag);
        Changed = true;
      }
      if (!Fn->hasFnAttribute("enzyme_shouldrecompute")) {
        Fn->addFnAttr("enzyme_shouldrecompute");
        Changed = true;
      }
      if (!Fn->isDeclaration())
        continue;

      Fn->setLinkage(GlobalValue::AvailableExternallyLinkage);
      Fn->setVisibility(GlobalValue::DefaultVisibility);
      Fn->setDSOLocal(false);
      Fn->removeFnAttr(Attribute::OptimizeNone);
      Fn->removeFnAttr(Attribute::NoInline);
      Fn->addFnAttr(Attribute::NoUnwind);
      Fn->addFnAttr(Attribute::WillReturn);
      Fn->setMemoryEffects(MemoryEffects::none());

      BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", Fn);
      IRBuilder<> B(EntryBB);
      Value *Ext = B.CreateFPExt(Fn->getArg(0), DblTy);
      Value *Call = B.CreateCall(DblFn, {Ext});
      B.CreateRet(B.CreateFPTrunc(Call, FltTy));
      Changed = true;
    }

    return Changed;
  }

  static LLVMValueRef clingEnzymeAllocator(LLVMBuilderRef BRef,
                                           LLVMTypeRef TRef,
                                           LLVMValueRef CountRef,
                                           LLVMValueRef AlignRef,
                                           uint8_t /*isDefault*/,
                                           LLVMValueRef *ZeroMem) {
    auto *B = unwrap(BRef);
    Type *T = unwrap(TRef);
    Value *Count = unwrap(CountRef);
    Value *Align = unwrap(AlignRef);
    Module &M = *B->GetInsertBlock()->getParent()->getParent();
    Type *IntPtrTy = M.getDataLayout().getIntPtrType(M.getContext(), 0);
    if (Count->getType() != IntPtrTy)
      Count = B->CreateIntCast(Count, IntPtrTy, /*isSigned=*/false);
    if (Align->getType() != IntPtrTy)
      Align = B->CreateIntCast(Align, IntPtrTy, /*isSigned=*/false);
    CallInst *MCall = B->CreateMalloc(IntPtrTy, T, Align, Count, nullptr, "");
    MCall->addRetAttr(Attribute::NoAlias);
    MCall->addRetAttr(Attribute::NonNull);
    if (ZeroMem) {
      Value *Bytes =
          B->CreateMul(Align, Count, "", /*HasNUW=*/true, /*HasNSW=*/true);
      CallInst *Z = B->CreateMemSet(MCall, B->getInt8(0), Bytes, MaybeAlign());
      *ZeroMem = wrap(Z);
    }
    return wrap(MCall);
  }

  static bool materializeDifferentiableLascl(llvm::Module &M) {
    // Enzyme's reverse-mode BLAS derivatives (e.g. sgemm_/dgemm_) emit calls to
    // LAPACK's slascl_/dlascl_, which do not yet have a built-in forward-mode
    // TableGen pattern in Enzyme. Synthesizing a compact matrix-scaling body
    // allows second-order forward-over-reverse (Hessian) differentiation of
    // BLAS-backed models while keeping native BLAS calls in both passes.
    bool NeedSingle = false;
    bool NeedDouble = false;
    for (const Function &F : M) {
      StringRef N = F.getName();
      if (N == "sgemm_" || N == "sgemv_" || N == "ssymm_" || N == "ssyrk_" ||
          N == "ssyr2k_" || N == "strmm_" || N == "strsm_" || N == "slacpy_" ||
          N == "slascl_") {
        NeedSingle = true;
      } else if (N == "dgemm_" || N == "dgemv_" || N == "dsymm_" ||
                 N == "dsyrk_" || N == "dsyr2k_" || N == "dtrmm_" ||
                 N == "dtrsm_" || N == "dlacpy_" || N == "dlascl_") {
        NeedDouble = true;
      }
    }
    if (!NeedSingle && !NeedDouble)
      return false;

    LLVMContext &Ctx = M.getContext();
    Type *PtrTy = PointerType::getUnqual(Ctx);
    Type *I8Ty = Type::getInt8Ty(Ctx);
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *I64Ty = Type::getInt64Ty(Ctx);
    Type *IntPtrTy = M.getDataLayout().getIntPtrType(Ctx, 0);
    bool Changed = false;

    for (bool IsDouble : {false, true}) {
      if (IsDouble ? !NeedDouble : !NeedSingle)
        continue;
      StringRef Name = IsDouble ? "dlascl_" : "slascl_";
      Type *ElemTy = IsDouble ? Type::getDoubleTy(Ctx) : Type::getFloatTy(Ctx);

      Function *Fn = M.getFunction(Name);
      if (!Fn) {
        SmallVector<Type *, 11> ParamTys(10, PtrTy);
        ParamTys.push_back(IntPtrTy);
        FunctionType *FT =
            FunctionType::get(Type::getVoidTy(Ctx), ParamTys, /*isVarArg=*/false);
        Fn = Function::Create(FT, GlobalValue::InternalLinkage, Name, &M);
      } else {
        if (!Fn->isDeclaration())
          continue;
        if (!Fn->getReturnType()->isVoidTy() || Fn->arg_size() < 10)
          continue;
        bool AllPtrs = true;
        for (unsigned i = 0; i < 10; ++i) {
          if (!Fn->getArg(i)->getType()->isPointerTy())
            AllPtrs = false;
        }
        if (!AllPtrs)
          continue;
        Fn->setLinkage(GlobalValue::AvailableExternallyLinkage);
        Fn->setVisibility(GlobalValue::DefaultVisibility);
        Fn->setDSOLocal(false);
      }

      Fn->removeFnAttr(Attribute::OptimizeNone);
      Fn->removeFnAttr(Attribute::NoInline);
      Fn->addFnAttr(Attribute::NoUnwind);
      Fn->addFnAttr(Attribute::WillReturn);
      for (unsigned InactiveIdx : {0u, 1u, 2u, 5u, 6u, 8u, 9u}) {
        Fn->addParamAttr(InactiveIdx, Attribute::get(Ctx, "enzyme_inactive"));
      }

      BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", Fn);
      BasicBlock *InfoBB = BasicBlock::Create(Ctx, "set_info", Fn);
      BasicBlock *CheckBB = BasicBlock::Create(Ctx, "check", Fn);
      BasicBlock *ColHdrBB = BasicBlock::Create(Ctx, "col_hdr", Fn);
      BasicBlock *RowBodyBB = BasicBlock::Create(Ctx, "row_body", Fn);
      BasicBlock *ColLatchBB = BasicBlock::Create(Ctx, "col_latch", Fn);
      BasicBlock *ExitBB = BasicBlock::Create(Ctx, "exit", Fn);

      IRBuilder<> B(EntryBB);
      Value *TypePtr = Fn->getArg(0);
      Value *CFromPtr = Fn->getArg(3);
      Value *CToPtr = Fn->getArg(4);
      Value *MPtr = Fn->getArg(5);
      Value *NPtr = Fn->getArg(6);
      Value *APtr = Fn->getArg(7);
      Value *LdaPtr = Fn->getArg(8);
      Value *InfoPtr = Fn->getArg(9);

      Value *HasInfo =
          B.CreateICmpNE(InfoPtr, ConstantPointerNull::get(cast<PointerType>(PtrTy)));
      B.CreateCondBr(HasInfo, InfoBB, CheckBB);

      B.SetInsertPoint(InfoBB);
      StoreInst *InfoStore = B.CreateStore(ConstantInt::get(I32Ty, 0), InfoPtr);
      InfoStore->setMetadata("enzyme_inactive", MDNode::get(Ctx, {}));
      B.CreateBr(CheckBB);

      B.SetInsertPoint(CheckBB);
      Value *TChar = B.CreateLoad(I8Ty, TypePtr);
      Value *CFrom = B.CreateLoad(ElemTy, CFromPtr);
      Value *CTo = B.CreateLoad(ElemTy, CToPtr);
      Value *Mul = B.CreateFDiv(CTo, CFrom);
      Value *Rows32 = B.CreateLoad(I32Ty, MPtr);
      Value *Cols32 = B.CreateLoad(I32Ty, NPtr);
      Value *Lda32 = B.CreateLoad(I32Ty, LdaPtr);

      Value *HasRows = B.CreateICmpSGT(Rows32, ConstantInt::get(I32Ty, 0));
      Value *HasCols = B.CreateICmpSGT(Cols32, ConstantInt::get(I32Ty, 0));
      Value *HasWork = B.CreateAnd(HasRows, HasCols);
      Value *NotOne = B.CreateFCmpUNE(Mul, ConstantFP::get(ElemTy, 1.0));
      Value *DoLoop = B.CreateAnd(HasWork, NotOne);

      Value *IsL = B.CreateOr(B.CreateICmpEQ(TChar, ConstantInt::get(I8Ty, 'L')),
                              B.CreateICmpEQ(TChar, ConstantInt::get(I8Ty, 'l')));
      Value *IsU = B.CreateOr(B.CreateICmpEQ(TChar, ConstantInt::get(I8Ty, 'U')),
                              B.CreateICmpEQ(TChar, ConstantInt::get(I8Ty, 'u')));
      Value *Rows64 = B.CreateSExt(Rows32, I64Ty);
      Value *Cols64 = B.CreateSExt(Cols32, I64Ty);
      Value *Lda64 = B.CreateSExt(Lda32, I64Ty);
      B.CreateCondBr(DoLoop, ColHdrBB, ExitBB);

      B.SetInsertPoint(ColHdrBB);
      PHINode *ColPhi = B.CreatePHI(I64Ty, 2, "c");
      ColPhi->addIncoming(ConstantInt::get(I64Ty, 0), CheckBB);
      Value *RStart = B.CreateSelect(IsL, ColPhi, ConstantInt::get(I64Ty, 0));
      Value *ColPlus1 =
          B.CreateAdd(ColPhi, ConstantInt::get(I64Ty, 1), "", true, true);
      Value *UEnd = B.CreateSelect(B.CreateICmpSLT(ColPlus1, Rows64), ColPlus1, Rows64);
      Value *REnd = B.CreateSelect(IsU, UEnd, Rows64);
      Value *ColOff = B.CreateNSWMul(ColPhi, Lda64);
      Value *HasRow = B.CreateICmpSLT(RStart, REnd);
      B.CreateCondBr(HasRow, RowBodyBB, ColLatchBB);

      B.SetInsertPoint(RowBodyBB);
      PHINode *RowPhi = B.CreatePHI(I64Ty, 2, "r");
      RowPhi->addIncoming(RStart, ColHdrBB);
      Value *Idx = B.CreateNSWAdd(ColOff, RowPhi);
      Value *ElemPtr = B.CreateInBoundsGEP(ElemTy, APtr, Idx);
      Value *OldVal = B.CreateLoad(ElemTy, ElemPtr);
      B.CreateStore(B.CreateFMul(OldVal, Mul), ElemPtr);
      Value *RowNext =
          B.CreateAdd(RowPhi, ConstantInt::get(I64Ty, 1), "", true, true);
      RowPhi->addIncoming(RowNext, RowBodyBB);
      B.CreateCondBr(B.CreateICmpSLT(RowNext, REnd), RowBodyBB, ColLatchBB);

      B.SetInsertPoint(ColLatchBB);
      Value *ColNext =
          B.CreateAdd(ColPhi, ConstantInt::get(I64Ty, 1), "", true, true);
      ColPhi->addIncoming(ColNext, ColLatchBB);
      B.CreateCondBr(B.CreateICmpSLT(ColNext, Cols64), ColHdrBB, ExitBB);

      B.SetInsertPoint(ExitBB);
      B.CreateRetVoid();
      Changed = true;
    }
    return Changed;
  }

  static bool stripUnsupportedCallSiteTBAA(llvm::Module &M) {
    // Clang attaches errno int TBAA metadata onto scalar libm/intrinsic math
    // calls (<cmath> @exp, @log, @llvm.log.f64, etc.). Enzyme's TypeAnalysis
    // only consumes MD_tbaa on memory/masked intrinsics, memset_pattern16, and
    // pointer-returning calls, printing a diagnostic on other CallBase
    // instructions. Strip MD_tbaa on those unsupported call sites before
    // EnzymeNewPM runs.
    bool Changed = false;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          auto *CB = dyn_cast<CallBase>(&I);
          if (!CB || !CB->getMetadata(LLVMContext::MD_tbaa))
            continue;
          if (CB->getType()->isPointerTy())
            continue;
          if (const Function *Callee = CB->getCalledFunction()) {
            if (Callee->getName() == "memset_pattern16")
              continue;
            switch (Callee->getIntrinsicID()) {
            case Intrinsic::memcpy:
            case Intrinsic::memmove:
            case Intrinsic::memset:
            case Intrinsic::experimental_memset_pattern:
            case Intrinsic::masked_gather:
            case Intrinsic::masked_scatter:
            case Intrinsic::masked_load:
            case Intrinsic::masked_store:
              continue;
            default:
              break;
            }
          }
          CB->setMetadata(LLVMContext::MD_tbaa, nullptr);
          Changed = true;
        }
      }
    }
    return Changed;
  }

  class EnzymeMaterializePass : public PassInfoMixin<EnzymeMaterializePass> {
    llvm::Module *m_Archive;
    bool m_PreEnzymeCleanup;

  public:
    explicit EnzymeMaterializePass(llvm::Module *Archive,
                                   bool PreEnzymeCleanup = false)
        : m_Archive(Archive), m_PreEnzymeCleanup(PreEnzymeCleanup) {}

    PreservedAnalyses run(llvm::Module &M, ModuleAnalysisManager &) {
      if (!m_Archive)
        return PreservedAnalyses::all();

      bool Changed = false;
      SmallVector<Function *, 32> Worklist;
      SmallPtrSet<Function *, 32> Queued;
      for (StringRef HelperName : {"digamma", "digammaf"}) {
        if (Function *ArchHelper = m_Archive->getFunction(HelperName)) {
          if (!ArchHelper->isDeclaration() && !M.getFunction(HelperName)) {
            Function *Decl = Function::Create(
                ArchHelper->getFunctionType(), GlobalValue::ExternalLinkage,
                ArchHelper->getAddressSpace(), HelperName, &M);
            Decl->setVisibility(GlobalValue::DefaultVisibility);
            Decl->setDSOLocal(false);
            Changed = true;
          }
        }
      }
      for (Function &F : M) {
        if (Queued.insert(&F).second)
          Worklist.push_back(&F);
      }

      SmallPtrSet<const Constant *, 32> VisitedConsts;
      auto enqueueConstantFunctions = [&](const Constant *Root) {
        SmallVector<const Constant *, 16> ConstStack;
        if (Root && VisitedConsts.insert(Root).second)
          ConstStack.push_back(Root);
        while (!ConstStack.empty()) {
          const Constant *C = ConstStack.pop_back_val();
          if (const auto *Callee =
                  dyn_cast<Function>(C->stripPointerCasts())) {
            auto *MutCallee = const_cast<Function *>(Callee);
            if (Queued.insert(MutCallee).second)
              Worklist.push_back(MutCallee);
            continue;
          }
          if (isa<GlobalValue>(C))
            continue;
          for (const Use &Op : C->operands()) {
            if (const auto *OpC = dyn_cast<Constant>(Op.get())) {
              if (VisitedConsts.insert(OpC).second)
                ConstStack.push_back(OpC);
            }
          }
        }
      };

      auto enqueueCallees = [&](Function &Fn) {
        for (BasicBlock &BB : Fn) {
          for (Instruction &I : BB) {
            for (Use &Op : I.operands()) {
              if (auto *Callee =
                      dyn_cast<Function>(Op.get()->stripPointerCasts())) {
                if (Queued.insert(Callee).second)
                  Worklist.push_back(Callee);
              } else if (const auto *C = dyn_cast<Constant>(Op.get())) {
                enqueueConstantFunctions(C);
              }
            }
          }
        }
      };

      while (!Worklist.empty()) {
        Function *F = Worklist.pop_back_val();
        if (!F->hasName() || F->isIntrinsic())
          continue;

        if (!F->isDeclaration())
          enqueueCallees(*F);

        Function *ArchF = m_Archive->getFunction(F->getName());
        if (!ArchF) {
          if (F->isDeclaration() && F->hasFnAttribute("enzyme_inactive") &&
              !F->hasFnAttribute(Attribute::NoFree)) {
            F->addFnAttr(Attribute::NoFree);
            Changed = true;
          }
          continue;
        }

        // Propagate archived Enzyme attributes & custom derivative metadata.
        for (StringRef AttrName : kEnzymeFnStrAttrs) {
          if (ArchF->hasFnAttribute(AttrName) && !F->hasFnAttribute(AttrName)) {
            F->addFnAttr(ArchF->getFnAttribute(AttrName));
            Changed = true;
          }
        }
        if (ArchF->hasFnAttribute(Attribute::NoFree) &&
            !F->hasFnAttribute(Attribute::NoFree)) {
          F->addFnAttr(Attribute::NoFree);
          Changed = true;
        }

        SmallVector<Function *, 4> NewlyMaterialized;
        for (StringRef MDName : kEnzymeCustomMDKinds) {
          if (MDNode *ArchMD = ArchF->getMetadata(MDName)) {
            if (!hasValidCustomDerivativeMD(*F, MDName)) {
              if (auto *Remapped = dyn_cast_or_null<MDNode>(
                      remapCustomDerivativeMD(ArchMD, M, *m_Archive,
                                              /*MaterializingIntoLive=*/true,
                                              &NewlyMaterialized))) {
                F->setMetadata(MDName, Remapped);
                Changed = true;
              }
            }
          }
        }
        for (Function *NewFn : NewlyMaterialized) {
          if (Queued.insert(NewFn).second)
            Worklist.push_back(NewFn);
          enqueueCallees(*NewFn);
        }

        // Materialize archived body as available_externally if F is an external
        // declaration without a custom rule / math override, or replace F's
        // body in-place when __enzyme_provide_body explicitly supplied an AD
        // body for F.
        const bool NeedsProvidedOverride =
            ArchF->hasFnAttribute("cling_enzyme_provided_body") &&
            !F->hasFnAttribute("cling_enzyme_provided_body");
        if ((F->isDeclaration() || NeedsProvidedOverride) &&
            !ArchF->isDeclaration() && !blocksEnzymeBodyMaterialization(*F)) {
          const bool WasDecl = F->isDeclaration();
          const auto OrigLinkage = F->getLinkage();
          Comdat *OrigComdat = F->getComdat();
          if (cloneFunctionBetweenModules(
                  ArchF, M, F, "", /*MaterializingIntoLiveModule=*/true)) {
            F->setLinkage(WasDecl ? GlobalValue::AvailableExternallyLinkage
                                  : OrigLinkage);
            if (!WasDecl && OrigComdat)
              F->setComdat(OrigComdat);
            F->setVisibility(GlobalValue::DefaultVisibility);
            F->setDSOLocal(false);
            F->removeFnAttr(Attribute::NoInline);
            enqueueCallees(*F);
            Changed = true;
          }
        }

        if (F->isDeclaration() && F->hasFnAttribute("enzyme_inactive") &&
            !F->hasFnAttribute(Attribute::NoFree)) {
          F->addFnAttr(Attribute::NoFree);
          Changed = true;
        }
      }

      if (m_PreEnzymeCleanup) {
        if (materializeDifferentiableGammaFamily(M))
          Changed = true;
        if (materializeDifferentiableLascl(M))
          Changed = true;
        if (stripUnsupportedCallSiteTBAA(M))
          Changed = true;
      }

      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };

  class EnzymeCaptureRegistrationsPass
      : public PassInfoMixin<EnzymeCaptureRegistrationsPass> {
    llvm::Module *m_Archive;
    bool m_RefreshBodiesOnly;

    static void retainRuleFunctionsInJIT(Metadata *MD, bool &Changed) {
      if (!MD)
        return;
      if (auto *VMD = dyn_cast<ValueAsMetadata>(MD)) {
        if (!VMD->getValue())
          return;
        if (auto *RuleFn =
                dyn_cast<Function>(VMD->getValue()->stripPointerCasts())) {
          if (!RuleFn->isDeclaration() &&
              !RuleFn->hasFnAttribute("cling_enzyme_temp_external") &&
              RuleFn->getLinkage() != GlobalValue::AvailableExternallyLinkage) {
            if (RuleFn->hasFnAttribute("prev_linkage")) {
              int64_t PrevVal = 0;
              if (!RuleFn->getFnAttribute("prev_linkage")
                       .getValueAsString()
                       .getAsInteger(10, PrevVal)) {
                auto PrevLink = static_cast<GlobalValue::LinkageTypes>(PrevVal);
                if (GlobalValue::isDiscardableIfUnused(PrevLink)) {
                  RuleFn->addFnAttr(
                      "prev_linkage",
                      std::to_string(GlobalValue::WeakODRLinkage));
                  RuleFn->setVisibility(GlobalValue::DefaultVisibility);
                  RuleFn->setDSOLocal(false);
                  Changed = true;
                }
              }
            } else if (RuleFn->isDiscardableIfUnused()) {
              RuleFn->setLinkage(GlobalValue::WeakODRLinkage);
              RuleFn->setVisibility(GlobalValue::DefaultVisibility);
              RuleFn->setDSOLocal(false);
              Changed = true;
            }
          }
        }
        return;
      }
      if (auto *Tuple = dyn_cast<MDTuple>(MD)) {
        for (const MDOperand &Op : Tuple->operands())
          retainRuleFunctionsInJIT(Op.get(), Changed);
      }
    }

    static bool hasNativeRuleBody(Metadata *MD) {
      if (!MD)
        return false;
      if (auto *VMD = dyn_cast<ValueAsMetadata>(MD)) {
        if (!VMD->getValue())
          return false;
        if (const auto *RuleFn =
                dyn_cast<Function>(VMD->getValue()->stripPointerCasts())) {
          return !RuleFn->isDeclaration() &&
                 !RuleFn->hasFnAttribute("cling_enzyme_temp_external") &&
                 RuleFn->getLinkage() !=
                     GlobalValue::AvailableExternallyLinkage;
        }
        return false;
      }
      if (auto *Tuple = dyn_cast<MDTuple>(MD)) {
        for (const MDOperand &Op : Tuple->operands()) {
          if (hasNativeRuleBody(Op.get()))
            return true;
        }
      }
      return false;
    }

  public:
    explicit EnzymeCaptureRegistrationsPass(llvm::Module *Archive,
                                            bool RefreshBodiesOnly = false)
        : m_Archive(Archive), m_RefreshBodiesOnly(RefreshBodiesOnly) {}

    PreservedAnalyses run(llvm::Module &M, ModuleAnalysisManager &) {
      if (!m_Archive)
        return PreservedAnalyses::all();

      bool Changed = false;
      for (Function &F : M) {
        if (!F.hasName() || !hasEnzymeRuleOrAttr(F))
          continue;

        Function *ArchF = m_Archive->getFunction(F.getName());
        if (!ArchF) {
          if (m_RefreshBodiesOnly)
            continue;
          ArchF = Function::Create(F.getFunctionType(),
                                   GlobalValue::ExternalLinkage,
                                   F.getAddressSpace(), F.getName(), m_Archive);
          ArchF->setVisibility(GlobalValue::DefaultVisibility);
          ArchF->setDSOLocal(false);
        }

        if (!m_RefreshBodiesOnly) {
          for (StringRef AttrName : kEnzymeFnStrAttrs) {
            if (F.hasFnAttribute(AttrName))
              ArchF->addFnAttr(F.getFnAttribute(AttrName));
          }
          if (F.hasFnAttribute(Attribute::NoFree))
            ArchF->addFnAttr(Attribute::NoFree);
        }

        for (StringRef MDName : kEnzymeCustomMDKinds) {
          if (!hasValidCustomDerivativeMD(F, MDName))
            continue;
          MDNode *MD = F.getMetadata(MDName);
          if (!m_RefreshBodiesOnly)
            retainRuleFunctionsInJIT(MD, Changed);
          if (m_RefreshBodiesOnly && !hasNativeRuleBody(MD))
            continue;
          if (auto *Remapped = dyn_cast_or_null<MDNode>(
                  remapCustomDerivativeMD(MD, *m_Archive, M,
                                          /*MaterializingIntoLive=*/false))) {
            ArchF->setMetadata(MDName, Remapped);
          }
        }
      }

      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };

  class EnzymeSpecializeThunkPass
      : public PassInfoMixin<EnzymeSpecializeThunkPass> {
    llvm::Module *m_Archive;

  public:
    explicit EnzymeSpecializeThunkPass(llvm::Module *Archive = nullptr)
        : m_Archive(Archive) {}

    PreservedAnalyses run(llvm::Module &M, ModuleAnalysisManager &) {
      bool Changed = false;
      SmallVector<Function *, 16> Worklist;
      for (Function &F : M) {
        if (!F.isDeclaration())
          Worklist.push_back(&F);
      }

      unsigned SpecId = 0;
      while (!Worklist.empty()) {
        Function *Caller = Worklist.pop_back_val();
        for (BasicBlock &BB : *Caller) {
          for (Instruction &I : BB) {
            auto *CI = dyn_cast<CallBase>(&I);
            if (!CI)
              continue;
            Function *Callee = CI->getCalledFunction();
            if (!Callee || !isEnzymeCallName(Callee->getName()))
              continue;
            const unsigned TargetOpIdx = CI->hasStructRetAttr() ? 1u : 0u;
            if (CI->arg_size() < TargetOpIdx + 3)
              continue;
            auto *TargetFn = dyn_cast<Function>(
                CI->getArgOperand(TargetOpIdx)->stripPointerCasts());
            if (!TargetFn || TargetFn->isDeclaration() ||
                !(TargetFn->getName().contains("FwdDiffNth") ||
                  TargetFn->getName().contains("EnzymeFnThunk")))
              continue;
            // Find the enzyme_const, <Constant FnPtr> pair for formal param 0.
            // Note: call-level flags like enzyme_width, B may precede param 0,
            // so scan for the first Function* constant operand (or load from a
            // constant-initialized global evaluator object) after TargetOpIdx.
            unsigned FnOpIdx = 0;
            Constant *InnerFnConst = nullptr;
            ConstantInt *GlobalTargetIdx = nullptr;
            for (unsigned OpIdx = TargetOpIdx + 2, E = CI->arg_size();
                 OpIdx < E; ++OpIdx) {
              Value *Stripped = CI->getArgOperand(OpIdx)->stripPointerCasts();
              if (auto *FnC = dyn_cast<Function>(Stripped)) {
                FnOpIdx = OpIdx;
                InnerFnConst = FnC;
                break;
              }
              if (auto *LI = dyn_cast<LoadInst>(Stripped)) {
                const Value *Ptr =
                    LI->getPointerOperand()->stripPointerCasts();
                if (const auto *GV = dyn_cast<GlobalVariable>(Ptr)) {
                  if (!hasRuntimeStoresToGlobal(*GV)) {
                    if (GV->hasInitializer()) {
                      if (const Function *InitFn =
                              extractEvaluatorFunction(GV->getInitializer())) {
                        FnOpIdx = OpIdx;
                        InnerFnConst = const_cast<Function *>(InitFn);
                        GlobalTargetIdx =
                            extractEvaluatorTargetIdx(GV->getInitializer());
                        break;
                      }
                    } else if (m_Archive && GV->hasName()) {
                      if (NamedMDNode *EvalMD = m_Archive->getNamedMetadata(
                              "cling.enzyme.evaluators")) {
                        for (int i =
                                 static_cast<int>(EvalMD->getNumOperands()) - 1;
                             i >= 0; --i) {
                          const MDNode *Entry = EvalMD->getOperand(i);
                          if (Entry->getNumOperands() < 2)
                            continue;
                          const auto *NameMD =
                              dyn_cast<MDString>(Entry->getOperand(0));
                          if (!NameMD || NameMD->getString() != GV->getName())
                            continue;
                          const auto *FnVMD =
                              dyn_cast<ValueAsMetadata>(Entry->getOperand(1));
                          if (!FnVMD || !FnVMD->getValue())
                            continue;
                          if (const auto *ArchFn = dyn_cast<Function>(
                                  FnVMD->getValue()->stripPointerCasts())) {
                            Function *LiveFn =
                                M.getFunction(ArchFn->getName());
                            if (!LiveFn) {
                              LiveFn = Function::Create(
                                  ArchFn->getFunctionType(),
                                  GlobalValue::ExternalLinkage,
                                  ArchFn->getAddressSpace(), ArchFn->getName(),
                                  &M);
                              LiveFn->setVisibility(
                                  GlobalValue::DefaultVisibility);
                              LiveFn->setDSOLocal(false);
                            }
                            FnOpIdx = OpIdx;
                            InnerFnConst = LiveFn;
                            if (Entry->getNumOperands() >= 3) {
                              if (const auto *IdxMD =
                                      dyn_cast<ConstantAsMetadata>(
                                          Entry->getOperand(2)))
                                GlobalTargetIdx =
                                    dyn_cast<ConstantInt>(IdxMD->getValue());
                            }
                            break;
                          }
                        }
                        if (InnerFnConst)
                          break;
                      }
                    }
                  }
                }
              }
            }
            if (!InnerFnConst)
              continue;

            ValueToValueMapTy VMap;
            Function *SpecFn = CloneFunction(TargetFn, VMap);
            SpecFn->setName(TargetFn->getName() + ".spec." +
                            Twine(SpecId++));
            SpecFn->setLinkage(GlobalValue::InternalLinkage);
            SpecFn->removeFnAttr(Attribute::NoInline);
            if (SpecFn->arg_size() >= 1) {
              Argument *FnArg = SpecFn->getArg(0);
              Constant *CastFn = ConstantExpr::getPointerBitCastOrAddrSpaceCast(
                  InnerFnConst, FnArg->getType());
              FnArg->replaceAllUsesWith(CastFn);
            }
            if (TargetFn->getName().contains("FwdDiffNth") &&
                FnOpIdx + 2 < CI->arg_size() && SpecFn->arg_size() >= 2) {
              ConstantInt *ConstIdx =
                  dyn_cast<ConstantInt>(CI->getArgOperand(FnOpIdx + 2));
              if (!ConstIdx)
                ConstIdx = GlobalTargetIdx;
              if (ConstIdx) {
                Argument *IdxArg = SpecFn->getArg(1);
                if (IdxArg->getType() == ConstIdx->getType())
                  IdxArg->replaceAllUsesWith(ConstIdx);
              }
            }
            CI->setArgOperand(
                TargetOpIdx,
                ConstantExpr::getPointerBitCastOrAddrSpaceCast(
                    SpecFn, CI->getArgOperand(TargetOpIdx)->getType()));
            Worklist.push_back(SpecFn);
            Changed = true;
          }
        }
      }

      // Erase unspecialized generic thunks that now have no remaining uses so
      // EnzymeNewPM does not attempt to differentiate dead indirect-call bodies.
      bool ErasedDeadThunk = false;
      do {
        ErasedDeadThunk = false;
        for (Function &F : make_early_inc_range(M)) {
          if (F.isDeclaration() || !F.isDiscardableIfUnused())
            continue;
          StringRef Name = F.getName();
          if (Name.contains(".spec.") ||
              !(Name.contains("FwdDiffNth") || Name.contains("EnzymeFnThunk")))
            continue;
          F.removeDeadConstantUsers();
          if (F.use_empty()) {
            F.eraseFromParent();
            ErasedDeadThunk = true;
            Changed = true;
          }
        }
      } while (ErasedDeadThunk);

      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };

  class EnzymeCleanupAvailableExternallyPass
      : public PassInfoMixin<EnzymeCleanupAvailableExternallyPass> {
  public:
    PreservedAnalyses run(llvm::Module &M, ModuleAnalysisManager &) {
      bool Changed = false;
      for (Function &F : M) {
        if (!F.isDeclaration() &&
            (F.getLinkage() == GlobalValue::AvailableExternallyLinkage ||
             F.hasFnAttribute("cling_enzyme_temp_external"))) {
          F.removeFnAttr("cling_enzyme_temp_external");
          F.deleteBody();
          F.setLinkage(GlobalValue::ExternalLinkage);
          F.setVisibility(GlobalValue::DefaultVisibility);
          F.setDSOLocal(false);
          Changed = true;
        }
      }
      bool ErasedLocal = false;
      do {
        ErasedLocal = false;
        for (Function &F : make_early_inc_range(M)) {
          if (F.isDeclaration() || !F.hasLocalLinkage())
            continue;
          F.removeDeadConstantUsers();
          if (F.use_empty()) {
            F.eraseFromParent();
            ErasedLocal = true;
            Changed = true;
          }
        }
      } while (ErasedLocal);
      for (GlobalVariable &GV : M.globals()) {
        if (!GV.isDeclaration() &&
            GV.getLinkage() == GlobalValue::AvailableExternallyLinkage) {
          GV.setInitializer(nullptr);
          GV.setLinkage(GlobalValue::ExternalLinkage);
          GV.setVisibility(GlobalValue::DefaultVisibility);
          GV.setDSOLocal(false);
          Changed = true;
        }
      }
      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
} // namespace

// From clang/lib/CodeGen/BackendUtil.cpp
static OptimizationLevel mapToLevel(int OptLevel, unsigned OptimizeSize) {
  switch (OptLevel) {
    default: llvm_unreachable("Invalid optimization level!");

    case 0: return OptimizationLevel::O0;

    case 1: return OptimizationLevel::O1;

    case 2:
      switch (OptimizeSize) {
        default: llvm_unreachable("Invalid optimization level for size!");

        case 0: return OptimizationLevel::O2;

        case 1: return OptimizationLevel::Os;

        case 2: return OptimizationLevel::Oz;
      }

    case 3: return OptimizationLevel::O3;
  }
}

BackendPasses::BackendPasses(const clang::CodeGenOptions &CGOpts,
                             IncrementalJIT &JIT, llvm::TargetMachine& TM):
   m_TM(TM),
   m_JIT(JIT),
   m_CGOpts(CGOpts)
{}


BackendPasses::~BackendPasses() {
  //delete m_PMBuilder->Inliner;
}

void BackendPasses::CreatePasses(int OptLevel, int OrigOptLevel,
                                 bool NeedsEnzymePipeline, bool HasEnzymeCalls,
                                 llvm::ModulePassManager& MPM,
                                 llvm::LoopAnalysisManager& LAM,
                                 llvm::FunctionAnalysisManager& FAM,
                                 llvm::CGSCCAnalysisManager& CGAM,
                                 llvm::ModuleAnalysisManager& MAM,
                                 PassInstrumentationCallbacks& PIC,
                                 StandardInstrumentations& SI) {

  // TODO: Remove this pass once we upgrade past LLVM 19 that includes the fix.
  MPM.addPass(WorkAroundConstructorPriorityBugPass());
  MPM.addPass(UniqueInitFunctionNamePass());
  MPM.addPass(KeepLocalGVPass());
  MPM.addPass(WeakTypeinfoVTablePass());
  MPM.addPass(EnzymePreArchivePass(m_EnzymeIRArchive, OrigOptLevel,
                                   NeedsEnzymePipeline));
  MPM.addPass(ReuseExistingWeakSymbols(m_JIT, NeedsEnzymePipeline));
  MPM.addPass(PreventLocalOptPass());
  if (HasEnzymeCalls)
    MPM.addPass(EnzymeMaterializePass(m_EnzymeIRArchive.get()));

  // Run verifier after local passes to make sure that IR remains untouched.
  if (m_CGOpts.VerifyModule)
    MPM.addPass(VerifierPass());

  // Handle disabling of LLVM optimization, where we want to preserve the
  // internal module before any optimization.
  if (m_CGOpts.DisableLLVMPasses && !NeedsEnzymePipeline) {
    // Always keep at least ForceInline - NoInlining is deadly for libc++.
    // Inlining = CGOpts.NoInlining;
    MPM.addPass(AlwaysInlinerPass());
  } else if (OptLevel <= 1) {
    // At O0 and O1 we only run the always inliner which is more efficient. At
    // higher optimization levels we run the normal inliner.
    MPM.addPass(AlwaysInlinerPass());

    // Register a callback for disabling all other inliner passes.
    PIC.registerShouldRunOptionalPassCallback(
        [NeedsEnzymePipeline](StringRef P, Any) {
          if (NeedsEnzymePipeline && P == "EliminateAvailableExternallyPass")
            return false;
          if (P == "ModuleInlinerWrapperPass" ||
              P == "InlineAdvisorAnalysisPrinterPass" ||
              P == "PartialInlinerPass" || P == "buildInlinerPipeline" ||
              P == "ModuleInlinerPass" || P == "InlinerPass" ||
              P == "InlineAdvisorAnalysis" ||
              P == "PartiallyInlineLibCallsPass" ||
              P == "RelLookupTableConverterPass" ||
              P == "InlineCostAnnotationPrinterPass" ||
              P == "InlineSizeEstimatorAnalysisPrinterPass" ||
              P == "InlineSizeEstimatorAnalysis")
            return false;

          return true;
        });
  } else {
    // Register a callback for disabling RelLookupTableConverterPass (and
    // retaining available_externally bodies until EnzymeNewPM completes).
    PIC.registerShouldRunOptionalPassCallback(
        [NeedsEnzymePipeline](StringRef P, Any) {
          if (NeedsEnzymePipeline && P == "EliminateAvailableExternallyPass")
            return false;
          return P != "RelLookupTableConverterPass";
        });
  }

  SI.registerCallbacks(PIC, &MAM);

  PipelineTuningOptions PTO;
  std::optional<PGOOptions> PGOOpt;
  PassBuilder PB(&m_TM, PTO, PGOOpt, &PIC);

#ifdef CLING_WITH_ADAPTIVECPP
  getAdaptiveCppPluginInfo().RegisterPassBuilderCallbacks(PB);
#endif

  if (HasEnzymeCalls) {
    llvm::Module *ArchivePtr = m_EnzymeIRArchive.get();
    PB.registerOptimizerEarlyEPCallback(
        [ArchivePtr](ModulePassManager &PM, OptimizationLevel,
                     ThinOrFullLTOPhase) {
          PM.addPass(EnzymeSpecializeThunkPass(ArchivePtr));
          PM.addPass(
              EnzymeMaterializePass(ArchivePtr, /*PreEnzymeCleanup=*/true));
        });
  }

  // Invoke callbacks registered by linked FrontendPlugins (e.g. EnzymePlugin).
  for (const auto &Callback : m_CGOpts.PassBuilderCallbacks) {
    Callback(PB);
  }

  // Attempt to load pass plugins and register their callbacks with PB.
  for (auto& PluginFN : m_CGOpts.PassPlugins) {
    auto PassPlugin = PassPlugin::Load(PluginFN);
    if (PassPlugin) {
      PassPlugin->registerPassBuilderCallbacks(PB);
    }
  }

  if (NeedsEnzymePipeline) {
    llvm::Module *ArchivePtr = m_EnzymeIRArchive.get();
    PB.registerPipelineStartEPCallback(
        [ArchivePtr](ModulePassManager &PM, OptimizationLevel) {
          PM.addPass(EnzymeCaptureRegistrationsPass(ArchivePtr));
        });
  }

  if (!m_CGOpts.DisableLLVMPasses || NeedsEnzymePipeline) {
    // Use the default pass pipeline. We also have to map our optimization
    // levels into one of the distinct levels used to configure the pipeline.
    OptimizationLevel Level = mapToLevel(OptLevel, m_CGOpts.OptimizeSize);
    if (OptLevel == 0) {
      // TODO: Remove this after https://reviews.llvm.org/D146200
      MPM.addPass(PB.buildO0DefaultPipeline(Level));
    } else {
      MPM.addPass(PB.buildPerModuleDefaultPipeline(Level));
    }
  }

  if (NeedsEnzymePipeline) {
    MPM.addPass(EnzymeCaptureRegistrationsPass(m_EnzymeIRArchive.get(),
                                               /*RefreshBodiesOnly=*/true));
    MPM.addPass(EnzymeCleanupAvailableExternallyPass());
  }

  // The function __cuda_module_ctor and __cuda_module_dtor will just generated,
  // if a CUDA fatbinary file exist. Without file path there is no need for the
  // function pass.
  if(!m_CGOpts.CudaGpuBinaryFileName.empty())
    MPM.addPass(UniqueCUDAStructorName());

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
}

void BackendPasses::runOnModule(Module& M, int OptLevel) {

  if (OptLevel < 0)
    OptLevel = 0;
  if (OptLevel > 3)
    OptLevel = 3;

  const int OrigOptLevel = OptLevel;
  const EnzymeModuleInfo EnzymeInfo = inspectEnzymeModule(M);
  if (EnzymeInfo.needsPipeline() && OptLevel < 2)
    OptLevel = 2;
  if (EnzymeInfo.needsPipeline() && &CustomAllocator != nullptr &&
      CustomAllocator == nullptr) {
    CustomAllocator = &clingEnzymeAllocator;
  }

  // Ensure m_EnzymeIRArchive is initialized before CreatePasses captures its
  // pointer for callbacks.
  if (!m_EnzymeIRArchive) {
    m_EnzymeIRArchive =
        std::make_unique<llvm::Module>("cling_enzyme_archive", M.getContext());
    m_EnzymeIRArchive->setDataLayout(M.getDataLayout());
    m_EnzymeIRArchive->setTargetTriple(M.getTargetTriple());
  }

  ModulePassManager MPM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(M.getContext(), m_CGOpts.DebugPassManager);

  CreatePasses(OptLevel, OrigOptLevel, EnzymeInfo.needsPipeline(),
               EnzymeInfo.HasCalls, MPM, LAM, FAM, CGAM, MAM, PIC, SI);

  static constexpr std::array<llvm::CodeGenOptLevel, 4> CGOptLevel{
      {llvm::CodeGenOptLevel::None, llvm::CodeGenOptLevel::Less,
       llvm::CodeGenOptLevel::Default, llvm::CodeGenOptLevel::Aggressive}};
  // TM's OptLevel is used to build orc::SimpleCompiler passes for every Module.
  m_TM.setOptLevel(CGOptLevel[OptLevel]);

  // Now that we have all of the passes ready, run them.
  MPM.run(M, MAM);
}
