#include "scalehls/Transforms/Passes.h"

using namespace mlir;
using namespace scalehls;
using namespace hls;

namespace mlir {
namespace scalehls {

void deleteUnusedFunctions(ModuleOp module, func::FuncOp topFunc) {
    llvm::SmallDenseSet<func::FuncOp> funcsSeen;
    funcsSeen.insert(topFunc);
    std::vector<func::FuncOp> funcQueue;
    funcQueue.push_back(topFunc);
    while (!funcQueue.empty()) {
        auto func = funcQueue.back();
        funcQueue.erase(funcQueue.end() - 1);
        func.walk([&](func::CallOp op) {
            auto callee = SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr());
            auto subFunc = dyn_cast<func::FuncOp>(callee);
            if (subFunc && !funcsSeen.contains(subFunc)) {
                llvm::errs() << "sub-function found: " << subFunc.getName() << "\n";
                funcsSeen.insert(subFunc);
                funcQueue.push_back(subFunc);
            }
        });
    }
    module.walk([&](func::FuncOp func) {
        if (!funcsSeen.contains(func)) {
            llvm::errs() << "Deleting unused function: " << func.getName() << "\n";
            func.erase();
        }
    });
}

} // namespace scalehls
} // namespace mlir

namespace {
    struct FuncDelete : public FuncDeleteBase<FuncDelete> {
      void runOnOperation() override {
        auto module = getOperation();
        for (auto func : module.getOps<func::FuncOp>()) {
            if (hasTopFuncAttr(func)) {
                deleteUnusedFunctions(module, func);
            }
        }
    }
};
} // namespace

std::unique_ptr<Pass> scalehls::createFuncDeletePass() {
  return std::make_unique<FuncDelete>();
}