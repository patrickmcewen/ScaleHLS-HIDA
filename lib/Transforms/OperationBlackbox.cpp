#include "scalehls/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace scalehls;
using namespace hls;

// Get the function name for a given arithmetic operation type
static std::string getBlackboxFunctionName(Operation *op) {
    std::string fn_name = "";
    if (isa<arith::AddFOp>(op)) fn_name += "addf";
    else if (isa<arith::MulFOp>(op)) fn_name += "mulf";
    else if (isa<arith::DivFOp>(op)) fn_name += "divf";
    else if (isa<arith::SubFOp>(op)) fn_name += "subf";
    else if (isa<math::ExpOp>(op)) fn_name += "exp_bb";
    else return "";
    if (!(op->getParentOfType<AffineForOp>())) fn_name += "_ctrl_chain";
    return fn_name;
}

// Get or create a blackbox function declaration
static func::FuncOp getOrCreateBlackboxFunction(ModuleOp module, StringRef funcName, 
                                                 Type resultType, 
                                                 ArrayRef<Type> operandTypes) {
    // Check if function already exists
    if (auto existingFunc = module.lookupSymbol<func::FuncOp>(funcName)) {
        return existingFunc;
    }
    
    // Create function type
    FunctionType funcType = FunctionType::get(module.getContext(), operandTypes, resultType);
    
    // Create function declaration
    OpBuilder builder(module.getBody(), module.getBody()->end());
    auto func = builder.create<func::FuncOp>(module.getLoc(), funcName, funcType);
    func.setPrivate();
    
    return func;
}

// Replace an operation with a blackbox function call
static void replaceOpWithBlackboxCall(Operation *op, ModuleOp module, 
                                           OpBuilder &builder) {
    std::string funcNameStr = getBlackboxFunctionName(op);
    if (funcNameStr.empty()) {
        return;
    }
    StringRef funcName(funcNameStr);
    
    // Get operand types and result type
    SmallVector<Type> operandTypes;
    for (Value operand : op->getOperands()) {
        operandTypes.push_back(operand.getType());
    }
    Type resultType = op->getResult(0).getType();
    
    // Get or create the blackbox function
    func::FuncOp blackboxFunc = getOrCreateBlackboxFunction(module, funcName, 
                                                           resultType, operandTypes);
    
    // Create function call before the operation
    builder.setInsertionPoint(op);
    auto callOp = builder.create<func::CallOp>(op->getLoc(), blackboxFunc, 
                                                op->getOperands());
    
    // Replace all uses of the operation result with the call result
    op->getResult(0).replaceAllUsesWith(callOp.getResult(0));
    
    // Erase the original operation
    op->erase();
}

namespace mlir {
namespace scalehls {

// Check if a function name corresponds to a blackbox function
bool isBlackboxFunctionName(StringRef funcName) {
    return funcName == "addf" || funcName == "subf" || funcName == "mulf" ||
           funcName == "divf" || funcName == "exp_bb" || funcName == "addf_ctrl_chain" ||
           funcName == "mulf_ctrl_chain" || funcName == "subf_ctrl_chain" || funcName == "divf_ctrl_chain" || funcName == "exp_bb_ctrl_chain";
}

void insertBlackboxFunctionCalls(ModuleOp module, func::FuncOp func) {
    func.walk([&](func::CallOp op) {
        auto callee = SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr());
        auto calleeFuncOp = dyn_cast<func::FuncOp>(callee);
        if (calleeFuncOp) {
            insertBlackboxFunctionCalls(module, calleeFuncOp);
        }
    });
    // Collect all operations to replace
    SmallVector<Operation *> opsToReplace;
    func.walk([&](Operation *op) {
        if (getBlackboxFunctionName(op) != "") {
            opsToReplace.push_back(op);
        }
    });
    
    // Replace each operation
    OpBuilder builder(func.getContext());
    for (Operation *op : opsToReplace) {
        replaceOpWithBlackboxCall(op, module, builder);
    }
}

} // namespace scalehls
} // namespace mlir

namespace {
    struct OperationBlackbox : public OperationBlackboxBase<OperationBlackbox> {
      void runOnOperation() override {
        auto module = getOperation();
        for (auto func : module.getOps<func::FuncOp>()) {
            if (hasTopFuncAttr(func)) {
                insertBlackboxFunctionCalls(module, func);
            }
        }
    }
};
} // namespace

std::unique_ptr<Pass> scalehls::createOperationBlackboxPass() {
  return std::make_unique<OperationBlackbox>();
}