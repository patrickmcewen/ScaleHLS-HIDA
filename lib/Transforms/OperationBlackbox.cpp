#include "scalehls/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace scalehls;
using namespace hls;

// Get the function name for a given arithmetic operation type
static StringRef getBlackboxFunctionName(Operation *op) {
    if (isa<arith::AddFOp>(op)) return "addf";
    if (isa<arith::SubFOp>(op)) return "subf";
    if (isa<arith::MulFOp>(op)) return "mulf";
    if (isa<arith::DivFOp>(op)) return "divf";
    if (isa<arith::RemFOp>(op)) return "remf";
    if (isa<arith::AddIOp>(op)) return "addi";
    if (isa<arith::SubIOp>(op)) return "subi";
    if (isa<arith::MulIOp>(op)) return "muli";
    if (isa<arith::DivSIOp>(op)) return "divsi";
    if (isa<arith::DivUIOp>(op)) return "divui";
    if (isa<arith::RemSIOp>(op)) return "remsi";
    if (isa<arith::RemUIOp>(op)) return "remui";
    if (isa<arith::MaxFOp>(op)) return "maxf";
    if (isa<arith::MinFOp>(op)) return "minf";
    if (isa<arith::MaxSIOp>(op)) return "maxsi";
    if (isa<arith::MinSIOp>(op)) return "minsi";
    if (isa<arith::MaxUIOp>(op)) return "maxui";
    if (isa<arith::MinUIOp>(op)) return "minui";
    if (isa<arith::NegFOp>(op)) return "negf";
    return "";
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

// Replace an arithmetic operation with a blackbox function call
static void replaceArithOpWithBlackboxCall(Operation *op, ModuleOp module, 
                                           OpBuilder &builder) {
    StringRef funcName = getBlackboxFunctionName(op);
    if (funcName.empty()) {
        return;
    }
    
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
           funcName == "divf" || funcName == "remf" || funcName == "addi" ||
           funcName == "subi" || funcName == "muli" || funcName == "divsi" ||
           funcName == "divui" || funcName == "remsi" || funcName == "remui" ||
           funcName == "maxf" || funcName == "minf" || funcName == "maxsi" ||
           funcName == "minsi" || funcName == "maxui" || funcName == "minui" ||
           funcName == "negf";
}

void insertBlackboxFunctionCalls(ModuleOp module, func::FuncOp func) {
    func.walk([&](func::CallOp op) {
        auto callee = SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr());
        auto calleeFuncOp = dyn_cast<func::FuncOp>(callee);
        if (calleeFuncOp) {
            insertBlackboxFunctionCalls(module, calleeFuncOp);
        }
    });
    // Collect all arithmetic operations
    SmallVector<Operation *> arithOps;
    func.walk([&](Operation *op) {
        if (getBlackboxFunctionName(op) != "") {
            arithOps.push_back(op);
        }
    });
    
    // Replace each arithmetic operation
    OpBuilder builder(func.getContext());
    for (Operation *op : arithOps) {
        replaceArithOpWithBlackboxCall(op, module, builder);
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