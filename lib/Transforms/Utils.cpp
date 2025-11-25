//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "scalehls/Transforms/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "scalehls/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "scalehls"

using namespace mlir;
using namespace scalehls;
using namespace hls;

static void addMemoryOptsPipeline(PassManager &pm) {
  // To factor out the redundant affine operations.
  pm.addPass(createAffineLoopNormalizePass());
  pm.addPass(createSimplifyAffineStructuresPass());
  pm.addPass(createCanonicalizerPass());

  // To simplify the memory accessing. Note that the store forwarding is
  // non-trivial and has a worst case complexity of O(n^2).
  pm.addPass(createSimplifyAffineIfPass());
  pm.addPass(createAffineStoreForwardPass());

  // Generic common sub expression elimination.
  pm.addPass(createCSEPass());
  pm.addPass(createReduceInitialIntervalPass());
}

/// Apply memory optimizations.
bool scalehls::applyMemoryOpts(func::FuncOp func) {
  PassManager optPM(func.getContext(), "func.func");
  addMemoryOptsPipeline(optPM);
  if (failed(optPM.run(func)))
    return false;
  return true;
}

/// Apply optimization strategy to a loop band. The ancestor function is also
/// passed in because the post-tiling optimizations have to take function as
/// target, e.g. canonicalizer and array partition.
bool scalehls::applyOptStrategy(AffineLoopBand &band, func::FuncOp func,
                                FactorList tileList, unsigned targetII) {
  // By design the input function must be the ancestor of the input loop band.
  if (!func->isProperAncestor(band.front()))
    return false;

  //auto startTime = std::chrono::high_resolution_clock::now();
  // Apply loop tiling.
  if (!applyLoopTiling(band, tileList))
    return false;

  //auto endTime = std::chrono::high_resolution_clock::now();
  //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  //LLVM_DEBUG(llvm::dbgs() << "applyLoopTiling() took " << duration.count() << " ms\n";);

  //startTime = std::chrono::high_resolution_clock::now();
  // Apply loop pipelining.
  if (!applyLoopPipelining(band, band.size() - 1, targetII))
    return false;

  //endTime = std::chrono::high_resolution_clock::now();
  //duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  //LLVM_DEBUG(llvm::dbgs() << "applyLoopPipelining() took " << duration.count() << " ms\n";);

  //startTime = std::chrono::high_resolution_clock::now();
  // Apply memory access optimizations and the best suitable array partition
  // strategy to the function.
  applyMemoryOpts(func);
  //endTime = std::chrono::high_resolution_clock::now();
  //duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  //LLVM_DEBUG(llvm::dbgs() << "applyMemoryOpts() took " << duration.count() << " ms\n";);

  //startTime = std::chrono::high_resolution_clock::now();
  applyAutoArrayPartition(func);

  //endTime = std::chrono::high_resolution_clock::now();
  //duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  //LLVM_DEBUG(llvm::dbgs() << "applyAutoArrayPartition() took " << duration.count() << " ms\n";);

  return true;
}

/*static void emitTileListDebugInfo(FactorList tileList) {
  LLVM_DEBUG(llvm::dbgs() << "Tile info: (";
             for (unsigned i = 0, e = tileList.size(); i < e; ++i) {
               llvm::dbgs() << tileList[i];
               if (i != e - 1)
                 llvm::dbgs() << ",";
               else
                 llvm::dbgs() << ")\n";
             });
}*/

/// Apply optimization strategy to a function.
bool scalehls::applyOptStrategy(func::FuncOp func,
                                ArrayRef<FactorList> tileLists,
                                ArrayRef<unsigned> targetIIs) {
  AffineLoopBands bands;
  getLoopBands(func.front(), bands);
  assert(bands.size() == tileLists.size() && bands.size() == targetIIs.size() &&
         "unexpected size of tile lists or target IIs");

  // Apply loop tiling to all loop bands.
  for (unsigned i = 0, e = bands.size(); i < e; ++i)
  {
    //LLVM_DEBUG(llvm::dbgs() << "Applying loop tiling to band " << i << "\n";);
    //emitTileListDebugInfo(tileLists[i]);
    if (!applyLoopTiling(bands[i], tileLists[i]))
      return false;
  }

  for (unsigned i = 0, e = bands.size(); i < e; ++i)
  {
    //LLVM_DEBUG(llvm::dbgs() << "Applying loop pipelining to band " << i << "\n";);
    if (!applyLoopPipelining(bands[i], bands[i].size() - 1, targetIIs[i]))
      return false;
  }

  // Apply memory access optimizations and the best suitable array partition
  // strategy to the function.
  applyMemoryOpts(func);
  applyAutoArrayPartition(func);
  return true;
}
