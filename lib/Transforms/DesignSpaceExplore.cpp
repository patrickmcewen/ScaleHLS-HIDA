//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Support/FileUtilities.h"
#include "scalehls/Transforms/Explorer.h"
#include "scalehls/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include <numeric>
#include <vector>
#include <chrono>
// #include <pthread.h>

#define DEBUG_TYPE "scalehls"

using namespace mlir;
using namespace scalehls;

/// Debug utility function to dump the MLIR representation of a function.
/// This can be called at any point during execution to inspect the function state.
/// 
/// Usage examples:
///   // Dump to debug stream (stderr) with a label
///   dumpFuncMLIR(func, "After loop tiling");
///   
///   // Dump to a file
///   dumpFuncMLIR(func, "After optimization", true, "/tmp/debug.mlir");
///   
///   // Simple dump to debug stream
///   dumpFuncMLIR(func);
/// 
/// \param func The function to dump
/// \param label Optional label/message to include in the output
/// \param dumpToFile If true, dump to a file instead of stderr
/// \param filePath Optional file path (only used if dumpToFile is true)
/// \return true if successful, false otherwise
static bool dumpFuncMLIR(func::FuncOp func, StringRef label = "", bool stopAfter = false,
                         bool dumpToFile = false, StringRef filePath = "") {
  if (dumpToFile && !filePath.empty()) {
    // Dump to file
    std::string errorMessage;
    auto outputFile = mlir::openOutputFile(filePath, &errorMessage);
    if (!outputFile) {
      llvm::errs() << "ERROR: Failed to dump function to file '" << filePath 
                   << "': " << errorMessage << "\n";
      return false;
    }
    
    auto &os = outputFile->os();
    if (!label.empty())
      os << "// " << label << "\n";
    
    // Dump just the function (not the entire module) for cleaner output
    // If the function needs module context, the caller should pass the module
    os << func << "\n";
    
    outputFile->keep();
    llvm::dbgs() << "[DEBUG] Dumped function '" << func.getName() 
                 << "' to file: " << filePath;
    if (!label.empty())
      llvm::dbgs() << " (label: " << label << ")";
    llvm::dbgs() << "\n";
    return true;
  } else {
    // Dump to stderr/debug stream
    llvm::dbgs() << "// ========================================\n";
    if (!label.empty())
      llvm::dbgs() << "// " << label << "\n";
    llvm::dbgs() << "// Function: " << func.getName() << "\n";
    llvm::dbgs() << "// ========================================\n";
    
    // Dump just the function
    llvm::dbgs() << func << "\n";
    
    llvm::dbgs() << "// ========================================\n\n";
    if (stopAfter) {
      exit(0);
    }
    return true;
  }
}

/// Helper function to clone a function by cloning its module first.
/// This preserves the symbol table hierarchy, allowing symbol resolution to work
/// correctly even for cloned functions that call other functions.
/// 
/// Returns the cloned function. The cloned module is kept alive because the
/// function is part of the module's region, so as long as we hold a reference
/// to the function, the module stays alive.
static std::pair<func::FuncOp, ModuleOp> cloneFunctionWithModule(func::FuncOp func) {
  // Get the module containing the function
  ModuleOp module = func->getParentOfType<ModuleOp>();
  
  // Clone the entire module to preserve symbol table hierarchy.
  // This ensures that all function calls within the cloned function can
  // resolve their callees using the symbol table, since the cloned function
  // and all its callees are in the same cloned module.
  auto clonedModule = cast<ModuleOp>(module->clone());
  
  // Find and return the function with the same name in the cloned module
  auto funcName = func.getName();
  auto clonedFunc = clonedModule.lookupSymbol<func::FuncOp>(funcName);
  if (!clonedFunc) {
    llvm::errs() << "ERROR: Could not find function '" << funcName 
                 << "' in cloned module. This should not happen.\n";
    assert(false && "Function not found in cloned module");
  }
  
  // Note: The clonedModule will stay alive as long as clonedFunc is alive
  // because clonedFunc is part of clonedModule's region. We don't need to
  // explicitly store the module reference.
  
  return std::make_pair(clonedFunc, clonedModule);
}

/// Update paretoPoints to remove design points that are not pareto frontiers.
/// Optionally filter by resource constraints.
template <typename ContainerType>
static void updateParetoPoints(ContainerType &paretoPoints,
                               unsigned maxDspNum = UINT_MAX,
                               unsigned maxBramNum = UINT_MAX) {
  using DesignPointType = typename ContainerType::value_type;
  //LLVM_DEBUG(llvm::dbgs() << "Updating pareto points with maxDspNum: " << maxDspNum << " and maxBramNum: " << maxBramNum << "\n";);
  //LLVM_DEBUG(llvm::dbgs() << "Number of pareto points before filtering: " << paretoPoints.size() << "\n";);
  //DEBUG CONTROL (uncomment to use): 
  /*unsigned maxDspSoFar = UINT_MAX;
  // First, filter by resource constraints if provided
  if (maxDspNum != UINT_MAX || maxBramNum != UINT_MAX) {
    std::vector<DesignPointType> filteredPoints;
    for (auto &point : paretoPoints) {
      bool withinConstraints = true;
      //LLVM_DEBUG(llvm::dbgs() << "Checking point with dspNum: " << point.dspNum << " and maxDspNum: " << maxDspNum << "\n";);
      if (maxDspNum != UINT_MAX && point.dspNum > maxDspNum) {
        withinConstraints = false;
      }
      // Note: DesignPointType only has dspNum, not bramNum
      // BRAM checking would need to be done separately where we have the resource object
      
      // DEBUG CONTROL
      if (filteredPoints.empty()) {
        filteredPoints.push_back(point);
      } /*else if (point.dspNum < maxDspSoFar) {
        filteredPoints[0] = point;
        maxDspSoFar = point.dspNum;
      } */
      // END DEBUG CONTROL

      // COMMENT THIS BIT OUT AND UNCOMMENT THE ABOVE TO ALWAYS GET THE MINIMUM DSP POINT
      /*if (withinConstraints) {
        filteredPoints.push_back(point);
      }*//*
    }
    //LLVM_DEBUG(llvm::dbgs() << "Number of pareto points in filtered points: " << filteredPoints.size() << "\n";);
    paretoPoints.assign(filteredPoints.begin(), filteredPoints.end());
  }*/

  //LLVM_DEBUG(llvm::dbgs() << "Number of pareto points after filtering: " << paretoPoints.size() << "\n";);

  if (paretoPoints.empty())
    return;

  // Sort the pareto points with in an ascending order of latency and the an
  // ascending order of dsp number.
  auto latencyThenDspNum = [&](const DesignPointType &a,
                               const DesignPointType &b) {
    return (a.latency < b.latency ||
            (a.latency == b.latency && a.dspNum < b.dspNum));
  };
  llvm::sort(paretoPoints, latencyThenDspNum);

  // Find pareto frontiers. After the sorting, the first design point must be a
  // pareto point.
  auto paretoPoint = paretoPoints[0];
  auto paretoLatency = paretoPoint.latency;
  auto paretoDspNum = paretoPoint.dspNum;
  std::vector<DesignPointType> frontiers;

  for (auto point : paretoPoints) {
    auto tmpLatency = point.latency;
    auto tmpDspNum = point.dspNum;

    if (tmpDspNum < paretoDspNum) {
      frontiers.push_back(point);

      paretoPoint = point;
      paretoLatency = tmpLatency;
      paretoDspNum = tmpDspNum;

    } else if (frontiers.empty())
      frontiers.push_back(point);
  }

  LLVM_DEBUG(llvm::dbgs() << "updated pareto points, number of points: " << frontiers.size() << "\n";);

  paretoPoints.assign(frontiers.begin(), frontiers.end());
}

//===----------------------------------------------------------------------===//
// LoopDesignSpace Class Definition
//===----------------------------------------------------------------------===//

static void emitTileListDebugInfo(FactorList tileList) {
  LLVM_DEBUG(llvm::dbgs() << "Tile info: (";
             for (unsigned i = 0, e = tileList.size(); i < e; ++i) {
               llvm::dbgs() << tileList[i];
               if (i != e - 1)
                 llvm::dbgs() << ",";
               else
                 llvm::dbgs() << ")";
             });
}

LoopDesignSpace::LoopDesignSpace(func::FuncOp func, ModuleOp parentModule, AffineLoopBand &band,
                                 ScaleHLSEstimator &estimator,
                                 unsigned maxDspNum, unsigned maxExplParallel,
                                 unsigned maxLoopParallel, bool directiveOnly)
    : func(func), parentModule(parentModule), band(band), estimator(estimator), maxDspNum(maxDspNum) {
  // Initialize tile vector related members.
  validTileConfigNum = 1;
  for (auto loop : band) {
    auto optionalTripCount = getConstantTripCount(loop);
    if (!optionalTripCount)
      loop.emitError("has variable loop bound");

    unsigned tripCount = optionalTripCount.value();
    tripCountList.push_back(tripCount);

    std::vector<unsigned> validSizes;
    unsigned size = 1;
    while (size <= std::min(tripCount, maxLoopParallel)) {
      // Push back the current size.
      validSizes.push_back(size);

      // Find the next possible size.
      ++size;
      while (size <= tripCount && tripCount % size != 0)
        ++size;
    }

    validTileSizesList.push_back(validSizes);
    validTileConfigNum *= validSizes.size();
  }

  // The last design point (all loops are fully unrolled) is removed.
  --validTileConfigNum;

  for (TileConfig config = 0; config < validTileConfigNum; ++config) {
    auto tileList = getTileList(config);

    // If the overall parallelism is out of bound, continue to next config.
    auto parallel = std::accumulate(tileList.begin(), tileList.end(),
                                    (unsigned)1, std::multiplies<unsigned>());
    if (parallel > maxExplParallel)
      continue;

    // In only directive opt should be applied, once one loop is unrolled, all
    // innter loops should be fully unrolled.
    if (directiveOnly) {
      bool mustFullyUnroll = false;
      bool isInvalid = false;
      unsigned i = 0;

      for (auto tile : tileList) {
        if (mustFullyUnroll && tile != tripCountList[i])
          isInvalid = true;
        if (tile != 1)
          mustFullyUnroll = true;
        ++i;
      }

      if (isInvalid)
        continue;
    }

    unestimatedTileConfigs.insert(config);
  }
}

/// Return the actual tile vector given a tile config.
FactorList LoopDesignSpace::getTileList(TileConfig config) {
  assert(config < validTileConfigNum && "invalid tile config");

  FactorList tileList;
  unsigned factor = 1;
  for (auto validSizes : validTileSizesList) {
    auto idx = config / factor % validSizes.size();
    factor *= validSizes.size();

    auto size = validSizes[idx];
    tileList.push_back(size);
  }
  return tileList;
}

/// Return the corresponding tile config given a tile list.
TileConfig LoopDesignSpace::getTileConfig(FactorList tileList) {
  assert(tileList.size() == validTileSizesList.size() && "invalid tile list");

  TileConfig config = 0;
  unsigned factor = 1;
  for (unsigned i = 0, e = tileList.size(); i < e; ++i) {
    auto tile = tileList[i];
    auto validSizes = validTileSizesList[i];

    auto idx = llvm::find(validSizes, tile) - validSizes.begin();

    assert(idx >= 0 && idx < (long)validSizes.size() && "invalid tile list");

    config += factor * idx;
    factor *= validSizes.size();
  }

  return config;
}

/// Calculate the Euclid distance of config a and config b.
float LoopDesignSpace::getTileConfigDistance(TileConfig configA,
                                             TileConfig configB) {
  assert(configA < validTileConfigNum && configB < validTileConfigNum &&
         "invalid tile config");

  int64_t distanceSquare = 0;
  unsigned factor = 1;
  for (auto validSizes : validTileSizesList) {
    int64_t idxA = configA / factor % validSizes.size();
    int64_t idxB = configB / factor % validSizes.size();
    factor *= validSizes.size();

    auto idxDistance = idxA - idxB;
    distanceSquare += idxDistance * idxDistance;
  }

  return sqrtf(distanceSquare);
}

/// Evaluate all design points under the given tile config.
bool LoopDesignSpace::evaluateTileConfig(TileConfig config) {
  // If the current tile config is already estimated, return false.
  if (!unestimatedTileConfigs.count(config))
    return false;

  // Annotate the current tile config as estimated.
  unestimatedTileConfigs.erase(config);

  // Clone a temporary loop band by cloning the outermost loop.
  auto outerLoop = band.front();
  auto tmpOuterLoop = outerLoop.clone();
  AffineLoopBand tmpBand;
  getLoopBandFromOutermost(tmpOuterLoop, tmpBand);

  // Insert the clone loop band to the front of the original band for the
  // convenience of the estimation.
  auto builder = OpBuilder(func);
  builder.setInsertionPoint(outerLoop);
  builder.insert(tmpOuterLoop);

  // Apply the tile config and estimate the loop band.
  auto tileList = getTileList(config);
  //emitTileListDebugInfo(tileList);

  // Calculate the total iteration number.
  unsigned iterNum = 1;
  for (unsigned i = 0, e = tileList.size(); i < e; ++i)
    iterNum *= tripCountList[i] / tileList[i];

  // We always don't fully unroll all loops in the loop band.
  if (iterNum == 1)
    return false;

  // Apply the current tiling config and start the estimation. Note that after
  // optimization, tmpBand is optimized in place and becomes a new loop band.
  if (!applyOptStrategy(tmpBand, func, tileList, (unsigned)1))
    return false;
  tmpOuterLoop = tmpBand.front();
  estimator.estimateLoop(tmpOuterLoop, func);

  // Fetch latency and resource utilization.
  auto tmpInnerLoop = tmpBand.back();
  auto info = getLoopInfo(tmpInnerLoop);
  auto resource = getResource(tmpOuterLoop);
  //LLVM_DEBUG(llvm::dbgs() << "The loop info of the inner loop is " << info << "\n";);
  //LLVM_DEBUG(llvm::dbgs() << "The resource of the outer loop is " << resource << "\n";);
  //if (func.getName() == "forward_node0") {
  //  dumpFuncMLIR(func, "forward_node0_after_opt", true);
  //}
  assert(info && resource && "loop info or resource is not estimated");
  auto totalDsp = resource.getDsp() * info.getMinII();

  // Improve target II until II is equal to iteration latency. Note that when II
  // equal to iteration latency, the pipeline pragma is similar to a region
  // fully unroll pragma which unrolls all contained loops.
  for (auto tmpII = info.getMinII(); tmpII <= info.getIterLatency(); ++tmpII) {
    auto tmpDspNum = totalDsp / tmpII + 1;
    auto tmpLatency = info.getIterLatency() + tmpII * (iterNum - 1) + 2;
    auto point = LoopDesignPoint(tmpLatency, tmpDspNum, config, tmpII);

    allPoints.push_back(point);
    if (tmpDspNum <= maxDspNum)
      paretoPoints.push_back(point);
  }

  // Erase the temporary loop band.
  tmpOuterLoop.erase();
  return true;
}

/// Initialize the design space.
void LoopDesignSpace::initializeLoopDesignSpace(unsigned maxInitParallel) {
  //LLVM_DEBUG(llvm::dbgs() << "Initialize the loop design space...\n";);

  for (TileConfig config = 0; config < validTileConfigNum; ++config) {
    auto tileList = getTileList(config);

    // We only evaluate the design points whose overall parallel is smaller
    // than the maxInitParallel to improve the efficiency.
    auto parallel = std::accumulate(tileList.begin(), tileList.end(),
                                    (unsigned)1, std::multiplies<unsigned>());
    //LLVM_DEBUG(llvm::dbgs() << "The parallel of the tile list " << config << " is " << parallel << "\n";);
    if (parallel <= maxInitParallel) {// || config == parallelConfig)
      // Time the evaluateTileConfig function
      auto startTime = std::chrono::high_resolution_clock::now();
      evaluateTileConfig(config);
      auto endTime = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
      LLVM_DEBUG(llvm::dbgs() << "evaluateTileConfig() on tile list ";);
      emitTileListDebugInfo(tileList);
      LLVM_DEBUG(llvm::dbgs() << " took " << duration.count() << " ms, evaluated from initializeLoopDesignSpace()\n";);
    }
  }

  //LLVM_DEBUG(llvm::dbgs() << "\n\n");
  updateParetoPoints(paretoPoints, maxDspNum);
}

/// Dump pareto and non-pareto points which have been evaluated in the design
/// space to a csv output file.
void LoopDesignSpace::dumpLoopDesignSpace(StringRef csvFilePath) {
  std::string errorMessage;
  auto csvFile = mlir::openOutputFile(csvFilePath, &errorMessage);
  if (!csvFile)
    return;
  auto &os = csvFile->os();

  // Print header row.
  for (unsigned i = 0; i < tripCountList.size(); ++i)
    os << "l" << i << ",";
  os << "ii,cycle,dsp,type\n";

  // Print pareto design points.
  for (auto &point : paretoPoints) {
    for (auto size : getTileList(point.tileConfig))
      os << size << ",";
    os << point.targetII << "," << point.latency << "," << point.dspNum
       << ",pareto\n";
  }

  // Print all design points.
  for (auto &point : allPoints) {
    for (auto size : getTileList(point.tileConfig))
      os << size << ",";
    os << point.targetII << "," << point.latency << "," << point.dspNum
       << ",non-pareto\n";
  }

  csvFile->keep();
  //LLVM_DEBUG(llvm::dbgs() << "Loop design space is dumped to file \""
  //                       << csvFilePath << "\".\n\n");
}

/// Get a random tile config which is one of the closest neighbors of "point".
Optional<TileConfig>
LoopDesignSpace::getRandomClosestNeighbor(LoopDesignPoint point,
                                          float maxDistance) {
  // Traverse all unestimated tile configs and collect all neighbors.
  std::vector<std::pair<float, TileConfig>> candidateConfigs;
  for (auto config : unestimatedTileConfigs) {
    auto distance = getTileConfigDistance(point.tileConfig, config);
    if (distance <= maxDistance)
      candidateConfigs.push_back(
          std::pair<float, TileConfig>(distance, config));
  }

  if (candidateConfigs.empty())
    return Optional<TileConfig>();

  // Sort candidate configs and collect the closest points.
  llvm::sort(candidateConfigs);
  std::vector<TileConfig> closestConfigs;
  float minDistance = maxDistance;

  for (auto configPair : candidateConfigs) {
    if (configPair.first <= minDistance) {
      closestConfigs.push_back(configPair.second);
      minDistance = configPair.first;
    } else
      break;
  }

  // Randomly pick one as the return point.
  std::srand(time(0));
  llvm::shuffle(closestConfigs.begin(), closestConfigs.end(),
                []() { return std::rand(); });

  return closestConfigs.front();
}

void LoopDesignSpace::exploreLoopDesignSpace(unsigned maxIterNum,
                                             float maxDistance) {
  LLVM_DEBUG(llvm::dbgs() << "Explore the loop design space...\n";);

  // Exploration loop of the dse.
  for (unsigned i = 0; i < maxIterNum; ++i) {
    std::srand(time(0));
    llvm::shuffle(paretoPoints.begin(), paretoPoints.end(),
                  []() { return std::rand(); });

    bool foundValidNeighbor = false;
    for (auto &point : paretoPoints) {
      if (!point.isActive)
        continue;

      auto closestNeighbor = getRandomClosestNeighbor(point, maxDistance);
      if (!closestNeighbor) {
        point.isActive = false;
        continue;
      }

      foundValidNeighbor = true;
      auto config = closestNeighbor.value();
      auto tileList = getTileList(config);

      // Time the evaluateTileConfig function
      auto startTime = std::chrono::high_resolution_clock::now();
      evaluateTileConfig(config);
      auto endTime = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
      LLVM_DEBUG(llvm::dbgs() << "evaluateTileConfig()  on tile list ";);
      emitTileListDebugInfo(tileList);
      LLVM_DEBUG(llvm::dbgs() << " took " << duration.count() << " ms, evaluated from exploreLoopDesignSpace()\n";);
      break;
    }

    // Early termination if no valid neighbor is found.
    if (!foundValidNeighbor)
      break;

    // Update pareto points after each dse iteration.
    updateParetoPoints(paretoPoints, maxDspNum);
  }
  LLVM_DEBUG(llvm::dbgs() << "\n\n";);
}

//===----------------------------------------------------------------------===//
// FuncDesignSpace Class Definition
//===----------------------------------------------------------------------===//

void FuncDesignSpace::dumpFuncDesignSpace(StringRef csvFilePath) {
  std::string errorMessage;
  auto csvFile = mlir::openOutputFile(csvFilePath, &errorMessage);
  if (!csvFile)
    return;
  auto &os = csvFile->os();

  // Print header row.
  for (unsigned i = 0, ei = loopDesignSpaces.size(); i < ei; ++i) {
    auto &loopSpace = loopDesignSpaces[i];

    for (unsigned j = 0, ej = loopSpace.tripCountList.size(); j < ej; ++j)
      os << "b" << i << "l" << j << ",";
    os << "b" << i << "ii,";
  }
  os << "cycle,dsp,type\n";

  // Print pareto design points.
  //LLVM_DEBUG(llvm::dbgs() << func.getName() << " Function design space has " << loopDesignSpaces.size() << " loop design spaces.\n";);
  //LLVM_DEBUG(llvm::dbgs() << func.getName() << " Function design space has " << paretoPoints.size() << " pareto points.\n";);
  for (auto &funcPoint : paretoPoints) {
    //LLVM_DEBUG(llvm::dbgs() << func.getName() << " Function design has " << funcPoint.loopDesignPoints.size() << " loop design points.\n";);
    for (unsigned i = 0, e = loopDesignSpaces.size(); i < e; ++i) {
      auto &loopPoint = funcPoint.loopDesignPoints[i];
      auto &loopSpace = loopDesignSpaces[i];

      for (auto size : loopSpace.getTileList(loopPoint.tileConfig))
        os << size << ",";
      os << loopPoint.targetII << ",";
    }
    os << funcPoint.latency << "," << funcPoint.dspNum << ",pareto\n";
  }

  csvFile->keep();
  //LLVM_DEBUG(llvm::dbgs() << "Function design space is dumped to file \""
  //                        << csvFilePath << "\".\n\n");
}

void cleanUpClonedModulesAndFunctions(FuncDesignSpace &funcDesignSpace) {
  for (unsigned ii = 0; ii < funcDesignSpace.loopDesignSpaces.size(); ++ii) {
    auto &loopDesignSpace = funcDesignSpace.loopDesignSpaces[ii];
    auto parentModule = loopDesignSpace.func->getParentOfType<ModuleOp>();
    if (parentModule) {
      LLVM_DEBUG(llvm::dbgs() << "Destroying cloned module from loop design space " << ii << "\n";);
      parentModule->destroy();
    }
  }
  auto parentModule = funcDesignSpace.func->getParentOfType<ModuleOp>();
  if (parentModule) {
    LLVM_DEBUG(llvm::dbgs() << "Destroying cloned module from func design space\n";);
    parentModule->destroy();
  }
}

void FuncDesignSpace::combLoopDesignSpaces() {
  //LLVM_DEBUG(llvm::dbgs() << "Combine the loop design spaces...\n";);

  // Check if there are any loop design spaces to combine.
  if (loopDesignSpaces.empty()) {
    //LLVM_DEBUG(llvm::dbgs() << "[DSE] WARNING: No loop bands found in function '" 
    //                 << func.getName() << "'. Will create a default function design point.\n";);
    // Time the estimateFunc function
    auto startTime = std::chrono::high_resolution_clock::now();
    estimator.estimateFunc(func);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    LLVM_DEBUG(llvm::dbgs() << "estimateFunc() took " << duration.count() << " ms, evaluated from combLoopDesignSpaces()\n";);
    auto latency = getTiming(func).getLatency();
    auto dspNum = getResource(func).getDsp();
    //dumpFuncMLIR(func, "default_func_design_point", false);
    auto funcPoint = FuncDesignPoint(latency, dspNum);
    paretoPoints.push_back(funcPoint);
    return;
  }

  // Initialize the function design space with the first loop design space.
  auto &firstLoopSpace = loopDesignSpaces[0];
  for (auto &loopPoint : firstLoopSpace.paretoPoints) {
    // Annotate the first loop.
    auto loop = targetLoops[0];
    //LLVM_DEBUG(llvm::dbgs() << "Annotating the first loop of function " << func.getName() << " with latency " << loopPoint.latency << " and dsp num " << loopPoint.dspNum << "\n";);
    setTiming(loop, -1, -1, loopPoint.latency, -1);
    setResource(loop, -1, loopPoint.dspNum, -1);

    //dumpFuncMLIR(func, "pre_estimated_func_design_point", false);
    //auto iterLatency = getLoopInfo(loop).getIterLatency();
    //LLVM_DEBUG(llvm::dbgs() << "Iter latency of the first loop of function " << func.getName() << " is " << iterLatency << "\n";);

    // Estimate the function and generate a new function design point.
    estimator.estimateFunc(func);
    auto latency = getTiming(func).getLatency();
    auto dspNum = getResource(func).getDsp();
    auto funcPoint = FuncDesignPoint(latency, dspNum, loopPoint);

    paretoPoints.push_back(funcPoint);
  }
  // If no loop design points are found, create a default function design point.
  if (paretoPoints.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No loop design points found, creating a default function design point.\n";);
    estimator.estimateFunc(func);
    auto latency = getTiming(func).getLatency();
    auto dspNum = getResource(func).getDsp();
    auto funcPoint = FuncDesignPoint(latency, dspNum);
    paretoPoints.push_back(funcPoint);
  }


  updateParetoPoints(paretoPoints, maxDspNum);
  LLVM_DEBUG(llvm::dbgs() << "Iteration 0 loop design space pareto points number: "
                          << paretoPoints.size() << "\n";);

  // Combine other loop design spaces to the function design space one by one.
  for (unsigned i = 1, e = loopDesignSpaces.size(); i < e; ++i) {
    std::vector<FuncDesignPoint> newParetoPoints;
    auto &loopSpace = loopDesignSpaces[i];

    // Traverse all function design points.
    for (auto &funcPoint : paretoPoints) {
      // Annotate latency and dsp to all loops that are already included in the
      // function point, they are static for all design points of the new loop.
      for (unsigned ii = 0; ii < i; ++ii) {
        auto &oldLoopPoint = funcPoint.loopDesignPoints[ii];
        auto oldLoop = targetLoops[ii];
        //LLVM_DEBUG(llvm::dbgs() << "Annotating the loop " << ii << " of function " << func.getName() << " with latency " << oldLoopPoint.latency << " and dsp num " << oldLoopPoint.dspNum << "\n";);
        setTiming(oldLoop, -1, -1, oldLoopPoint.latency, -1);
        setResource(oldLoop, -1, oldLoopPoint.dspNum, -1);
      }

      // Traverse all design points of the NEW loop.
      for (auto &loopPoint : loopSpace.paretoPoints) {
        // Annotate the new loop,
        auto loop = targetLoops[i];
        //LLVM_DEBUG(llvm::dbgs() << "Annotating the loop " << i << " of function " << func.getName() << " with latency " << loopPoint.latency << " and dsp num " << loopPoint.dspNum << "\n";);
        setTiming(loop, -1, -1, loopPoint.latency, -1);
        setResource(loop, -1, loopPoint.dspNum, -1);

        // Estimate the function and generate a new function design point.
        auto loopPoints = funcPoint.loopDesignPoints;
        loopPoints.push_back(loopPoint);

        estimator.estimateFunc(func);
        auto latency = getTiming(func).getLatency();
        auto dspNum = getResource(func).getDsp();
        auto newFuncPoint = FuncDesignPoint(latency, dspNum, loopPoints);

        newParetoPoints.push_back(newFuncPoint);
      }
    }

    // Update pareto points after each combination.
    updateParetoPoints(newParetoPoints, maxDspNum);
    paretoPoints = newParetoPoints;
    LLVM_DEBUG(llvm::dbgs() << "Iteration " << i << " loop design space pareto points number: "
                            << paretoPoints.size() << "\n";);
  }
  LLVM_DEBUG(llvm::dbgs() << "\n";);
}

bool FuncDesignSpace::exportParetoDesigns(unsigned outputNum,
                                          StringRef outputRootPath) {
  unsigned paretoNum = paretoPoints.size();
  auto sampleStep = std::max(paretoNum / outputNum, (unsigned)1);

  // Traverse all detected pareto points.
  unsigned sampleIndex = 0;
  for (auto &funcPoint : paretoPoints) {
    // Only export sampled points.
    if (sampleIndex % sampleStep == 0) {
      std::vector<FactorList> tileLists;
      std::vector<unsigned> targetIIs;

      for (unsigned i = 0; i < loopDesignSpaces.size(); ++i) {
        auto &loopSpace = loopDesignSpaces[i];
        auto &loopPoint = funcPoint.loopDesignPoints[i];
        auto tileList = loopSpace.getTileList(loopPoint.tileConfig);
        auto targetII = loopPoint.targetII;

        tileLists.push_back(tileList);
        targetIIs.push_back(targetII);
      }

      //dumpFuncMLIR(func, "pre_optimized_func_design_point", false);

      // Clone a new function (with its module) and apply optimization.
      auto [tmpFunc, tmpModule] = cloneFunctionWithModule(func);
      //dumpFuncMLIR(tmpFunc, "before_optimized_func_design_point", false);
      if (!applyOptStrategy(tmpFunc, tileLists, targetIIs))
        return false;
      estimator.estimateFunc(tmpFunc);
      //dumpFuncMLIR(tmpFunc, "after_optimized_func_design_point", false);

      //dumpFuncMLIR(func, "post_optimized_func_design_point", false);

      // Parse a new output file.
      auto outputFilePath = outputRootPath.str() + "/function_output/" + func.getName().str() +
                            "_pareto_" + std::to_string(sampleIndex) + ".mlir";

      std::string errorMessage;
      auto outputFile = mlir::openOutputFile(outputFilePath, &errorMessage);
      if (!outputFile)
        return false;

      auto &os = outputFile->os();
      os << tmpFunc << "\n";
      outputFile->keep();

      tmpModule->destroy();
    }
    ++sampleIndex;
  }

  LLVM_DEBUG(
      llvm::dbgs() << "Sampled pareto points MLIR files are exported to path \""
                   << outputRootPath << "\".\n\n");
  return true;
}

void dumpHierFuncDesignPoints(StringRef funcName, std::vector<HierFuncDesignPoint> &hierFuncDesignPoints, HierFuncDesignSpace &hierFuncDesignSpace) {
  LLVM_DEBUG(llvm::dbgs() << "Dumping " << hierFuncDesignPoints.size() << " hierarchical function design points for function " << funcName << "...\n";);
  for (auto &hierFuncDesignPoint : hierFuncDesignPoints) {
    auto latency = hierFuncDesignPoint.latency;
    auto dspNum = hierFuncDesignPoint.dspNum;
    LLVM_DEBUG(llvm::dbgs() << "A hierarchical function design point is found for function " << funcName << " with latency " << latency << " and dsp num " << dspNum << "\n";);
    LLVM_DEBUG(llvm::dbgs() << "The number of sub hierarchical function design points are " << hierFuncDesignPoint.subHierFuncDesignPoints.size() << "\n";);
    for (unsigned i = 0; i < hierFuncDesignPoint.subHierFuncDesignPoints.size(); ++i) {
      auto &subHierFuncDesignPoint = hierFuncDesignPoint.subHierFuncDesignPoints[i];
      auto &subHierFuncSpace = hierFuncDesignSpace.subHierFuncDesignSpaces[i];
      auto subFuncLatency = subHierFuncDesignPoint.latency;
      auto subFuncDspNum = subHierFuncDesignPoint.dspNum;
      LLVM_DEBUG(llvm::dbgs() << "A sub function design point is found for function " << subHierFuncSpace.func.getName() << " with latency " << subFuncLatency << " and dsp num " << subFuncDspNum << "\n";);
    }
    LLVM_DEBUG(llvm::dbgs() << "\n";);
  }
  LLVM_DEBUG(llvm::dbgs() << "\n";);
}

//===----------------------------------------------------------------------===//
// HierFuncDesignSpace Class Definition
//===----------------------------------------------------------------------===//

func::CallOp getSubFuncCallOp(func::FuncOp func, StringRef subFuncName) {
  func::CallOp foundCallOp = nullptr;
  func.walk([&](func::CallOp callOp) {
    if (callOp.getCallee() == subFuncName) {
      foundCallOp = callOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  if (!foundCallOp) {
    llvm::errs() << "[DSE] ERROR: Cannot find sub function call op for function '"
                 << func.getName() << "' and sub function '" << subFuncName << "'\n";
    assert(false && "Cannot find sub function call op");
  }
  return foundCallOp;
}

void setTimingAndResourceSubFunc(func::FuncOp func, func::FuncOp subFunc, int64_t latency, int64_t dspNum) {
  //LLVM_DEBUG(llvm::dbgs() << "Annotating the sub function " << subFunc.getName() << " of function " << func.getName() << " with latency " << latency << " and dsp num " << dspNum << "\n";);
  //setTiming(subFunc, -1, -1, latency, -1);
  //setResource(subFunc, -1, dspNum, -1);
  auto callOpSubFunc = getSubFuncCallOp(func, subFunc.getName());

  setTiming(callOpSubFunc, -1, -1, latency, -1);
  setResource(callOpSubFunc, -1, dspNum, -1);
  callOpSubFunc->setAttr("no_touch", BoolAttr::get(func.getContext(), true));
}

func::FuncOp HierFuncDesignSpace::getSubFunc(func::FuncOp func, StringRef subFuncName) {
  auto subFuncNameAttr = StringAttr::get(func.getContext(), subFuncName);
  auto subFuncActual = SymbolTable::lookupNearestSymbolFrom(func, subFuncNameAttr);
  if (!subFuncActual) {
    llvm::errs() << "[DSE] ERROR: Cannot find sub function '"
                 << subFuncName << "' in function '"
                 << func.getName() << "'\n";
    assert(false && "Cannot find sub function");
  }
  return dyn_cast<func::FuncOp>(subFuncActual);
}

func::FuncOp HierFuncDesignSpace::getSubFuncFromModule(ModuleOp module, StringRef subFuncName) {
  auto subFuncActual = module.lookupSymbol(subFuncName);
  if (!subFuncActual) {
    llvm::errs() << "[DSE] ERROR: Cannot find sub function '"
                 << subFuncName << "' in module\n";
    assert(false && "Cannot find sub function");
  }
  auto subFunc = dyn_cast<func::FuncOp>(subFuncActual);
  if (!subFunc) {
    llvm::errs() << "[DSE] ERROR: Symbol '" << subFuncName 
                 << "' is not a function operation\n";
    assert(false && "Symbol is not a function operation");
  }
  return subFunc;
}

HierFuncDesignPoint createHierFuncDesignPoint(FuncDesignPoint funcPoint, std::vector<HierFuncDesignPoint> &subHierFuncPoints) {
  auto latency = funcPoint.latency;
  auto dspNum = funcPoint.dspNum;
  return HierFuncDesignPoint(latency, dspNum, funcPoint, subHierFuncPoints);
}

HierFuncDesignPoint createHierFuncDesignPoint(FuncDesignPoint funcPoint) {
  auto latency = funcPoint.latency;
  auto dspNum = funcPoint.dspNum;
  return HierFuncDesignPoint(latency, dspNum, funcPoint);
}

void HierFuncDesignSpace::combFuncDesignSpaces(ScaleHLSExplorer &explorer, bool directiveOnly, StringRef outputRootPath, StringRef csvRootPath, bool isTop) {
  LLVM_DEBUG(llvm::dbgs() << "\nCombine the function design spaces for function '"
                          << func.getName() << "'...\n";);

  // Base case: if there are no sub functions, just explore the loop design space of the current function.
  if (subHierFuncDesignSpaces.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No sub functions found in function '" << func.getName() << "', exploring the loop design space...\n";);
    //dumpFuncMLIR(func, "before_explore_loop_design_space", false);
    auto newFuncDesignSpace = explorer.exploreDesignSpace(func, directiveOnly, outputRootPath, csvRootPath, false); // if top function has no hierarchy, explore its space no matter what
    //dumpFuncMLIR(func, "after_explore_loop_design_space", false);
    setFuncDesignSpace(newFuncDesignSpace);
    for (auto &funcPoint : newFuncDesignSpace.paretoPoints) {
      auto newHierFuncPoint = createHierFuncDesignPoint(funcPoint);
      paretoPoints.push_back(newHierFuncPoint);
    }
    LLVM_DEBUG(llvm::dbgs() << "Loop design space of " << func.getName() << " is explored, the number of pareto points is " << paretoPoints.size() << ".\n";);
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Will traverse " << subHierFuncDesignSpaces.size() << " sub function design spaces in" << func.getName() << ".\n";);
  for (unsigned i = 0, e = subHierFuncDesignSpaces.size(); i < e; ++i) {
    auto &subHierFuncSpace = subHierFuncDesignSpaces[i];
    LLVM_DEBUG(llvm::dbgs() << "sub function " << i << " is " << subHierFuncSpace.func.getName() << ". There are " << subHierFuncSpace.paretoPoints.size() << " design points.\n";);
  }

  if (sampleSubFuncs) {
    unsigned iters = sampleIterNum;
    LLVM_DEBUG(llvm::dbgs() << "Sampling hierarchical function design points for function " << func.getName() << " with " << iters << " iterations.\n";);
    for (unsigned iter = 0; iter < iters; ++iter) {
      auto startTime = std::chrono::high_resolution_clock::now();
      std::vector<HierFuncDesignPoint> subHierFuncPoints;
      // Annotate latency and dsp to all other hierarchical function design spaces in the
      // hierarchical function point, they are static for all design points of the new function.
      for (unsigned ii = 0; ii < subHierFuncDesignSpaces.size(); ++ii) {
        auto randomIndex = rand() % subHierFuncDesignSpaces[ii].paretoPoints.size();
        auto &otherSubHierFuncSpace = subHierFuncDesignSpaces[ii];
        auto otherSubFuncPoint = otherSubHierFuncSpace.paretoPoints[randomIndex];
        auto otherSubFunc = otherSubHierFuncSpace.func;
        setTimingAndResourceSubFunc(func, otherSubFunc, otherSubFuncPoint.latency, otherSubFuncPoint.dspNum);
        subHierFuncPoints.push_back(otherSubFuncPoint);
      }

      // Explore the loop design space of the current function for the given configurations of sub functions
      auto newFuncDesignSpace = explorer.exploreDesignSpace(func, directiveOnly, outputRootPath, csvRootPath, isTop); // fully explore loops for the sample case
      setFuncDesignSpace(newFuncDesignSpace);
      for (auto &funcPoint : newFuncDesignSpace.paretoPoints) {
        auto newHierFuncPoint = createHierFuncDesignPoint(funcPoint, subHierFuncPoints);
        paretoPoints.push_back(newHierFuncPoint);
      }
      //dumpHierFuncDesignPoints(func.getName(), paretoPoints, *this);
      auto endTime = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
      LLVM_DEBUG(llvm::dbgs() << "iteration " << iter << " took " << duration.count() << " ms, for the function " << func.getName() << "\n";);
      
      cleanUpClonedModulesAndFunctions(newFuncDesignSpace);
    }
    updateParetoPoints(paretoPoints, maxDspNum);
    LLVM_DEBUG(llvm::dbgs() << "Done sampling hierarchical function design points for function " << func.getName() << ". There are now " << paretoPoints.size() << " pareto points in the current function design space.\n";);

  } else {

    // Initialize the hierarchical function design space with the first hierarchical function design space.
    LLVM_DEBUG(llvm::dbgs() << "Traversing all design points of the first sub function " << subHierFuncDesignSpaces[0].func.getName() << ". There are " << subHierFuncDesignSpaces[0].paretoPoints.size() << " design points.\n";);
    unsigned iter = 0;
    for (auto &subHierFuncPoint : subHierFuncDesignSpaces[0].paretoPoints) {
      auto startTime = std::chrono::high_resolution_clock::now();

      auto subFunc = getSubFunc(func, subHierFuncDesignSpaces[0].func.getName());
      setTimingAndResourceSubFunc(func, subFunc, subHierFuncPoint.latency, subHierFuncPoint.dspNum);
      
      std::vector<HierFuncDesignPoint> subHierFuncPoints;
      // form the sub hierarchical function design points. It includes the given point of the first function and the minimum resource point of the other functions.
      subHierFuncPoints.push_back(subHierFuncPoint);
      for (unsigned ii = 1; ii < subHierFuncDesignSpaces.size(); ++ii) {
        auto &otherSubHierFuncSpace = subHierFuncDesignSpaces[ii];
        auto otherSubFuncPoint = otherSubHierFuncSpace.paretoPoints[otherSubHierFuncSpace.paretoPoints.size() - 1];
        subHierFuncPoints.push_back(otherSubFuncPoint);
      }
      estimator.estimateFunc(func);
      auto curDspNum = getResource(func).getDsp();
      if (curDspNum > maxDspNum) {
        LLVM_DEBUG(llvm::dbgs() << "The current function " << func.getName() << " has " << curDspNum << " DSPs, which is greater than the maximum DSP number " << maxDspNum << ". Skipping this design point.\n";);
        iter--;
        continue;
      }

      // Explore the loop design space of the current function for the given configurations of sub functions
      auto newFuncDesignSpace = explorer.exploreDesignSpace(func, directiveOnly, outputRootPath, csvRootPath, isTop);
      setFuncDesignSpace(newFuncDesignSpace);
      for (auto &funcPoint : newFuncDesignSpace.paretoPoints) {
        auto newHierFuncPoint = createHierFuncDesignPoint(funcPoint, subHierFuncPoints);
        paretoPoints.push_back(newHierFuncPoint);
      }
      //dumpHierFuncDesignPoints(func.getName(), paretoPoints, *this);
      iter++;
      auto endTime = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
      LLVM_DEBUG(llvm::dbgs() << "iteration " << iter << " took " << duration.count() << " ms, for the first sub function " << subHierFuncDesignSpaces[0].func.getName() << "\n";);
      
      cleanUpClonedModulesAndFunctions(newFuncDesignSpace);
    }
    updateParetoPoints(paretoPoints, maxDspNum);
    LLVM_DEBUG(llvm::dbgs() << "Done traversing all design points of the first sub function " << subHierFuncDesignSpaces[0].func.getName() << ". There are now " << paretoPoints.size() << " pareto points in the current function design space.\n";);

    // Loop over the rest of the sub function design spaces.
    for (unsigned i = 1, e = subHierFuncDesignSpaces.size(); i < e; ++i) {
      unsigned iter = 0;
      std::vector<HierFuncDesignPoint> newParetoPoints;
      auto &subHierFuncSpace = subHierFuncDesignSpaces[i];

      // Traverse all hierarchical function design points.
      for (auto &hierFuncPoint : paretoPoints) {
        // Annotate latency and dsp to all other hierarchical function design spaces in the
        // hierarchical function point, they are static for all design points of the new function.
        for (unsigned ii = 0; ii < subHierFuncDesignSpaces.size(); ++ii) {
          if (ii != i) {
            auto &otherSubFuncPoint = hierFuncPoint.subHierFuncDesignPoints[ii];
            auto &otherSubHierFuncSpace = subHierFuncDesignSpaces[ii];
            auto otherSubFunc = otherSubHierFuncSpace.func;
            setTimingAndResourceSubFunc(func, otherSubFunc, otherSubFuncPoint.latency, otherSubFuncPoint.dspNum);
          }
        }

        // Traverse all design points of the next hierarchical function.
        //LLVM_DEBUG(llvm::dbgs() << "Traversing all design points of the next sub function " << subHierFuncSpace.func.getName() << ". There are " << subHierFuncSpace.paretoPoints.size() << " design points.\n";);
        for (auto &subHierFuncPoint : subHierFuncSpace.paretoPoints) {
          auto startTime = std::chrono::high_resolution_clock::now();
          // Annotate the next hierarchical function.
          auto subFunc = getSubFunc(func, subHierFuncSpace.func.getName());
          setTimingAndResourceSubFunc(func, subFunc, subHierFuncPoint.latency, subHierFuncPoint.dspNum);

          // Estimate the top-level function and generate a new hierarchical function design point.
          hierFuncPoint.subHierFuncDesignPoints[i] = subHierFuncPoint;
          // Explore the loop design space of the current function for the given configurations of sub functions
          auto newFuncDesignSpace = explorer.exploreDesignSpace(func, directiveOnly, outputRootPath, csvRootPath, isTop);
          setFuncDesignSpace(newFuncDesignSpace);
          for (auto &funcPoint : newFuncDesignSpace.paretoPoints) {
            auto newHierFuncPoint = createHierFuncDesignPoint(funcPoint, hierFuncPoint.subHierFuncDesignPoints);
            newParetoPoints.push_back(newHierFuncPoint);
          }
          //dumpHierFuncDesignPoints(func.getName(), newParetoPoints, *this);
          auto endTime = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
          LLVM_DEBUG(llvm::dbgs() << "iteration " << iter << " took " << duration.count() << " ms, for the sub function " << subHierFuncSpace.func.getName() << "\n";);
          iter++;
          cleanUpClonedModulesAndFunctions(newFuncDesignSpace);
        }
      }

      // Update pareto points after each combination, filtering by resources.
      updateParetoPoints(newParetoPoints, maxDspNum);
      paretoPoints = newParetoPoints;
      LLVM_DEBUG(llvm::dbgs() << "Done traversing all design points of the sub function " << subHierFuncSpace.func.getName() << ". The number of pareto points is " << paretoPoints.size() << ".\n";);
    }
    LLVM_DEBUG(llvm::dbgs() << "\n";);
  }
}

void HierFuncDesignSpace::dumpHierFuncDesignSpace(StringRef csvFilePath) {
  std::string errorMessage;
  auto csvFile = mlir::openOutputFile(csvFilePath, &errorMessage);
  if (!csvFile)
    return;
  auto &os = csvFile->os();

  // Print header row.

  // track current function design space
  auto &funcDesignSpaceRef = getFuncDesignSpace();
  LLVM_DEBUG(llvm::dbgs() << "The number of loop design spaces in the current function is " << funcDesignSpaceRef.loopDesignSpaces.size() << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "The number of sub hierarchical function design spaces is " << subHierFuncDesignSpaces.size() << "\n";);
  for (unsigned i = 0; i < funcDesignSpaceRef.loopDesignSpaces.size(); ++i) {
    auto &loopDesignSpace = funcDesignSpaceRef.loopDesignSpaces[i];
    LLVM_DEBUG(llvm::dbgs() << "The number of trip count list in the loop design space " << i << " is " << loopDesignSpace.tripCountList.size() << "\n";);
    for (unsigned j = 0; j < loopDesignSpace.tripCountList.size(); ++j)
      os << "tf" << 0 << "b" << i << "l" << j << ",";
    os << "tf" << 0 << "b" << i << "ii,";
  }
  os << "tf_cycle,tf_dsp,";
  LLVM_DEBUG(llvm::dbgs() << "The number of sub hierarchical function design spaces is " << subHierFuncDesignSpaces.size() << "\n";);
  for (unsigned h = 0; h < subHierFuncDesignSpaces.size(); ++h) {
    os << "sf" << h << "cycle" << ",";
    os << "sf" << h << "dsp" << ",";
  }
  os << "cycle,dsp,type\n";

  // Print pareto design points.

  // start with current function space (loops, not hierarchical)
  //LLVM_DEBUG(llvm::dbgs() << "The number of pareto points is " << paretoPoints.size() << "\n";);
  for (auto &hierFuncPoint : paretoPoints) {
    auto &funcPoint = hierFuncPoint.funcDesignPoint;
    for (unsigned i = 0, e = funcDesignSpaceRef.loopDesignSpaces.size(); i < e; ++i) {
      auto &loopPoint = funcPoint.loopDesignPoints[i];
      auto &loopSpace = funcDesignSpaceRef.loopDesignSpaces[i];
      //LLVM_DEBUG(llvm::dbgs() << "The size of the tile list in the loop design space " << i << " is " << loopSpace.getTileList(loopPoint.tileConfig).size() << "\n";);
      for (auto size : loopSpace.getTileList(loopPoint.tileConfig))
        os << size << ",";
      os << loopPoint.targetII << ",";
    }
    os << funcPoint.latency << "," << funcPoint.dspNum << ",";

    for (unsigned i = 0; i < subHierFuncDesignSpaces.size(); ++i) {
      auto &subHierFuncPoint = hierFuncPoint.subHierFuncDesignPoints[i];
      os << subHierFuncPoint.latency << "," << subHierFuncPoint.dspNum << ",";
    }
    os << hierFuncPoint.latency << "," << hierFuncPoint.dspNum << "," << "pareto\n";
  }

  csvFile->keep();
  LLVM_DEBUG(llvm::dbgs() << "Hierarchical function design space is dumped to file \""
                          << csvFilePath << "\".\n\n");
}

bool HierFuncDesignSpace::applyOptStrategyRecursive(func::FuncOp currentFunc, HierFuncDesignPoint hierFuncPoint, ModuleOp parentModule, unsigned sampleIndex) {
  //LLVM_DEBUG(llvm::dbgs() << "Apply optimization strategies to the current function '"
  //                        << currentFunc.getName() << "' for sample index " << sampleIndex << "...\n";);
  //LLVM_DEBUG(llvm::dbgs() << "The optimized latency of the current function is " << optimizedLatency << " and the optimized dsp num is " << optimizedDspNum << "\n";);
  //dumpFuncMLIR(currentFunc, "optimized_func", false);
  // STEP 2: Apply optimization strategies to the sub functions
  //LLVM_DEBUG(llvm::dbgs() << hierFuncPoint.subHierFuncDesignPoints.size() << " sub function design points to apply optimization strategies to\n";);
  //LLVM_DEBUG(llvm::dbgs() << "Number of design spaces in the sub hierarchical function design space is " << subHierFuncDesignSpaces.size() << "\n";);
  for (unsigned i = 0; i < hierFuncPoint.subHierFuncDesignPoints.size(); ++i) {
    auto &subHierFuncPoint = hierFuncPoint.subHierFuncDesignPoints[i];
    auto &subHierFuncSpace = subHierFuncDesignSpaces[i];
    auto subFunc = getSubFuncFromModule(parentModule, subHierFuncSpace.funcName);
    if (!subFunc) {
      llvm::errs() << "[DSE] ERROR: Cannot find sub function '" << subHierFuncSpace.funcName 
                   << "' in module\n";
      return false;
    }
    if (!subHierFuncSpace.applyOptStrategyRecursive(subFunc, subHierFuncPoint, parentModule, sampleIndex))
      return false;
  }

  // STEP 1: Apply optimization strategies to the current function
  //dumpFuncMLIR(currentFunc, "before_optimized_func", false);
  auto funcPoint = hierFuncPoint.funcDesignPoint;
  auto dspNum = hierFuncPoint.dspNum;
  auto latency = hierFuncPoint.latency;
  //LLVM_DEBUG(llvm::dbgs() << "The dsp num of the current function is " << dspNum << " and the latency is " << latency << "\n";);
  std::vector<FactorList> tileLists;
  std::vector<unsigned> targetIIs;
  auto &funcDesignSpaceRef = getFuncDesignSpace();
  for (unsigned i = 0, e = funcDesignSpaceRef.loopDesignSpaces.size(); i < e; ++i) {
    auto &loopPoint = funcPoint.loopDesignPoints[i];
    auto &loopSpace = funcDesignSpaceRef.loopDesignSpaces[i];
    auto tileList = loopSpace.getTileList(loopPoint.tileConfig);
    auto targetII = loopPoint.targetII;
    tileLists.push_back(tileList);
    targetIIs.push_back(targetII);
  }
  if (!applyOptStrategy(currentFunc, tileLists, targetIIs)){
    llvm::errs() << "[DSE] ERROR: Failed to apply optimization strategies to the current function '"
                 << currentFunc.getName() << "'\n";
    //dumpFuncMLIR(currentFunc, "failed_optimized_func", true);
    return false;
  }
  //LLVM_DEBUG(llvm::dbgs() << "Optimization strategies applied to the current function '"
  //                        << currentFunc.getName() << "' for sample index " << sampleIndex << ".\n";);
  estimator.estimateFunc(currentFunc);
  auto optimizedLatency = getTiming(currentFunc).getLatency();
  auto optimizedDspNum = getResource(currentFunc).getDsp();
  //LLVM_DEBUG(llvm::dbgs() << "Optimization strategies applied to the sub functions of '"
  //                        << currentFunc.getName() << "' for sample index " << sampleIndex << ".\n";);
  return true;
}

bool HierFuncDesignSpace::exportParetoDesigns(unsigned outputNum,
                                              StringRef outputRootPath, ModuleOp topModule) {
  unsigned paretoNum = paretoPoints.size();
  auto sampleStep = std::max(paretoNum / outputNum, (unsigned)1);

  // Traverse all detected pareto points.
  unsigned sampleIndex = 0;
  LLVM_DEBUG(llvm::dbgs() << "Will export " << outputNum << " pareto design points of function " << func.getName() << ".\n";);
  for (auto &hierFuncPoint : paretoPoints) {
    // Only export sampled points.
    if (sampleIndex % sampleStep == 0) {
      // Clone function with its module to preserve symbol table
      // Clone the module and cast to ModuleOp
      auto [tmpFunc, tmpModule] = cloneFunctionWithModule(func);

      //dumpFuncMLIR(tmpFunc, "before_optimized_func_hier", false);
      if (!applyOptStrategyRecursive(tmpFunc, hierFuncPoint, tmpModule, sampleIndex))
        return false;
      //dumpFuncMLIR(tmpFunc, "after_optimized_func_hier", false);
      estimator.estimateFunc(tmpFunc);

      // Parse a new output file.
      auto outputFilePath = outputRootPath.str() + "/function_hier_output/" + func.getName().str() +
                            "_pareto_" + std::to_string(sampleIndex) + ".mlir";

      std::string errorMessage;
      auto outputFile = mlir::openOutputFile(outputFilePath, &errorMessage);
      if (!outputFile)
        return false;

      auto &os = outputFile->os();
      os << tmpModule << "\n";
      outputFile->keep();

      tmpModule->destroy();
    }
    ++sampleIndex;
  }

  LLVM_DEBUG(
      llvm::dbgs() << "Sampled pareto points MLIR files are exported to path \""
                    << outputRootPath << "\".\n\n");
  return true;
}

//===----------------------------------------------------------------------===//
// Explorer Class Definition
//===----------------------------------------------------------------------===//

bool ScaleHLSExplorer::emitQoRDebugInfo(func::FuncOp func,
                                        std::string message) {
  estimator.estimateFunc(func);
  // auto latency = getTiming(func).getLatency();
  
  // Debug: Check what getResource returns
  auto resource = getResource(func);
  llvm::dbgs() << "[DEBUG emitQoRDebugInfo] Function: " << func.getName() << "\n";
  llvm::dbgs() << "[DEBUG emitQoRDebugInfo] Resource attribute: " << resource << "\n";
  if (!resource) {
    llvm::dbgs() << "[DEBUG emitQoRDebugInfo] ERROR: Resource attribute is null!\n";
    llvm::dbgs() << "[DEBUG emitQoRDebugInfo] Function attributes: ";
    func->dump();
    llvm::dbgs() << "\n";
    assert(false && "Resource attribute is null after estimation");
  }
  
  llvm::dbgs() << "[DEBUG emitQoRDebugInfo] About to call getDsp()...\n";
  auto dspNum = resource.getDsp();
  llvm::dbgs() << "[DEBUG emitQoRDebugInfo] getDsp() returned: " << dspNum << "\n";

  LLVM_DEBUG(llvm::dbgs() << message + "\n";
             //  llvm::dbgs() << "The clock cycle is " << Twine(latency)
             //               << ", DSP usage is " << Twine(dspNum) << ".\n\n";
  );

  return dspNum <= maxDspNum;
}

static int64_t getInnerParallelism(Block &block) {
  int64_t count = 0;
  for (auto loop : block.getOps<AffineForOp>()) {
    auto innerCount = getInnerParallelism(loop.getLoopBody().front());
    if (auto trip = getAverageTripCount(loop))
      count += trip.value() * innerCount;
    else
      count += innerCount;
  }

  // If the current loop is innermost loop, count should be one.
  return std::max(count, (int64_t)1);
}

bool ScaleHLSExplorer::evaluateFuncPipeline(func::FuncOp func) { return true; }

/// DSE Stage1: Simplify loop nests by unrolling. If we take the following loops
/// as example, where each nodes represents one sequential loop nests (LN). In
/// the simplification, we'll first try to pipeline LN1 and LN6. Suppose
/// unrolling LN6's region meets the resource constaints while pipelining LN1
/// not, we'll unroll LN6's region (fully unroll LN7 and LN8) and keep LN1
/// untouched. In the next step, we'll look into LN1 and check whether LN2's
/// region can be unrolled. Suppose unrolling LN2's region meets the resource
/// constraints, we'll unrolling LN2's region (fully unroll LN7 and LN8). Note
/// that in this simplification, all LNs that don't contain any LNs will not be
/// touched, such as LN5. Their optimization will be explored later. This
/// procedure will recursively applied to inner LNs until no eligible LN exists.
///
///     LN1      LN6
///      |        |
///     / \      / \
///   LN2 LN5  LN7 LN8
///    |
///   / \
/// LN3 LN4
///
/// After the simplification, the loop becomes the following one, where LN1 has
/// been proved untouchable as region loop unrolling is the primary optimization
/// that consumes the least extra resources. Formally, in the simplified
/// function, all non-leaf LNs is untouchable (LN1) and only leaf LNs can be
/// further optimized (LN2, LN5, and LN6).
///
///     LN1      LN6
///      |
///     / \
///   LN2 LN5
///
/// TODO: there is a large design space in this simplification.
bool ScaleHLSExplorer::simplifyLoopNests(func::FuncOp func) {
  LLVM_DEBUG(llvm::dbgs()
                 << "----------\nStage1: Simplify loop nests structure...\n";);

  auto funcForOps = func.getOps<AffineForOp>();
  std::vector<AffineForOp> targetLoops(funcForOps.begin(), funcForOps.end());

  while (!targetLoops.empty()) {
    std::vector<std::pair<int64_t, AffineForOp>> candidateLoops;

    // Collect all candidate loops. Here, only loops whose innermost loop has
    // more than one inner loops will be considered as a candidate.
    for (auto target : targetLoops) {
      AffineLoopBand loopBand;
      auto innermostLoop = getLoopBandFromOutermost(target, loopBand);

      // Calculate the overall introduced parallelism if the innermost loop of
      // the current loop band is fully unrolled.
      auto parallelism =
          getInnerParallelism(innermostLoop.getLoopBody().front());

      // Collect all candidate loops into an vector, we'll ignore too large
      // parallelism as unrolling them typically introduce very high cost.
      if (parallelism > 1 && parallelism <= 512)
        candidateLoops.push_back(
            std::pair<int64_t, AffineForOp>(parallelism, innermostLoop));
    }

    // Since all target loops have been handled, clear the targetLoops vector.
    targetLoops.clear();

    // Sort the candidate loops.
    llvm::sort(candidateLoops);

    // Traverse all candidates to check whether applying fully loop unrolling
    // has violation with the resource constraints. If so, add all inner loops
    // into targetLoops. Otherwise, fully unroll the candidate.
    for (auto pair : candidateLoops) {
      auto candidate = pair.second;

      // Create a temporary function (with its module to preserve symbol table).
      // Note: We need to set the flag on the original function first, then clone,
      // because the cloned function will be used for optimization.
      candidate->setAttr("opt_flag", BoolAttr::get(func.getContext(), true));
      auto [tmpFunc, tmpModule] = cloneFunctionWithModule(func);

      // Find the candidate loop in the temporary function and apply fully loop
      // unrolling to it.
      tmpFunc.walk([&](AffineForOp loop) {
        if (loop->getAttrOfType<BoolAttr>("opt_flag")) {
          applyFullyLoopUnrolling(*loop.getBody());
          applyMemoryOpts(tmpFunc);
          applyAutoArrayPartition(tmpFunc);
          return;
        }
      });

      // Estimate the temporary function.
      estimator.estimateFunc(tmpFunc);

      // Fully unroll the candidate loop or delve into child loops.
      if (getResource(tmpFunc).getDsp() <= maxDspNum) {
        applyFullyLoopUnrolling(*candidate.getBody());
        applyMemoryOpts(func);
        applyAutoArrayPartition(func);
      } else {
        auto childForOps = candidate.getOps<AffineForOp>();
        targetLoops.insert(targetLoops.end(), childForOps.begin(), childForOps.end());
      }

      candidate->removeAttr("opt_flag");
      tmpModule->destroy();
    }
  }

  return emitQoRDebugInfo(func, "\nFinish Stage1.");
}

/// DSE Stage2: Optimize leaf loop nests. Different optimization conbinations
/// will be applied to each leaf LNs, and the best one which meets the resource
/// constraints will be picked as the final solution.
/// TODO: better handle variable bound kernels.
bool ScaleHLSExplorer::optimizeLoopBands(func::FuncOp func,
                                         bool directiveOnly) {
  LLVM_DEBUG(llvm::dbgs() << "----------\nStage2: Apply loop perfection, loop "
                             "order opt, and remove variable loop bound...\n";);

  AffineLoopBands targetBands;
  getLoopBands(func.front(), targetBands);
  unsigned targetNum = targetBands.size();

  // Loop perfection, remove variable bound, and loop order optimization are
  // always applied for the convenience of polyhedral optimizations.
  for (unsigned i = 0; i < targetNum; ++i) {
    auto &band = targetBands[i];
    LLVM_DEBUG(llvm::dbgs() << "Loop band " << i << ": ";);
    applyAffineLoopPerfection(band);

    // If only explore directive optimizations, disable loop oreder opt.
    if (!directiveOnly)
      applyAffineLoopOrderOpt(band);

    applyRemoveVariableBound(band);
  }

  return emitQoRDebugInfo(func, "\nFinish Stage2.");
}

HierFuncDesignSpace ScaleHLSExplorer::exploreHierDesignSpace(func::FuncOp func, bool directiveOnly,
                                              StringRef outputRootPath,
                                              StringRef csvRootPath, bool isTop, bool sampleSubFuncs) {
  LLVM_DEBUG(llvm::dbgs() << "----------\nStage3: Conduct hierarchical function "
                             "design space exploration for function '"
                             << func.getName() << "'...\n";);

  // STEP 1: Recursively create hierarchical function design spaces for sub functions.
  // Find all call operations and resolve their callee functions.
  std::vector<HierFuncDesignSpace> subHierFuncDesignSpaces;
  // Track which functions we've already explored to avoid duplicates
  llvm::DenseSet<StringRef> processedCallees;
  
  func.walk([&](func::CallOp callOp) {
    // Resolve the callee function from the call operation
    auto callee = SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr());
    
    // If nearest symbol lookup failed, try module-level lookup as fallback.
    // This can happen during DSE when functions/loops are cloned and inserted
    // into contexts where the symbol table chain is broken.
    if (!callee) {
      auto calleeName = callOp.getCallee();
      if (auto module = callOp->getParentOfType<ModuleOp>()) {
        callee = module.lookupSymbol(calleeName);
      }
      if (!callee) {
        LLVM_DEBUG(llvm::dbgs() << "Warning: Cannot find callee for call op: " 
                                << callOp << " (skipping)\n";);
        return;
      }
    }
    
    auto subFunc = dyn_cast<func::FuncOp>(callee);
    if (!subFunc) {
      LLVM_DEBUG(llvm::dbgs() << "Warning: Callee is not a function operation for call: " 
                              << callOp << " (skipping)\n";);
      return;
    }
    
    // Skip if we've already processed this callee (same function called multiple times)
    StringRef calleeName = subFunc.getName();
    if (processedCallees.contains(calleeName)) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping duplicate sub-function: " << calleeName << "\n";);
      return;
    }
    processedCallees.insert(calleeName);
    
    LLVM_DEBUG(llvm::dbgs() << "Exploring hierarchical function design space for sub function '"
                            << calleeName << "'...\n";);
    auto subHierFuncSpace = exploreHierDesignSpace(subFunc, directiveOnly, outputRootPath, csvRootPath, false, false); // only sample sub functions on top level
    subHierFuncDesignSpaces.push_back(subHierFuncSpace);
  });
  
  LLVM_DEBUG(llvm::dbgs() << "Hierarchical function design spaces for sub functions created.\n";);
  LLVM_DEBUG(llvm::dbgs() << "Total " << subHierFuncDesignSpaces.size() << " sub functions explored for function '"
               << func.getName() << "'.\n";);
  // STEP : Combine function design spaces into current hierarchical function design space.
  //dumpFuncMLIR(func, "post_explore_design_space", false);
  HierFuncDesignSpace hierFuncSpace = HierFuncDesignSpace(func, func->getParentOfType<ModuleOp>(), subHierFuncDesignSpaces, estimator, 100000, sampleSubFuncs, sampleIterNum);
  hierFuncSpace.combFuncDesignSpaces(*this, directiveOnly, outputRootPath, csvRootPath, isTop);

  hierFuncSpace.dumpHierFuncDesignSpace(csvRootPath.str() + "function_hier_output/" + func.getName().str() + "_space.csv");

  return hierFuncSpace;
}

/// DSE Stage3: Explore the function design space through dynamic programming.
FuncDesignSpace ScaleHLSExplorer::exploreDesignSpace(func::FuncOp func, bool directiveOnly,
                                          StringRef outputRootPath,
                                          StringRef csvRootPath, bool isTop) {
  auto startExploreDesignSpaceTime = std::chrono::high_resolution_clock::now();
  //LLVM_DEBUG(llvm::dbgs() << "----------\nStage3: conduct single function design "
  //                           "space exploration...\n";);

  //dumpFuncMLIR(func, "pre_explore_design_space", false);

  // Clone the function by cloning its module first to preserve symbol table
  auto [tmpFunc, tmpModule] = cloneFunctionWithModule(func);
  //func::FuncOp tmpFunc = func;
  AffineLoopBands targetBands;
  getLoopBands(tmpFunc.front(), targetBands);
  
  // Try to perfect non-perfectly-nested loop bands before processing.
  // This attempts to sink operations between loops into the innermost loop.
  AffineLoopBands processedBands;
  for (auto &band : targetBands) {
    if (isPerfectlyNested(band)) {
      processedBands.push_back(band);
    } else {
      // Try to apply loop perfection to make it perfectly nested.
      LLVM_DEBUG(llvm::dbgs() << "Attempting to perfect non-perfectly-nested loop band in function '" 
                               << func.getName() << "'\n";);
      if (applyAffineLoopPerfection(band)) {
        // Check if it's now perfectly nested after perfection.
        processedBands.push_back(band);
        if (!isPerfectlyNested(band)) {
          LLVM_DEBUG(llvm::dbgs() << "Loop perfection applied but band is still not perfectly nested (likely due to scf.if or other unsupported operations)\n";); 
          assert(false && "Loop perfection applied but band is still not perfectly nested");
        }
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Failed to apply loop perfection\n";);
        //assert(false && "Failed to apply loop perfection");
      }
    }
  }
  
  unsigned targetNum = processedBands.size();

  //llvm::errs() << "[DSE] Found " << targetNum << " loop band(s) in function '" 
  //              << func.getName() << "'\n";

  // Search for the pareto frontiers of each target loop band.
  std::vector<LoopDesignSpace> loopSpaces;
  for (unsigned i = 0; i < targetNum; ++i) {
    auto space =
        LoopDesignSpace(tmpFunc, tmpModule, targetBands[i], estimator, maxDspNum,
                        maxExplParallel, maxLoopParallel, directiveOnly);

    unsigned initParallel = isTop ? 1 : maxInitParallel;
    //LLVM_DEBUG(llvm::dbgs() << "Loop band " << i << ": ";);
    space.initializeLoopDesignSpace(initParallel);

    //LLVM_DEBUG(llvm::dbgs() << "Loop band " << i << ": ";);
    space.exploreLoopDesignSpace(maxIterNum, maxDistance);
    loopSpaces.push_back(space);

    // Dump design points to csv file for each loop band.
    auto loopCsvFilePath = csvRootPath.str() + "loop_output/" + func.getName().str() + "_loop_" +
                           std::to_string(i) + "_space.csv";
    space.dumpLoopDesignSpace(loopCsvFilePath);
  }

  // Combine all loop design spaces into a function design space.
  // Clone the function again (with its module) for the function design space
  auto [tmpFunc2, tmpModule2] = cloneFunctionWithModule(func);
  auto funcSpace = FuncDesignSpace(tmpFunc2, tmpModule2, loopSpaces, estimator, maxDspNum);

  auto startCombineLoopDesignSpacesTime = std::chrono::high_resolution_clock::now();
  funcSpace.combLoopDesignSpaces();
  auto endCombineLoopDesignSpacesTime = std::chrono::high_resolution_clock::now();
  auto durationCombineLoopDesignSpaces = std::chrono::duration_cast<std::chrono::milliseconds>(endCombineLoopDesignSpacesTime - startCombineLoopDesignSpacesTime);
  LLVM_DEBUG(llvm::dbgs() << "Combine loop design spaces took " << durationCombineLoopDesignSpaces.count() << " ms\n";);

  // Dump design points to csv file for each function.
  auto funcCsvFilePath =
      csvRootPath.str() + "loop_output/" + func.getName().str() + "_space.csv";
  funcSpace.dumpFuncDesignSpace(funcCsvFilePath);

  auto endExploreDesignSpaceTime = std::chrono::high_resolution_clock::now();
  auto durationExploreDesignSpace = std::chrono::duration_cast<std::chrono::milliseconds>(endExploreDesignSpaceTime - startExploreDesignSpaceTime);
  LLVM_DEBUG(llvm::dbgs() << "Single function design space exploration took " << durationExploreDesignSpace.count() << " ms\n";);

  return funcSpace;
}

//===----------------------------------------------------------------------===//
// DesignSpaceExplore Entry
//===----------------------------------------------------------------------===//

/// This is a temporary approach that does not scale.
void ScaleHLSExplorer::applyDesignSpaceExplore(func::FuncOp func,
                                               bool directiveOnly,
                                               StringRef outputRootPath,
                                               StringRef csvRootPath) {
  emitQoRDebugInfo(func, "Start multiple level DSE.");

  // Create output directories if they don't exist
  std::string outputDir = outputRootPath.str() + "/function_output";
  std::string csvDir = csvRootPath.str() + "/loop_output";
  std::string hierOutputDir = outputRootPath.str() + "/function_hier_output";
  if (std::error_code EC = llvm::sys::fs::create_directories(outputDir)) {
    llvm::errs() << "Failed to create function output directory: " << outputDir << ": " << EC.message() << "\n";
    return;
  }
  if (std::error_code EC = llvm::sys::fs::create_directories(csvDir)) {
    llvm::errs() << "Failed to create loop CSV directory: " << csvDir << ": " << EC.message() << "\n";
  }
  if (std::error_code EC = llvm::sys::fs::create_directories(hierOutputDir)) {
    llvm::errs() << "Failed to create hierarchical function output directory: " << hierOutputDir << ": " << EC.message() << "\n";
  }
  // Simplify loop nests by unrolling.
  if (!simplifyLoopNests(func)) {
    assert(false && "Failed to simplify loop nests, you may need to increase the max DSPs");
    return;
  }

  // Optimize loop bands by loop perfection, loop order permutation, and loop
  // rectangularization.
  if (!optimizeLoopBands(func, directiveOnly)) {
    assert(false && "Failed to optimize loop bands");
    return;
  }

  // Explore the design space through a multiple level approach.
  //if (!exploreDesignSpace(func, directiveOnly, outputRootPath, csvRootPath))
  //  return;
  //dumpFuncMLIR(func, "before_explore_hier_design_space", false);
  auto hierFuncSpace = exploreHierDesignSpace(func, directiveOnly, outputRootPath, csvRootPath, true, sampleSubFuncs);
  //dumpFuncMLIR(func, "after_explore_hier_design_space", false);
  hierFuncSpace.exportParetoDesigns(outputNum, outputRootPath, topModule);
}

namespace {
struct DesignSpaceExplore : public DesignSpaceExploreBase<DesignSpaceExplore> {
  DesignSpaceExplore() = default;
  DesignSpaceExplore(std::string dseTargetSpec) { targetSpec = dseTargetSpec; }

  void runOnOperation() override {
    auto module = getOperation();

    // Read target specification JSON file.
    std::string errorMessage;
    auto configFile = mlir::openInputFile(targetSpec, &errorMessage);
    if (!configFile) {
      llvm::errs() << errorMessage << "\n";
      return signalPassFailure();
    }

    // Parse JSON file into memory.
    auto config = llvm::json::parse(configFile->getBuffer());
    if (!config) {
      llvm::errs() << "failed to parse the target spec json file\n";
      return signalPassFailure();
    }
    auto configObj = config.get().getAsObject();
    if (!configObj) {
      llvm::errs() << "support an object in the target spec json file, found "
                      "something else\n";
      return signalPassFailure();
    }

    // Collect DSE configurations.
    unsigned outputNum = configObj->getInteger("output_num").value_or(30);

    unsigned maxInitParallel =
        configObj->getInteger("max_init_parallel").value_or(32);
    unsigned maxExplParallel =
        configObj->getInteger("max_expl_parallel").value_or(1024);
    unsigned maxLoopParallel =
        configObj->getInteger("max_loop_parallel").value_or(128);

    assert(maxInitParallel <= maxExplParallel &&
           maxLoopParallel <= maxExplParallel &&
           "invalid configuration of DSE");

    unsigned maxIterNum = configObj->getInteger("max_iter_num").value_or(30);
    float maxDistance = configObj->getNumber("max_distance").value_or(3.0);

    bool directiveOnly =
        configObj->getBoolean("directive_only").value_or(false);
    bool resourceConstr =
        configObj->getBoolean("resource_constr").value_or(true);

    bool sampleSubFuncs = configObj->getBoolean("sample-sub-funcs").value_or(false);
    unsigned sampleIterNum = configObj->getInteger("sample-iter-num").value_or(100);

    LLVM_DEBUG(llvm::dbgs() << "Sample sub functions: " << sampleSubFuncs << "\n";);
    LLVM_DEBUG(llvm::dbgs() << "Sample iter num: " << sampleIterNum << "\n";);

    // Collect profiling latency and DSP usage data, where default values are
    // based on Xilinx PYNQ-Z1 board.
    llvm::StringMap<int64_t> latencyMap;
    getLatencyMap(configObj, latencyMap);
    llvm::StringMap<int64_t> dspUsageMap;
    getDspUsageMap(configObj, dspUsageMap);

    unsigned maxDspNum = ceil(configObj->getInteger("dsp").value_or(220) * 1.1);
    if (!resourceConstr)
      maxDspNum = UINT_MAX;

    // Initialize an performance and resource estimator.
    auto estimator = ScaleHLSEstimator(latencyMap, dspUsageMap, true);
    auto explorer = ScaleHLSExplorer(estimator, outputNum, maxDspNum,
                                     maxInitParallel, maxExplParallel,
                                     maxLoopParallel, maxIterNum, maxDistance, module, sampleSubFuncs, sampleIterNum);

    // Optimize the top function.
    // TODO: Support to contain sub-functions.
    for (auto func : module.getOps<func::FuncOp>()) {
      if (hasTopFuncAttr(func)) {
        auto startTime = std::chrono::high_resolution_clock::now();
        explorer.applyDesignSpaceExplore(func, directiveOnly, outputPath,
                                         csvPath);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        LLVM_DEBUG(llvm::dbgs() << "Design Space Exploration for function " << func.getName() << " took " << duration.count() << " ms\n";);
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass>
scalehls::createDesignSpaceExplorePass(std::string dseTargetSpec) {
  return std::make_unique<DesignSpaceExplore>(dseTargetSpec);
}
