//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef SCALEHLS_TRANSFORMS_EXPLORER_H
#define SCALEHLS_TRANSFORMS_EXPLORER_H

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "scalehls/Transforms/Estimator.h"
#include <vector>
#include <optional>
#include <cassert>

namespace mlir {
namespace scalehls {

using TileConfig = unsigned;

//===----------------------------------------------------------------------===//
// LoopDesignSpace Class Declaration
//===----------------------------------------------------------------------===//

struct LoopDesignPoint {
  explicit LoopDesignPoint(int64_t latency, int64_t dspNum,
                           TileConfig tileConfig, unsigned targetII)
      : latency(latency), dspNum(dspNum), tileConfig(tileConfig),
        targetII(targetII) {}

  int64_t latency;
  int64_t dspNum;

  TileConfig tileConfig;
  unsigned targetII;

  bool isActive = true;
};

class LoopDesignSpace {
public:
  explicit LoopDesignSpace(func::FuncOp func, ModuleOp parentModule, AffineLoopBand &band,
                           ScaleHLSEstimator &estimator, unsigned maxDspNum,
                           unsigned maxExplParallel, unsigned maxLoopParallel,
                           bool directiveOnly);

  /// Return the actual tile vector given a tile config.
  FactorList getTileList(TileConfig config);

  /// Return the corresponding tile config given a tile list.
  TileConfig getTileConfig(FactorList tileList);

  /// Calculate the Euclid distance of config a and config b.
  float getTileConfigDistance(TileConfig configA, TileConfig configB);

  /// Evaluate all design points under the given tile config.
  bool evaluateTileConfig(TileConfig config);

  /// Initialize the design space.
  void initializeLoopDesignSpace(unsigned maxInitParallel);

  /// Dump pareto and non-pareto points which have been evaluated in the design
  /// space to a csv output file.
  void dumpLoopDesignSpace(StringRef csvFilePath);

  /// Get a random tile config which is one of the closest neighbors of "point".
  Optional<TileConfig> getRandomClosestNeighbor(LoopDesignPoint point,
                                                float maxDistance);

  void exploreLoopDesignSpace(unsigned maxIterNum, float maxDistance);

  /// Stores current pareto frontiers and all evaluated design points. The
  /// "allPoints" is mainly used for design space dumping, which is actually not
  /// used in the DSE procedure.
  std::vector<LoopDesignPoint> paretoPoints;
  std::vector<LoopDesignPoint> allPoints;

  /// Associated function, loop band, and estimator.
  func::FuncOp func;
  ModuleOp parentModule;
  AffineLoopBand &band;
  ScaleHLSEstimator &estimator;
  unsigned maxDspNum;

  /// Records the trip count of each loop level.
  std::vector<unsigned> tripCountList;

  /// The dimension of this list is same to the number of loops in the loop
  /// band. The n-th element of this list stores all valid tile sizes of the
  /// n-th loop in the loop band.
  std::vector<std::vector<unsigned>> validTileSizesList;

  /// Holds the total number of valid tile size combinations.
  unsigned validTileConfigNum;

  /// Holds all tile configs that have not been estimated.
  llvm::SmallDenseSet<TileConfig, 32> unestimatedTileConfigs;

  // Whether to include loop transformation into the loop design space.
  bool directiveOnly;
};

//===----------------------------------------------------------------------===//
// FuncDesignSpace Class Declaration
//===----------------------------------------------------------------------===//

/// Each function design point contains multiple loop design point.
struct FuncDesignPoint {
  explicit FuncDesignPoint(int64_t latency, int64_t dspNum)
      : latency(latency), dspNum(dspNum) {}

  explicit FuncDesignPoint(int64_t latency, int64_t dspNum,
                           LoopDesignPoint point)
      : latency(latency), dspNum(dspNum) {
    loopDesignPoints.push_back(point);
  }

  explicit FuncDesignPoint(int64_t latency, int64_t dspNum,
                           std::vector<LoopDesignPoint> &points)
      : latency(latency), dspNum(dspNum) {
    loopDesignPoints = points;
  }

  int64_t latency;
  int64_t dspNum;

  std::vector<LoopDesignPoint> loopDesignPoints;
};

class FuncDesignSpace {
public:
  explicit FuncDesignSpace(func::FuncOp func, ModuleOp parentModule,
                           std::vector<LoopDesignSpace> loopDesignSpaces,
                           ScaleHLSEstimator &estimator, unsigned maxDspNum, bool sampleSubLoops, unsigned sampleIterNum)
      : func(func), parentModule(parentModule), loopDesignSpaces(loopDesignSpaces), estimator(estimator),
        maxDspNum(maxDspNum), sampleSubLoops(sampleSubLoops), sampleIterNum(sampleIterNum) {
    AffineLoopBands targetBands;
    getLoopBands(func.front(), targetBands);

    // Try to perfect non-perfectly-nested loop bands, matching what we do in exploreDesignSpace.
    for (auto &band : targetBands) {
      if (isPerfectlyNested(band)) {
        targetLoops.push_back(band.front());
        band.front()->setAttr("no_touch", BoolAttr::get(func.getContext(), true));
      } else {
        // Try to apply loop perfection to make it perfectly nested.
        if (applyAffineLoopPerfection(band) && isPerfectlyNested(band)) {
          targetLoops.push_back(band.front());
          band.front()->setAttr("no_touch", BoolAttr::get(func.getContext(), true));
        }
      }
    }
  }

  void combLoopDesignSpaces();

  void dumpFuncDesignSpace(StringRef csvFilePath);
  bool exportParetoDesigns(unsigned outputNum, StringRef outputRootPath);

  std::vector<FuncDesignPoint> paretoPoints;

  /// Associated function, loop design spaces, and estimator.
  func::FuncOp func;
  ModuleOp parentModule;
  std::vector<LoopDesignSpace> loopDesignSpaces;
  ScaleHLSEstimator &estimator;
  unsigned maxDspNum;
  bool sampleSubLoops;
  unsigned sampleIterNum;
  std::vector<AffineForOp> targetLoops;
};

struct HierFuncDesignPoint {
  explicit HierFuncDesignPoint(int64_t latency, int64_t dspNum,
                                FuncDesignPoint point)
      : latency(latency), dspNum(dspNum), funcDesignPoint(point) {}

  explicit HierFuncDesignPoint(int64_t latency, int64_t dspNum,
                                FuncDesignPoint point,
                                std::vector<HierFuncDesignPoint> &points)
      : latency(latency), dspNum(dspNum), funcDesignPoint(point) {
    subHierFuncDesignPoints = points;
  }

  int64_t latency;
  int64_t dspNum;

  FuncDesignPoint funcDesignPoint;

  std::vector<HierFuncDesignPoint> subHierFuncDesignPoints;
};

// Forward declaration to resolve circular dependency
class ScaleHLSExplorer;

class HierFuncDesignSpace {
public:
  // Constructor with funcDesignSpace
  explicit HierFuncDesignSpace(func::FuncOp func, ModuleOp parentModule,
                               FuncDesignSpace funcDesignSpace,
                               std::vector<HierFuncDesignSpace> subHierFuncDesignSpaces,
                               ScaleHLSEstimator &estimator, unsigned maxDspNum, bool sampleSubFuncs, unsigned sampleIterNum, bool noUnrollTopFunc)
      : func(func), parentModule(parentModule),funcDesignSpace(std::move(funcDesignSpace)), 
        subHierFuncDesignSpaces(std::move(subHierFuncDesignSpaces)), 
        estimator(estimator), maxDspNum(maxDspNum), funcName(func.getName().str()), sampleSubFuncs(sampleSubFuncs), sampleIterNum(sampleIterNum), noUnrollTopFunc(noUnrollTopFunc) {}

  // Constructor without funcDesignSpace - can be set later using setFuncDesignSpace()
  explicit HierFuncDesignSpace(func::FuncOp func, ModuleOp parentModule,
                               std::vector<HierFuncDesignSpace> subHierFuncDesignSpaces,
                               ScaleHLSEstimator &estimator, unsigned maxDspNum, bool sampleSubFuncs, unsigned sampleIterNum, bool noUnrollTopFunc)
      : func(func), parentModule(parentModule),
        subHierFuncDesignSpaces(std::move(subHierFuncDesignSpaces)), 
        estimator(estimator), maxDspNum(maxDspNum), funcName(func.getName().str()), sampleSubFuncs(sampleSubFuncs), sampleIterNum(sampleIterNum), noUnrollTopFunc(noUnrollTopFunc) {}

  // Setter to assign funcDesignSpace after construction
  void setFuncDesignSpace(FuncDesignSpace funcDesignSpace) {
    if (this->funcDesignSpace.has_value()) {
      this->funcDesignSpace.reset();
    }
    this->funcDesignSpace.emplace(std::move(funcDesignSpace));
  }

  // Getter for funcDesignSpace (returns reference, asserts it's set)
  FuncDesignSpace &getFuncDesignSpace() {
    assert(funcDesignSpace.has_value() && "funcDesignSpace must be set before use");
    return *funcDesignSpace;
  }

  const FuncDesignSpace &getFuncDesignSpace() const {
    assert(funcDesignSpace.has_value() && "funcDesignSpace must be set before use");
    return *funcDesignSpace;
  }

  // Check if funcDesignSpace is set
  bool hasFuncDesignSpace() const {
    return funcDesignSpace.has_value();
  }

  void combFuncDesignSpaces(ScaleHLSExplorer &explorer, bool directiveOnly, StringRef outputRootPath, StringRef csvRootPath, bool isTop);
  func::FuncOp getSubFunc(func::FuncOp func, StringRef subFuncName);
  func::FuncOp getSubFuncFromModule(ModuleOp module, StringRef subFuncName);
  bool applyOptStrategyRecursive(func::FuncOp func, HierFuncDesignPoint hierFuncPoint, ModuleOp parentModule, unsigned sampleIndex);

  void dumpHierFuncDesignSpace(StringRef csvFilePath);
  bool exportParetoDesigns(unsigned outputNum, StringRef outputRootPath, ModuleOp topModule);

  std::vector<HierFuncDesignPoint> paretoPoints;

  func::FuncOp func;
  ModuleOp parentModule;
  std::optional<FuncDesignSpace> funcDesignSpace;  // Optional to allow deferred initialization
  std::vector<HierFuncDesignSpace> subHierFuncDesignSpaces;  // Store by value to avoid dangling references
  ScaleHLSEstimator &estimator;
  unsigned maxDspNum;
  std::string funcName;
  bool sampleSubFuncs;
  unsigned sampleIterNum;
  bool noUnrollTopFunc;
  //SmallVector<AffineForOp, 4> targetLoops;
};


//===----------------------------------------------------------------------===//
// ScaleHLSExplorer Class Declaration
//===----------------------------------------------------------------------===//

class ScaleHLSExplorer {
public:
  explicit ScaleHLSExplorer(ScaleHLSEstimator &estimator, unsigned outputNum,
                            unsigned maxDspNum, unsigned maxInitParallel,
                            unsigned maxExplParallel, unsigned maxLoopParallel,
                            unsigned maxIterNum, float maxDistance, ModuleOp module, bool sampleSubFuncs, unsigned sampleIterNum, bool noUnrollTopFunc)
      : estimator(estimator), outputNum(outputNum), maxDspNum(maxDspNum),
        maxInitParallel(maxInitParallel), maxExplParallel(maxExplParallel),
        maxLoopParallel(maxLoopParallel), maxIterNum(maxIterNum),
        maxDistance(maxDistance), topModule(cast<ModuleOp>(module->clone())), sampleSubFuncs(sampleSubFuncs), sampleIterNum(sampleIterNum), noUnrollTopFunc(noUnrollTopFunc) {}

  bool emitQoRDebugInfo(func::FuncOp func, std::string message);

  bool evaluateFuncPipeline(func::FuncOp func);
  bool simplifyLoopNests(func::FuncOp func);
  bool optimizeLoopBands(func::FuncOp func, bool directiveOnly);
  FuncDesignSpace exploreDesignSpace(func::FuncOp func, bool directiveOnly,
                          StringRef outputRootPath, StringRef csvRootPath, bool isTop);

  HierFuncDesignSpace exploreHierDesignSpace(func::FuncOp func, bool directiveOnly,
                              StringRef outputRootPath, StringRef csvRootPath, bool isTop, bool sampleSubFuncs);

  void applyDesignSpaceExplore(func::FuncOp func, bool directiveOnly,
                               StringRef outputRootPath, StringRef csvRootPath);

  ScaleHLSEstimator &estimator;

  // The number of pareto designs that will be generated.
  unsigned outputNum;

  unsigned maxDspNum;

  // The maximum parallelism of the initiation and exploration of phase of DSE.
  unsigned maxInitParallel;
  unsigned maxExplParallel;

  // The maximum parallelism of each loop.
  unsigned maxLoopParallel;

  // The maximum iteration number of DSE.
  unsigned maxIterNum;

  // The maximum distance in the neighbor search of DSE.
  float maxDistance;

  ModuleOp topModule;
  bool sampleSubFuncs;
  unsigned sampleIterNum;
  bool noUnrollTopFunc;
};

} // namespace scalehls
} // namespace mlir

#endif // SCALEHLS_TRANSFORMS_EXPLORER_H
