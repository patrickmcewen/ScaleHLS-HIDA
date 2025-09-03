func.func @test_gemm(%arg0: f32, %arg1: f32, %arg2: memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>, %arg3: memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>, %arg4: memref<2x2xf32, #hls.partition<[cyclic, cyclic], [2, 2]>, #hls.mem<lutram_2p>>) attributes {llvm.linkage = #llvm.linkage<external>, resource = #hls.res<lut = 0, dsp = 2, bram = 0>, timing = #hls.time<0 -> 40, latency = 40, interval = 40>, top_func} {
  affine.for %arg5 = 0 to 2 {
    %0 = affine.load %arg2[%arg5, 0] {partition_indices = [0, 0], timing = #hls.time<4 -> 6, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
    %1 = arith.mulf %0, %arg1 {timing = #hls.time<6 -> 10, latency = 4, interval = 1>} : f32
    %2 = affine.load %arg3[%arg5, 0] {partition_indices = [0, 0], timing = #hls.time<0 -> 2, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
    %3 = arith.mulf %arg0, %2 {timing = #hls.time<2 -> 6, latency = 4, interval = 1>} : f32
    %4 = affine.load %arg4[0, 0] {partition_indices = [0, 0], timing = #hls.time<4 -> 6, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[cyclic, cyclic], [2, 2]>, #hls.mem<lutram_2p>>
    %5 = arith.mulf %3, %4 {timing = #hls.time<6 -> 10, latency = 4, interval = 1>} : f32
    %6 = arith.addf %1, %5 {timing = #hls.time<10 -> 15, latency = 5, interval = 1>} : f32
    %7 = affine.load %arg2[%arg5, 1] {partition_indices = [0, 1], timing = #hls.time<4 -> 6, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
    %8 = arith.mulf %7, %arg1 {timing = #hls.time<6 -> 10, latency = 4, interval = 1>} : f32
    %9 = affine.load %arg4[0, 1] {partition_indices = [0, 1], timing = #hls.time<4 -> 6, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[cyclic, cyclic], [2, 2]>, #hls.mem<lutram_2p>>
    %10 = arith.mulf %3, %9 {timing = #hls.time<6 -> 10, latency = 4, interval = 1>} : f32
    %11 = arith.addf %8, %10 {timing = #hls.time<10 -> 15, latency = 5, interval = 1>} : f32
    %12 = affine.load %arg3[%arg5, 1] {partition_indices = [0, 1], timing = #hls.time<5 -> 7, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
    %13 = arith.mulf %arg0, %12 {timing = #hls.time<7 -> 11, latency = 4, interval = 1>} : f32
    %14 = affine.load %arg4[1, 0] {partition_indices = [1, 0], timing = #hls.time<9 -> 11, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[cyclic, cyclic], [2, 2]>, #hls.mem<lutram_2p>>
    %15 = arith.mulf %13, %14 {timing = #hls.time<11 -> 15, latency = 4, interval = 1>} : f32
    %16 = arith.addf %6, %15 {timing = #hls.time<15 -> 20, latency = 5, interval = 1>} : f32
    affine.store %16, %arg2[%arg5, 0] {partition_indices = [0, 0], timing = #hls.time<20 -> 21, latency = 1, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
    %17 = affine.load %arg4[1, 1] {partition_indices = [1, 1], timing = #hls.time<9 -> 11, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[cyclic, cyclic], [2, 2]>, #hls.mem<lutram_2p>>
    %18 = arith.mulf %13, %17 {timing = #hls.time<11 -> 15, latency = 4, interval = 1>} : f32
    %19 = arith.addf %11, %18 {timing = #hls.time<15 -> 20, latency = 5, interval = 1>} : f32
    affine.store %19, %arg2[%arg5, 1] {partition_indices = [0, 1], timing = #hls.time<20 -> 21, latency = 1, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
  } {loop_directive = #hls.loop<pipeline = true, target_ii = 15, dataflow = false, flatten = false>, loop_info = #hls.info<flatten_trip_count = 2, iter_latency = 21, min_ii = 15>, parallel, timing = #hls.time<0 -> 38, latency = 38, interval = 38>}
  return {timing = #hls.time<38 -> 38, latency = 0, interval = 0>}
}
