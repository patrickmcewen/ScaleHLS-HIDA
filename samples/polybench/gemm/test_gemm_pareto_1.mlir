func.func @test_gemm(%arg0: f32, %arg1: f32, %arg2: memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>, %arg3: memref<2x2xf32>, %arg4: memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>) attributes {llvm.linkage = #llvm.linkage<external>, resource = #hls.res<lut = 0, dsp = 4, bram = 0>, timing = #hls.time<0 -> 26, latency = 26, interval = 26>, top_func} {
  affine.for %arg5 = 0 to 2 {
    affine.for %arg6 = 0 to 2 {
      %0 = affine.load %arg2[%arg6, 0] {partition_indices = [0, 0], timing = #hls.time<4 -> 6, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
      %1 = arith.mulf %0, %arg1 {timing = #hls.time<6 -> 10, latency = 4, interval = 1>} : f32
      %2 = hls.affine.select affine_set<(d0) : (d0 == 0)>(%arg5) %1, %0 {timing = #hls.time<10 -> 10, latency = 0, interval = 0>} : f32
      %3 = affine.load %arg3[%arg6, %arg5] {partition_indices = [0, 0], timing = #hls.time<0 -> 2, latency = 2, interval = 1>} : memref<2x2xf32>
      %4 = arith.mulf %arg0, %3 {timing = #hls.time<2 -> 6, latency = 4, interval = 1>} : f32
      %5 = affine.load %arg4[%arg5, 0] {partition_indices = [0, 0], timing = #hls.time<4 -> 6, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
      %6 = arith.mulf %4, %5 {timing = #hls.time<6 -> 10, latency = 4, interval = 1>} : f32
      %7 = arith.addf %2, %6 {timing = #hls.time<10 -> 15, latency = 5, interval = 1>} : f32
      affine.store %7, %arg2[%arg6, 0] {partition_indices = [0, 0], timing = #hls.time<15 -> 16, latency = 1, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
      %8 = affine.load %arg2[%arg6, 1] {partition_indices = [0, 1], timing = #hls.time<4 -> 6, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
      %9 = arith.mulf %8, %arg1 {timing = #hls.time<6 -> 10, latency = 4, interval = 1>} : f32
      %10 = hls.affine.select affine_set<(d0) : (d0 == 0)>(%arg5) %9, %8 {timing = #hls.time<10 -> 10, latency = 0, interval = 0>} : f32
      %11 = affine.load %arg4[%arg5, 1] {partition_indices = [0, 1], timing = #hls.time<4 -> 6, latency = 2, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
      %12 = arith.mulf %4, %11 {timing = #hls.time<6 -> 10, latency = 4, interval = 1>} : f32
      %13 = arith.addf %10, %12 {timing = #hls.time<10 -> 15, latency = 5, interval = 1>} : f32
      affine.store %13, %arg2[%arg6, 1] {partition_indices = [0, 1], timing = #hls.time<15 -> 16, latency = 1, interval = 1>} : memref<2x2xf32, #hls.partition<[none, cyclic], [1, 2]>, #hls.mem<lutram_2p>>
    } {loop_directive = #hls.loop<pipeline = true, target_ii = 2, dataflow = false, flatten = false>, loop_info = #hls.info<flatten_trip_count = 2, iter_latency = 16, min_ii = 2>, parallel, timing = #hls.time<0 -> 20, latency = 20, interval = 20>}
  } {loop_directive = #hls.loop<pipeline = false, target_ii = 1, dataflow = false, flatten = true>, loop_info = #hls.info<flatten_trip_count = 4, iter_latency = 16, min_ii = 2>, timing = #hls.time<0 -> 24, latency = 24, interval = 24>}
  return {timing = #hls.time<24 -> 24, latency = 0, interval = 0>}
}
