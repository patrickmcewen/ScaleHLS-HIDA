
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

using namespace std;

/// This is top function.
/// Latency=23, interval=23
/// DSP=7, BRAM=0
void test_gemm(
  float v0,
  float v1,
  float v2[2][2],
  float v3[2][2],
  float v4[2][2]
) {	// L3, [0,23)
  #pragma HLS interface s_axilite port=return bundle=ctrl
  #pragma HLS interface s_axilite port=v0 bundle=ctrl
  #pragma HLS interface s_axilite port=v1 bundle=ctrl
  #pragma HLS interface bram port=v2
  #pragma HLS array_partition variable=v2 cyclic factor=2 dim=2

  #pragma HLS interface bram port=v3
  #pragma HLS interface bram port=v4
  #pragma HLS array_partition variable=v4 cyclic factor=2 dim=2

  for (int v5 = 0; v5 < 2; v5 += 1) {	// L4, [0,21), iterCycle=16, II=1
    for (int v6 = 0; v6 < 2; v6 += 1) {	// L5, [0,19), iterCycle=16, II=1
      #pragma HLS pipeline II=1
      float v7 = v2[v6][0];	// L6, [4,6)
      float v8 = v7 * v1;	// L7, [6,10)
      float v9 = (v5 == 0) ? v8 : v7;	// L8, [10,10)
      float v10 = v3[v6][v5];	// L9, [0,2)
      float v11 = v0 * v10;	// L10, [2,6)
      float v12 = v4[v5][0];	// L11, [4,6)
      float v13 = v11 * v12;	// L12, [6,10)
      float v14 = v9 + v13;	// L13, [10,15)
      v2[v6][0] = v14;	// L14, [15,16)
      float v15 = v2[v6][1];	// L15, [4,6)
      float v16 = v15 * v1;	// L16, [6,10)
      float v17 = (v5 == 0) ? v16 : v15;	// L17, [10,10)
      float v18 = v4[v5][1];	// L18, [4,6)
      float v19 = v11 * v18;	// L19, [6,10)
      float v20 = v17 + v19;	// L20, [10,15)
      v2[v6][1] = v20;	// L21, [15,16)
    }
  }
}

