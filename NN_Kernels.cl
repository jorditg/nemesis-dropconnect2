#define TILEX 4

#define TILEX_SHIFT 2
#define TILEY 4
#define TILEY_SHIFT 2

#define NULL 0

#define SIGMOID 1
#define TANH 2
#define RELU 3
// Max-norm limit value for ReLU Activation Function
// Typical values between 3-4. Disable MAX_NORM fixing it to high value
//#define RELU_MAX_NORM	((float4) (3.0f))	

#define ACTIVATION_FUNCTION RELU

__constant float4 zeros = (float4) (0.0f);
__constant float4 ones = (float4) (1.0f);
__constant float4 epsilon = (float4) (1E-30);

// sequence used for getting the memory positions of the float4xfloat4 blocks of data without indexing
__constant int4 normal_seq = (int4) (0, 1, 2, 3);

/*
 *  Returns the index of the element located in (row, col) in a 
 *  row-major memory ordered matrix 
 */
int4 get_index(int offset, int r, int c, int nr_c, int4 r_sequence)
{
    const int off = offset + c;
    int4 result = (int4) (r);
    result += r_sequence;
    result *= nr_c;
    result += off;
    
    return result;
}

float4 activation_function(float4 x)
{
#if ACTIVATION_FUNCTION == SIGMOID
    return ones / ( ones + exp( -x ) );
#elif ACTIVATION_FUNCTION == TANH
    const float4 eplus = exp(x);
    const float4 eminus = exp(-x);
    return (eplus - eminus) / (eplus + eminus);
#elif ACTIVATION_FUNCTION == RELU
    #ifdef RELU_MAX_NORM
    x = (x > RELU_MAX_NORM)?RELU_MAX_NORM:x;
    #endif
    return (x > zeros)?x:zeros;
#endif	
}


float4 activation_function_derivative(float4 val)
{
#if ACTIVATION_FUNCTION == SIGMOID
    return val*(ones - val);
#elif ACTIVATION_FUNCTION == TANH
    return ones - val*val;
#elif ACTIVATION_FUNCTION == RELU
    #ifdef RELU_MAX_NORM
    return (val > zeros && val < RELU_MAX_NORM)?ones:zeros;
    #else
    return (val > zeros)?ones:zeros;
    #endif
#endif	
}

float4 cross_entropy(float4 t, float4 y)
{
  // cross-entropy "traditional"
  //return ( t * log(y + epsilon) + (ones - t) * log (ones - y + epsilon) );
  
  // log-likelihood "cross entropy"
  return  t * log(y + epsilon);
}

float4 masked_value(float4 value, uchar byte_mask, int nibble)
{
  /*
    byte_mask saves the mask of 8 elements (one per bit).
    As the elements are accesed of blocksof 4 elements (float4)
    we haveto choose between the two nibbles (high 4bitand low 4bit)
    Attention!!: The high nibble contains the mask of the pair index
    element (that is to say, the one with the lower index).
    This has to be taken into account in the tests because the mask
    order every 2 elements has to be reversed.
   */
  const int shift = (nibble << 2);  // nibble = 1 --> shift = 4, nibble = 0 --> shift = 0
  const uchar nibble_mask = (byte_mask >> shift);
  const uchar4 decompressed = (uchar4) (
					((nibble_mask >> 3) & 1), 
					((nibble_mask >> 2) & 1), 
					((nibble_mask >> 1) & 1), 
					(nibble_mask & 1)
				       );
  const float4 ret = value * convert_float4_rtz(decompressed);
  return ret;
}

/* Matrix A is cached into local memory block */
/* Required global threads = (colsC / 4, rowsC / 4) 
 * Required sizes: rowsC, colsC, rowsA, colsA, rowsB, colsB
 * multiples of 8.
 */
__kernel void matrixMultiplicationSigmoidKernelLocal
                             (__global float4 *matrixA,
                              __global float4 *matrixB,
                              __global float4 *matrixC,
                              __global float4 *bias,
                              const int colsAfloat4,
                              const int offsetA,
                              const int offsetB,
                              const int offsetC,
                              const int offsetBias,
                              __local float4 *blockA,
                              const int calcSigmoid,
                              const int averageResultBeforeSigmoid,
                              const int AInColMajorOrder,
                              const int BInColMajorOrder,
                              const int sumToMatrixC,
                              const float multPrevVal,
                              const float multSum,
                              __global uchar *maskA,
                              __global uchar *maskB,
                              __global uchar *maskBias)
{
    const int gid0 = get_global_id(0);
    const int gid1 = get_global_id(1);
    const int lid0 = get_local_id(0);
    const int lid1 = get_local_id(1);
    const int lsz0 = get_local_size(0);
    const int lsz1 = get_local_size(1);
    const int gsz0 = get_global_size(0);
    const int gsz1 = get_global_size(1);
    
    
    int4 blockPos = get_index(0, (lid1 << TILEY_SHIFT), lid0, lsz0, normal_seq);

    /* Position of thread will be according to the number of values it writes i.e TILE size */
    
    const int col_C = gid0;
    const int row_C = gid1;
    const int nr_cols_C = gsz0; 
    const int4 globalPos = get_index(offsetC, (row_C << TILEY_SHIFT), col_C, nr_cols_C, normal_seq);

    /* Each thread writes 4 float4s */
    float4 sum0 = (float4)(0);
    float4 sum1 = (float4)(0);
    float4 sum2 = (float4)(0);
    float4 sum3 = (float4)(0);

    int temp = colsAfloat4;
    
    /* This loop runs for number of blocks of A in horizontal direction */
    for(int i = 0; i < (temp / lsz0); i++)
    {
        /* Calculate global ids of threads from the particular block to load from matrix A depending on i */
        //int globalPosA = offsetA + i * get_local_size(0) + get_local_id(0) + (get_global_id(1) << TILEY_SHIFT) * temp;

        const int col_A = i * lsz0 + lid0;
        const int row_A = gid1;
        const int nr_rows_A = gsz1;
        const int nr_cols_A = temp; 
        
        if(!AInColMajorOrder) {
            const int4 globalPosA = get_index(offsetA, (row_A << TILEY_SHIFT), col_A, nr_cols_A, normal_seq);	
	    uchar4 mask = (uchar4) (0xFF);
	    const int4 nibble = globalPosA & 1;	    
	    if(maskA) {
              const int4 maskIdx = (globalPosA >> 1)*(col_C << TILEY_SHIFT);
	      mask = (uchar4) (maskA[maskIdx.x], maskA[maskIdx.y], maskA[maskIdx.z], maskA[maskIdx.w]);
	    } 
            /* Load values in blockA from matrixA */    
            blockA[blockPos.x] = masked_value(matrixA[globalPosA.x], mask.x, nibble.x);
            blockA[blockPos.y] = masked_value(matrixA[globalPosA.y], mask.y, nibble.y);
            blockA[blockPos.z] = masked_value(matrixA[globalPosA.z], mask.z, nibble.z);
            blockA[blockPos.w] = masked_value(matrixA[globalPosA.w], mask.w, nibble.w);
        } else {
            // If A is in column major order not only the index calculation is different but the float4xfloat4 block
            // of data has to be transposed
            const int4 globalPosA = get_index(offsetA, (col_A << TILEY_SHIFT), row_A, nr_rows_A, normal_seq);
	    uchar4 mask = (uchar4) (0xFF);
	    const int4 nibble = globalPosA & 1;	    
	    if(maskA) {
              const int4 maskIdx = (globalPosA >> 1)*(col_C << TILEY_SHIFT);
	      mask = (uchar4) (maskA[maskIdx.x], maskA[maskIdx.y], maskA[maskIdx.z], maskA[maskIdx.w]);
	    }
            // first of all we load the block to private memory
            float4 v1 = masked_value(matrixA[globalPosA.x], mask.x, nibble.x);
            float4 v2 = masked_value(matrixA[globalPosA.y], mask.y, nibble.y);
            float4 v3 = masked_value(matrixA[globalPosA.z], mask.z, nibble.z);
            float4 v4 = masked_value(matrixA[globalPosA.w], mask.w, nibble.w);

            // now we transpose it and assign it to the block of memory
            blockA[blockPos.x] = (float4) (v1.x, v2.x, v3.x, v4.x);
            blockA[blockPos.y] = (float4) (v1.y, v2.y, v3.y, v4.y);
            blockA[blockPos.z] = (float4) (v1.z, v2.z, v3.z, v4.z);
            blockA[blockPos.w] = (float4) (v1.w, v2.w, v3.w, v4.w);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Calculate global ids of threads from the particular block to load from matrix B depending on i */
        const int col_B = gid0;
        const int row_B = i * lsz0;
        const int nr_rows_B = temp;
        const int nr_cols_B = gsz0; 

        /* This loop runs for number of threads in horizontal direction in the block of A */
        for(int j = 0; j < lsz0 * 4; j=j+4)
        {
            /* Load 4 float4s from blockA : access patters = strided from local memory */
            float4 tempA0 = blockA[(j >> 2) + lid1 * TILEY * lsz0];
            float4 tempA1 = blockA[(j >> 2) + (lid1 * TILEY + 1) * lsz0];
            float4 tempA2 = blockA[(j >> 2) + (lid1 * TILEY + 2) * lsz0];
            float4 tempA3 = blockA[(j >> 2) + (lid1 * TILEY + 3) * lsz0];

            /* Load corresponding values from matrixB, access pattern = linear from global memory */
            float4 tempB0;
            float4 tempB1;
            float4 tempB2;
            float4 tempB3;

            if(!BInColMajorOrder) {
              const int4 globalPosB = get_index(offsetB, (row_B << TILEY_SHIFT) + j, col_B, nr_cols_B, normal_seq);
  	      uchar4 mask = (uchar4) (0xFF);
   	      const int4 nibble = globalPosB & 1;
	      if(maskB) {
                const int4 maskIdx = (globalPosB >> 1)*(row_C << TILEY_SHIFT);
	        mask = (uchar4) (maskB[maskIdx.x], maskB[maskIdx.y], maskB[maskIdx.z], maskB[maskIdx.w]);
	      }
              tempB0 = masked_value(matrixB[globalPosB.x], mask.x, nibble.x);
              tempB1 = masked_value(matrixB[globalPosB.y], mask.y, nibble.y);
              tempB2 = masked_value(matrixB[globalPosB.z], mask.z, nibble.z);
              tempB3 = masked_value(matrixB[globalPosB.w], mask.w, nibble.w);
            } else {
              const int4 globalPosB = get_index(offsetB, (col_B << TILEY_SHIFT), row_B + (j >> 2), nr_rows_B, normal_seq);
	      uchar4 mask = (uchar4) (0xFF);
   	      const int4 nibble = globalPosB & 1;
	      if(maskB) {
                const int4 maskIdx = (globalPosB >> 1)*(row_C << TILEY_SHIFT);
	        mask = (uchar4) (maskB[maskIdx.x], maskB[maskIdx.y], maskB[maskIdx.z], maskB[maskIdx.w]);
	      }

              // load block in private memory
              float4 v1 = masked_value(matrixB[globalPosB.x], mask.x, nibble.x);
              float4 v2 = masked_value(matrixB[globalPosB.y], mask.y, nibble.y);
              float4 v3 = masked_value(matrixB[globalPosB.z], mask.z, nibble.z);
              float4 v4 = masked_value(matrixB[globalPosB.w], mask.w, nibble.w);

              // now we transpose it
              tempB0 = (float4) (v1.x, v2.x, v3.x, v4.x);
              tempB1 = (float4) (v1.y, v2.y, v3.y, v4.y);
              tempB2 = (float4) (v1.z, v2.z, v3.z, v4.z);
              tempB3 = (float4) (v1.w, v2.w, v3.w, v4.w);

            }
            sum0.x += tempA0.x * tempB0.x + tempA0.y * tempB1.x + tempA0.z * tempB2.x + tempA0.w * tempB3.x;
            sum0.y += tempA0.x * tempB0.y + tempA0.y * tempB1.y + tempA0.z * tempB2.y + tempA0.w * tempB3.y;
            sum0.z += tempA0.x * tempB0.z + tempA0.y * tempB1.z + tempA0.z * tempB2.z + tempA0.w * tempB3.z;
            sum0.w += tempA0.x * tempB0.w + tempA0.y * tempB1.w + tempA0.z * tempB2.w + tempA0.w * tempB3.w;

            sum1.x += tempA1.x * tempB0.x + tempA1.y * tempB1.x + tempA1.z * tempB2.x + tempA1.w * tempB3.x;
            sum1.y += tempA1.x * tempB0.y + tempA1.y * tempB1.y + tempA1.z * tempB2.y + tempA1.w * tempB3.y;
            sum1.z += tempA1.x * tempB0.z + tempA1.y * tempB1.z + tempA1.z * tempB2.z + tempA1.w * tempB3.z;
            sum1.w += tempA1.x * tempB0.w + tempA1.y * tempB1.w + tempA1.z * tempB2.w + tempA1.w * tempB3.w;

            sum2.x += tempA2.x * tempB0.x + tempA2.y * tempB1.x + tempA2.z * tempB2.x + tempA2.w * tempB3.x;
            sum2.y += tempA2.x * tempB0.y + tempA2.y * tempB1.y + tempA2.z * tempB2.y + tempA2.w * tempB3.y;
            sum2.z += tempA2.x * tempB0.z + tempA2.y * tempB1.z + tempA2.z * tempB2.z + tempA2.w * tempB3.z;
            sum2.w += tempA2.x * tempB0.w + tempA2.y * tempB1.w + tempA2.z * tempB2.w + tempA2.w * tempB3.w;

            sum3.x += tempA3.x * tempB0.x + tempA3.y * tempB1.x + tempA3.z * tempB2.x + tempA3.w * tempB3.x;
            sum3.y += tempA3.x * tempB0.y + tempA3.y * tempB1.y + tempA3.z * tempB2.y + tempA3.w * tempB3.y;
            sum3.z += tempA3.x * tempB0.z + tempA3.y * tempB1.z + tempA3.z * tempB2.z + tempA3.w * tempB3.z;
            sum3.w += tempA3.x * tempB0.w + tempA3.y * tempB1.w + tempA3.z * tempB2.w + tempA3.w * tempB3.w;

        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
	// If bias not NULL
    if(bias != NULL) 
	{
        const int idx = offsetBias + gid0;
		const int nibble = idx & 1;	// The nibble is the same for all biases because the number of neurons
									// is multiple of 8
        float4 bias_val = bias[idx];
        const uchar4 mask = (uchar4) (0xFF);
        if (maskBias)
        {
            const int row_C_tiley = (row_C << TILEY_SHIFT);
            const int idxdiv2 = (idx >> 1);
            const int4 maskIdx = (
                                  idxdiv2*(row_C_tiley + 0),  // bias only used in feed-forward
                                  idxdiv2*(row_C_tiley + 1),
                                  idxdiv2*(row_C_tiley + 2),
                                  idxdiv2*(row_C_tiley + 3)
                                 );
            const uchar4 mask = (uchar4) (
                                  maskBias[maskIdx.x],
                                  maskBias[maskIdx.y],
                                  maskBias[maskIdx.z],
                                  maskBias[maskIdx.w]
                                 );
        }    
        sum0 += masked_value(bias_val, mask.x, nibble);
        sum1 += masked_value(bias_val, mask.y, nibble);
        sum2 += masked_value(bias_val, mask.z, nibble);
        sum3 += masked_value(bias_val, mask.w, nibble);
    }

    // Calculate the sigmoid function of the sum
    if(calcSigmoid) {        
        if(averageResultBeforeSigmoid) {
            sum0 *= 0.5f;
            sum1 *= 0.5f;
            sum2 *= 0.5f;
            sum3 *= 0.5f;
        }

        sum0 = activation_function(sum0);
        sum1 = activation_function(sum1);
        sum2 = activation_function(sum2);
        sum3 = activation_function(sum3);
    }
    // end of calculation of sigmoid function

    /* Write 16 values to matrixC */
    if(sumToMatrixC) {
        const float4 a = matrixC[globalPos.x] * multPrevVal;
        const float4 b = matrixC[globalPos.y] * multPrevVal;
        const float4 c = matrixC[globalPos.z] * multPrevVal;
        const float4 d = matrixC[globalPos.w] * multPrevVal;  

        matrixC[globalPos.x] =  a + multSum*sum0;
        matrixC[globalPos.y] =  b + multSum*sum1;
        matrixC[globalPos.z] =  c + multSum*sum2;
        matrixC[globalPos.w] =  d + multSum*sum3;    
    } else {
       matrixC[globalPos.x] = sum0;
       matrixC[globalPos.y] = sum1;
       matrixC[globalPos.z] = sum2;
       matrixC[globalPos.w] = sum3;
    }
}

/* Substracts element by element. NDRange of one dimension. 
 * Take care that every element is a float4 element.
 * The dimension should be the total number of elements divided by 4
 * This function is used to calculate the deltas of the output layer.
 */
__kernel void elementWiseSubstractKernel(__global float4 *A,
                                         __global float4 *B,
                                         __global float4* R,
                                         int offset_A,
                                         int offset_B,
                                         int offset_R)
{
    const int i = get_global_id(0);
    
    const float4 a = A[offset_A + i];
    const float4 b = B[offset_B + i];
    
    R[offset_R + i] =  a - b;
}

/* Adds element by element. NDRange of one dimension. 
 * Take care that every element is a float4 element.
 * The dimension should be the total number of elements divided by 4
 * This function is used to calculate the deltas of the output layer.
 */
__kernel void elementWiseSumKernel(__global float4* A,
                                   __global float4* B,
                                   __global float4* R,
                                   int offset_A,
                                   int offset_B,
                                   int offset_R,
                                   float mult_A,
                                   float mult_B)
{
    const int i = get_global_id(0);
    
    const float4 a = mult_A*A[offset_A + i];
    const float4 b = mult_B*B[offset_B + i];
       
    R[offset_R + i] =  a + b;

}

/*
 * Implements the function R = (A+a).*(B+b)
 * Where A and B are vectors and a and b are scalars. All the operations
 * involved are elementwise operations 
 */

__kernel void elementWiseMultKernel(__global float4* A,
                                    __global float4* B,
                                    __global float4* R,
                                    int offset_A,
                                    int offset_B,
                                    int offset_R,
                                    float a,
                                    float b)
{
    
    const int i = get_global_id(0);
    
    const float4 v1 = A[offset_A + i] + (float4) (a);
    const float4 v2 = B[offset_B + i] + (float4) (b);
       
    R[offset_R + i] =  v1 * v2;

}


__kernel void elementWiseMultiplicationBySigmoidDerivativeKernel(
                                         __global float4 *del,
                                         __global float4 *act,
                                         int offset_del,
                                         int offset_act)
{
    int i = get_global_id(0);

    float4 a = activation_function_derivative(act[offset_act + i]);
    
    del[offset_del + i] *= a;
}


__kernel void crossEntropyKernelLocal(__global float4* t, 
                                      __global float4* y, 
                                      __global float4* output, 
                                      __local float4* sdata,
                                      int offset_y)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    unsigned int stride = gid * 2;
    
    float4 y1 = y[offset_y + stride];
    float4 t1 = t[stride];
    float4 i1 = cross_entropy(t1, y1);
    
    float4 y2 = y[offset_y + stride + 1];
    float4 t2 = t[stride + 1];
    float4 i2 = cross_entropy(t2, y2);
    
    sdata[tid] = i1 + i2;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // do reduction in shared mem
    for(unsigned int s = localSize >> 1; s > 0; s >>= 1) 
    {
        if(tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) output[bid] = sdata[0];	
}

__kernel void level2RegularizationKernelLocal(__global float4* W, 
                                              __global float4* O, 
                                              __local float4* sdata)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    unsigned int stride = gid * 2;
    
    float4 w1 = W[stride];
    float4 i1 = w1*w1;
    
    float4 w2 = W[stride + 1];
    float4 i2 = w2*w2;
    
    sdata[tid] = i1 + i2;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    // do reduction in shared mem
    for(unsigned int s = localSize >> 1; s > 0; s >>= 1) 
    {
        if(tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) O[bid] = sdata[0];	
}


// Al finalizar la función se obtiene un vector de output de tamaño igual al número de grupos
// que hay que sumar, obteniendo el resultado final

/*
 * Calculates a softmax of local_size float4's => local_size*4 float elements 
 * Required local_size = number of output elements of the softmax to calculate divided by 4
 * Required global size = all the elements / 4 (floats4)
 */
__kernel void softmaxKernelLocal(__global float4* z, 
                                 __local float4* sdata,
                                 int offset_z)
{
    // load shared mem
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    
    unsigned int idx = offset_z + gid;
    
    sdata[lid] = z[idx];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // calculate the max value of all the elements
    float4 maxval = (float4) (-9E20);
    for(int i = 0; i < localSize; i++) {
        maxval = max(maxval, sdata[i]);
    }
    const float2 maxv2 = max((float2) (maxval.x, maxval.y), (float2) (maxval.z, maxval.w));
    const float maxv = max(maxv2.x, maxv2.y);
    
    float4 sum = (float4) (0.0f);
    for(int i = 0; i < localSize; i++) {
        sum += exp(sdata[i] - maxv);
    }    
    
    float total = sum.x + sum.y + sum.z + sum.w;
    
    
    z[idx] = exp(sdata[lid] - maxv)/total;
}

/* 
 *  1 dimensional NDRange = number of columns of floats / 4 
 *  Sums the values of all the rows
 */
__kernel void rowSumKernel(__global float4 * matrixA,
                           __global float4 *bias_inc,
                           int nrRowsA,
                           float multExisting,
                           float multNew)
{
    const int gid = get_global_id(0);
    const int gsz = get_global_size(0);

    float4 result = (float4) (0.0f);
    for(int i = 0; i < nrRowsA; i++) {
        const int idx = i*gsz + gid;
        result += matrixA[idx];
    }

    const float4 a = multExisting*bias_inc[gid];
    const float4 b = multNew*result;
    
    bias_inc[gid] = a + b;
}

/* 
 *  1 dimensional NDRange = number of columns of floats / 4 
 *  Sums the values of all the rows
 */
__kernel void matrixScalarMultiplicationKernel
                          (__global float4 * matrix,
                           float scalar)
{
    const int gid = get_global_id(0);
    matrix[gid] *= scalar;
}

//
// PRNG Xorshift+. 128 bits of state. Passes BigCrush.
// Uses two vectorized seeds and generates from them a
// vector of random numbers stored in vector rnd.
// 1D NDRange with global size equal to the size of the
// vectors s0, s1 and rnd that are assumed to have the same size 
//
__kernel void randomBitsGeneratorKernel(
                  __global uchar8 *s0,
                  __global uchar8 *s1,
                  __global uchar8 *rnd)
{
    int gid = get_global_id(0);
    
    union bit64 {
        uchar8 bytes;
        ulong  qword;
    } x, y, z;
        
    x.bytes = s0[gid];
    y.bytes = s1[gid];
    
    s0[gid] = y.bytes;
    
    x.qword ^= x.qword << 23; // a
    x.qword ^= x.qword >> 17;	// b
    x.qword ^= y.qword ^ (y.qword >> 26); // c
    
    s1[gid] = x.bytes;	
    
    z.qword = x.qword + y.qword;
    
    rnd[gid] = z.bytes;
}
