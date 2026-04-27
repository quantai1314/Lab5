#include <stdlib.h>
#include <string.h>

#include "convolution.h"

#define KERNEL_RADIUS (KERNEL_SIZE / 2)

/** 
 * 计算卷积核覆盖区域的加权和，包含边界检查。
 * 作为基线实现的一部分，保证正确性但性能较差。
 */
void calculateKernelResponse(int i, int j, int n, float *src, float *kernel, float *dst) {
    dst[i * n + j] = 0.0f;
    // 卷积核中心对称，ki, kj 从 0 到 KERNEL_SIZE-1
    int ki, kj;
    float pixel_value;
    for (kj = 0; kj < KERNEL_SIZE; ++kj) { //! 列优先访问，性能较差
        for (ki = 0; ki < KERNEL_SIZE; ++ki) {
            // 映射到 src 的偏移位置
            int ii = i + ki - KERNEL_SIZE / 2;
            int jj = j + kj - KERNEL_SIZE / 2;
            // 边界检查：如果 ii,jj 超出 src 范围，则视为 0
            if (ii >= 0 && ii < n && jj >= 0 && jj < n) {//! 引入大量分支，影响性能
                pixel_value = src[ii * n + jj];
            }
            else {
                pixel_value = 0.0f; 
            }
            dst[i * n + j] += pixel_value * kernel[ki * KERNEL_SIZE + kj];//! 读写内存开销较大
        }
    }
}

/**
 * 基线实现：直接在内层循环中进行边界检查，保证正确性但性能较差。
 * 作为性能优化的对照，并且提供了一个清晰的正确性基准。
 */
void baseline_convolution(int n, float *src, float *kernel, float *dst)
{
    int i, j;
    for (j = 0; j < n; ++j) {//! 列优先访问，性能较差
        for (i = 0; i < n; ++i) {
            // 对每个输出像素 (i,j)，计算卷积核覆盖区域的加权和
            calculateKernelResponse(i, j, n, src, kernel, dst);//! 调用函数（增加函数调用开销）
        }
    }
}


/**
 * 优化实现：请在此处实现你的卷积函数。
 */
void convolution(int n, float *src, float *kernel, float *dst)
{
    if (n <= KERNEL_SIZE) {
        baseline_convolution(n, src, kernel, dst);
        return;
    }

    const int last = n - KERNEL_RADIUS;
    const float k00 = kernel[0],  k01 = kernel[1],  k02 = kernel[2],  k03 = kernel[3],  k04 = kernel[4];
    const float k10 = kernel[5],  k11 = kernel[6],  k12 = kernel[7],  k13 = kernel[8],  k14 = kernel[9];
    const float k20 = kernel[10], k21 = kernel[11], k22 = kernel[12], k23 = kernel[13], k24 = kernel[14];
    const float k30 = kernel[15], k31 = kernel[16], k32 = kernel[17], k33 = kernel[18], k34 = kernel[19];
    const float k40 = kernel[20], k41 = kernel[21], k42 = kernel[22], k43 = kernel[23], k44 = kernel[24];
    int i, j;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            if (i < KERNEL_RADIUS || i >= last || j < KERNEL_RADIUS || j >= last) {
                int ki, kj;
                float sum = 0.0f;

                for (ki = 0; ki < KERNEL_SIZE; ++ki) {
                    const int ii = i + ki - KERNEL_RADIUS;
                    if (ii < 0 || ii >= n) {
                        continue;
                    }

                    for (kj = 0; kj < KERNEL_SIZE; ++kj) {
                        const int jj = j + kj - KERNEL_RADIUS;
                        if (jj >= 0 && jj < n) {
                            sum += src[ii * n + jj] * kernel[ki * KERNEL_SIZE + kj];
                        }
                    }
                }

                dst[i * n + j] = sum;
            }
            else {
                const float *r0 = src + (i - 2) * n + j;
                const float *r1 = r0 + n;
                const float *r2 = r1 + n;
                const float *r3 = r2 + n;
                const float *r4 = r3 + n;

                dst[i * n + j] =
                    r0[-2] * k00 + r0[-1] * k01 + r0[0] * k02 + r0[1] * k03 + r0[2] * k04 +
                    r1[-2] * k10 + r1[-1] * k11 + r1[0] * k12 + r1[1] * k13 + r1[2] * k14 +
                    r2[-2] * k20 + r2[-1] * k21 + r2[0] * k22 + r2[1] * k23 + r2[2] * k24 +
                    r3[-2] * k30 + r3[-1] * k31 + r3[0] * k32 + r3[1] * k33 + r3[2] * k34 +
                    r4[-2] * k40 + r4[-1] * k41 + r4[0] * k42 + r4[1] * k43 + r4[2] * k44;
            }
        }
    }
}
