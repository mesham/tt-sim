#pragma once

#include <vector>
#include <cstdint>
#include <cstring>

namespace nekbone {

/**
 * @brief Create 16*16 scaled identity matrix (for testing)
 */
inline std::vector<double> create_scaled_identity_16x16(double scale) {
    std::vector<double> D(16 * 16, 0.0);
    for (int i = 0; i < 16; i++) {
        D[i * 16 + i] = scale;
    }
    return D;
}

/**
 * @brief Transpose a 16*16 matrix
 */
inline std::vector<double> transpose_16x16(const std::vector<double>& D) {
    std::vector<double> Dt(16 * 16);
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            Dt[j * 16 + i] = D[i * 16 + j];
        }
    }
    return Dt;
}

/**
 * @brief Expand 16*16 matrix to 32*32 block-diagonal (2 blocks)
 */
inline std::vector<float> create_block_diagonal_32x32(
    const std::vector<double>& D_16x16
) {
    
    
    constexpr int TILE_SIZE = 32;
    constexpr int BLOCK_SIZE = 16;
    constexpr int NUM_BLOCKS = 2;

    std::vector<float> D_32x32(TILE_SIZE * TILE_SIZE, 0.0f);

    for (int block = 0; block < NUM_BLOCKS; block++) {
        int block_row = block * BLOCK_SIZE;
        int block_col = block * BLOCK_SIZE;
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                int src_idx = i * BLOCK_SIZE + j;
                int dst_row = block_row + i;
                int dst_col = block_col + j;
                int dst_idx = dst_row * TILE_SIZE + dst_col;
                
                D_32x32[dst_idx] = static_cast<float>(D_16x16[src_idx]);
            }
        }
    }
    
    return D_32x32;
}

/**
 * @brief Create identity G matrix (3*3)
 */
inline std::vector<double> create_identity_G_3x3() {
    return {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
}

/**
 * @brief Apply 3*3 geometric matrix to gradient vector
 */
inline void apply_geometric_3x3(
    const double* G,
    double ur, double us, double ut,
    double& ur_new, double& us_new, double& ut_new
) {
    double g11 = G[0], g12 = G[1], g13 = G[2];
    double g22 = G[3], g23 = G[4], g33 = G[5];
    
    ur_new = g11*ur + g12*us + g13*ut;
    us_new = g12*ur + g22*us + g23*ut;
    ut_new = g13*ur + g23*us + g33*ut;
}

/**
 * @brief CPU reference: ur = D * u (x-direction)
 */
inline void cpu_matmul_D_u(
    const std::vector<double>& D,
    const double* u,
    double* result
) {
    constexpr int N = 16;
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += D[row * N + k] * u[k * N + col];
            }
            result[row * N + col] = sum;
        }
    }
}

/**
 * @brief CPU reference: us = u * D^T (y-direction)
 */
inline void cpu_matmul_u_Dt(
    const std::vector<double>& Dt,
    const double* u,
    double* result
) {
    constexpr int N = 16;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += u[i * N + k] * Dt[k * N + j];
            }
            result[i * N + j] = sum;
        }
    }
}

/**
 * @brief CORRECT CPU reference: ut = D * u (z-direction)
 * 
 * This is the PROPER z-gradient that accesses all z-layers!
 */
inline void cpu_compute_ut_proper(
    const std::vector<double>& D,
    const double* u_element,  // One element: 16*16*16
    double* ut_element,       // Output: 16*16*16
    int nx, int ny, int nz
) {
    const int points_per_layer = nx * ny;  // 256
    
    // For each (x,y) position in the 16*16 grid
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int xy_offset = y * nx + x;
            
            // Gather u values across all z-layers for this (x,y)
            std::vector<double> u_column(nz);
            for (int z = 0; z < nz; z++) {
                u_column[z] = u_element[z * points_per_layer + xy_offset];
            }
            
            // Apply D matrix: ut_column = D * u_column
            std::vector<double> ut_column(nz, 0.0);
            for (int i = 0; i < nz; i++) {
                for (int j = 0; j < nz; j++) {
                    ut_column[i] += D[i * nz + j] * u_column[j];
                }
            }
            
            // Scatter ut values back to all z-layers
            for (int z = 0; z < nz; z++) {
                ut_element[z * points_per_layer + xy_offset] = ut_column[z];
            }
        }
    }
}

/**
 * @brief Compute complete AX operator with CORRECT z-gradient
 */
inline std::vector<double> compute_ax_reference(
    const std::vector<double>& D,
    const std::vector<double>& Dt,
    const std::vector<double>& G,
    const std::vector<double>& u_elements,
    int nx, int ny, int nz
) {
    constexpr int BATCH_SIZE = 4;
    const int points_per_element = nx * ny * nz;
    const int points_per_layer = nx * ny;
    
    std::vector<double> w_elements(BATCH_SIZE * points_per_element);
    
    std::vector<double> ur_layer(points_per_layer);
    std::vector<double> us_layer(points_per_layer);
    std::vector<double> ut_element(points_per_element);  // Full element for ut
    std::vector<double> temp_layer(points_per_layer);
    
    for (int elem = 0; elem < BATCH_SIZE; elem++) {
        const double* u_elem = &u_elements[elem * points_per_element];
        
        // Compute ut ONCE per element (proper z-gradient)
        cpu_compute_ut_proper(D, u_elem, ut_element.data(), nx, ny, nz);
        
        // Process each z-layer
        for (int z = 0; z < nz; z++) {
            const double* u_layer = &u_elem[z * points_per_layer];
            const double* ut_layer = &ut_element[z * points_per_layer];
            double* w_layer = &w_elements[elem * points_per_element + z * points_per_layer];
            
            // Compute ur, us for this layer
            cpu_matmul_D_u(D, u_layer, ur_layer.data());
            cpu_matmul_u_Dt(Dt, u_layer, us_layer.data());
            
            // Apply geometric factors
            for (int i = 0; i < points_per_layer; i++) {
                double ur_old = ur_layer[i];
                double us_old = us_layer[i];
                double ut_old = ut_layer[i];
                
                apply_geometric_3x3(
                    G.data(),
                    ur_old, us_old, ut_old,
                    ur_layer[i], us_layer[i], ut_element[z * points_per_layer + i]
                );
            }
            
            // Accumulate: w = D^T*ur + us*D + ut*D
            cpu_matmul_u_Dt(Dt, ur_layer.data(), w_layer);
            
            cpu_matmul_u_Dt(D, us_layer.data(), temp_layer.data());
            for (int i = 0; i < points_per_layer; i++) {
                w_layer[i] += temp_layer[i];
            }
            
            cpu_matmul_u_Dt(D, &ut_element[z * points_per_layer], temp_layer.data());
            for (int i = 0; i < points_per_layer; i++) {
                w_layer[i] += temp_layer[i];
            }
        }
    }
    
    return w_elements;
}

}  // namespace nekbone