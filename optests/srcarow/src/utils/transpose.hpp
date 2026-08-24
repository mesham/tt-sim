#pragma once

#include <vector>
#include <cstdint>

namespace nekbone {

/**
 * @brief Transpose element from (z,y,x) to (y,x,z) layout
 * 
 * Input:  element[z * 256 + y * 16 + x]
 * Output: transposed[y * 256 + x * 16 + z]
 * 
 * This reorganizes data so z-dimension becomes contiguous,
 * allowing D matrix to operate on z naturally.
 */
inline std::vector<double> transpose_zyx_to_yxz(
    const double* element,  // 4096 values in (z,y,x) order
    int nx, int ny, int nz
) {
    const int points_per_element = nx * ny * nz;
    std::vector<double> transposed(points_per_element);
    
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int z = 0; z < nz; z++) {
                int src_idx = z * (ny * nx) + y * nx + x;
                int dst_idx = y * (nx * nz) + x * nz + z;
                transposed[dst_idx] = element[src_idx];
            }
        }
    }
    
    return transposed;
}

/**
 * @brief Transpose element from (y,x,z) back to (z,y,x) layout
 * 
 * Reverse of transpose_zyx_to_yxz
 */
inline std::vector<double> transpose_yxz_to_zyx(
    const double* transposed,  // 4096 values in (y,x,z) order
    int nx, int ny, int nz
) {
    const int points_per_element = nx * ny * nz;
    std::vector<double> element(points_per_element);
    
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                int src_idx = y * (nx * nz) + x * nz + z;
                int dst_idx = z * (ny * nx) + y * nx + x;
                element[dst_idx] = transposed[src_idx];
            }
        }
    }
    
    return element;
}

/**
 * @brief Transpose all 4 elements from (z,y,x) to (y,x,z)
 */
inline std::vector<double> transpose_elements_zyx_to_yxz(
    const std::vector<double>& elements,
    int nx, int ny, int nz
) {
    constexpr int BATCH_SIZE = 4;
    const int points_per_element = nx * ny * nz;
    
    std::vector<double> transposed(BATCH_SIZE * points_per_element);
    
    for (int elem = 0; elem < BATCH_SIZE; elem++) {
        const double* elem_ptr = &elements[elem * points_per_element];
        auto elem_transposed = transpose_zyx_to_yxz(elem_ptr, nx, ny, nz);
        
        std::copy(
            elem_transposed.begin(),
            elem_transposed.end(),
            transposed.begin() + elem * points_per_element
        );
    }
    
    return transposed;
}

/**
 * @brief Transpose all 4 elements from (y,x,z) to (z,y,x)
 */
inline std::vector<double> transpose_elements_yxz_to_zyx(
    const std::vector<double>& transposed,
    int nx, int ny, int nz
) {
    constexpr int BATCH_SIZE = 4;
    const int points_per_element = nx * ny * nz;
    
    std::vector<double> elements(BATCH_SIZE * points_per_element);
    
    for (int elem = 0; elem < BATCH_SIZE; elem++) {
        const double* trans_ptr = &transposed[elem * points_per_element];
        auto elem_normal = transpose_yxz_to_zyx(trans_ptr, nx, ny, nz);
        
        std::copy(
            elem_normal.begin(),
            elem_normal.end(),
            elements.begin() + elem * points_per_element
        );
    }
    
    return elements;
}

}  // namespace nekbone