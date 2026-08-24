#pragma once

#include <vector>
#include <cstdint>

namespace nekbone {

/**
 * @brief Tilize FP32 data from row-major to TT hardware tile format.
 *
 * TT 32x32 tile layout: 4 faces (each 16x16, row-major within face)
 *   Face 0: rows  0-15, cols  0-15  (top-left)
 *   Face 1: rows  0-15, cols 16-31  (top-right)
 *   Face 2: rows 16-31, cols  0-15  (bottom-left)
 *   Face 3: rows 16-31, cols 16-31  (bottom-right)
 *
 * @param data         Row-major FP32 data
 * @param width        Number of columns (must be multiple of 32)
 * @param total_height Total number of rows (must be multiple of 32)
 * @return std::vector<float> Tilized data
 */
inline std::vector<float> tilize_nfaces_fp32(
    const std::vector<float>& data,
    int width,
    int total_height
) {
    const int num_tiles_x = width / 32;
    const int num_tiles_y = total_height / 32;
    std::vector<float> result(data.size());

    for (int ty = 0; ty < num_tiles_y; ty++) {
        for (int tx = 0; tx < num_tiles_x; tx++) {
            int tile_idx = ty * num_tiles_x + tx;
            int tile_base_dst = tile_idx * 1024;  // 32*32 elements per tile

            for (int r = 0; r < 32; r++) {
                for (int c = 0; c < 32; c++) {
                    int face = (r / 16) * 2 + (c / 16);
                    int local_r = r % 16;
                    int local_c = c % 16;

                    int src_idx = (ty * 32 + r) * width + (tx * 32 + c);
                    int dst_idx = tile_base_dst + face * 256 + local_r * 16 + local_c;
                    result[dst_idx] = data[src_idx];
                }
            }
        }
    }
    return result;
}

/**
 * @brief Untilize FP32 data from TT hardware tile format back to row-major.
 *
 * Inverse of tilize_nfaces_fp32.
 *
 * @param data         Tilized FP32 data
 * @param width        Number of columns (must be multiple of 32)
 * @param total_height Total number of rows (must be multiple of 32)
 * @return std::vector<float> Row-major data
 */
inline std::vector<float> untilize_nfaces_fp32(
    const std::vector<float>& data,
    int width,
    int total_height
) {
    const int num_tiles_x = width / 32;
    const int num_tiles_y = total_height / 32;
    std::vector<float> result(data.size());

    for (int ty = 0; ty < num_tiles_y; ty++) {
        for (int tx = 0; tx < num_tiles_x; tx++) {
            int tile_idx = ty * num_tiles_x + tx;
            int tile_base_src = tile_idx * 1024;

            for (int r = 0; r < 32; r++) {
                for (int c = 0; c < 32; c++) {
                    int face = (r / 16) * 2 + (c / 16);
                    int local_r = r % 16;
                    int local_c = c % 16;

                    int src_idx = tile_base_src + face * 256 + local_r * 16 + local_c;
                    int dst_idx = (ty * 32 + r) * width + (tx * 32 + c);
                    result[dst_idx] = data[src_idx];
                }
            }
        }
    }
    return result;
}

}  // namespace nekbone
