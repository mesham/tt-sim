#pragma once

#include <vector>
#include <cstdint>

namespace nekbone {

/**
 * SFPU matmul via outer product decomposition:
 *
 *   C = A * B  (32x32 tiles)
 *     = sum_{k=0}^{15} col_broadcast(A, k) .* row_broadcast(B, k)
 *
 * D/Dt are block-diagonal (same 16x16 block replicated), so 16 iterations
 * suffice for the full 32x32 matmul.
 *
 * Two cases depending on which operand is block-diagonal:
 *   blockdiag * full:  D * u,  Dt * ur,  D * u_trans
 *   full * blockdiag:  u * Dt, us * D,   ut * D
 */

constexpr int BLK = 16;
constexpr int TILE = 32;

/**
 * Column broadcasts of a block-diagonal matrix (left operand of blockdiag * full).
 *
 * For column k: tile[i][j] = M_16[i % 16][k]  for all i, j.
 * Same value in top and bottom halves (block-diagonal has identical blocks).
 *
 * @param blkdiag_32x32  Block-diagonal 32x32 matrix in row-major
 * @return 16 tiles, each 1024 floats in row-major
 */
inline std::vector<std::vector<float>> blockdiag_col_broadcasts(
    const std::vector<float>& blkdiag_32x32
) {
    std::vector<std::vector<float>> tiles(BLK, std::vector<float>(TILE * TILE, 0.0f));
    for (int k = 0; k < BLK; k++) {
        auto& t = tiles[k];
        for (int i = 0; i < TILE; i++) {
            float val = blkdiag_32x32[(i % BLK) * TILE + k];  // top-left block
            for (int j = 0; j < TILE; j++) {
                t[i * TILE + j] = val;
            }
        }
    }
    return tiles;
}

/**
 * Row broadcasts of a block-diagonal matrix (right operand of full * blockdiag).
 *
 * For row k: tile[i][j] = M_16[k][j % 16]  for all i, j.
 * Same value in left and right halves.
 *
 * @param blkdiag_32x32  Block-diagonal 32x32 matrix in row-major
 * @return 16 tiles, each 1024 floats in row-major
 */
inline std::vector<std::vector<float>> blockdiag_row_broadcasts(
    const std::vector<float>& blkdiag_32x32
) {
    std::vector<std::vector<float>> tiles(BLK, std::vector<float>(TILE * TILE, 0.0f));
    for (int k = 0; k < BLK; k++) {
        auto& t = tiles[k];
        for (int i = 0; i < TILE; i++) {
            for (int j = 0; j < TILE; j++) {
                t[i * TILE + j] = blkdiag_32x32[k * TILE + (j % BLK)];  // top-left block
            }
        }
    }
    return tiles;
}

/**
 * Row broadcasts of a full tile (right operand of blockdiag * full).
 *
 * For row k: tile[i][j] = src[k][j]      if i < 16
 *                        = src[k+16][j]   if i >= 16
 * Top half uses row k, bottom half uses row k+16.
 *
 * @param full_32x32  Full 32x32 tile in row-major (e.g. packed u tile)
 * @return 16 tiles, each 1024 floats in row-major
 */
inline std::vector<std::vector<float>> full_row_broadcasts(
    const std::vector<float>& full_32x32
) {
    std::vector<std::vector<float>> tiles(BLK, std::vector<float>(TILE * TILE, 0.0f));
    for (int k = 0; k < BLK; k++) {
        auto& t = tiles[k];
        // Top half: broadcast row k
        for (int i = 0; i < BLK; i++) {
            for (int j = 0; j < TILE; j++) {
                t[i * TILE + j] = full_32x32[k * TILE + j];
            }
        }
        // Bottom half: broadcast row k+16
        for (int i = 0; i < BLK; i++) {
            for (int j = 0; j < TILE; j++) {
                t[(i + BLK) * TILE + j] = full_32x32[(k + BLK) * TILE + j];
            }
        }
    }
    return tiles;
}

/**
 * Column broadcasts of a full tile (left operand of full * blockdiag).
 *
 * For column k: tile[i][j] = src[i][k]      if j < 16
 *                           = src[i][k+16]   if j >= 16
 * Left half uses column k, right half uses column k+16.
 *
 * @param full_32x32  Full 32x32 tile in row-major
 * @return 16 tiles, each 1024 floats in row-major
 */
inline std::vector<std::vector<float>> full_col_broadcasts(
    const std::vector<float>& full_32x32
) {
    std::vector<std::vector<float>> tiles(BLK, std::vector<float>(TILE * TILE, 0.0f));
    for (int k = 0; k < BLK; k++) {
        auto& t = tiles[k];
        for (int i = 0; i < TILE; i++) {
            float val_left = full_32x32[i * TILE + k];
            float val_right = full_32x32[i * TILE + k + BLK];
            // Left half columns
            for (int j = 0; j < BLK; j++) {
                t[i * TILE + j] = val_left;
            }
            // Right half columns
            for (int j = BLK; j < TILE; j++) {
                t[i * TILE + j] = val_right;
            }
        }
    }
    return tiles;
}

/**
 * Flatten broadcast tiles into a contiguous vector for tilization and upload.
 * Tiles are stacked vertically: width=32, height=32*num_tiles.
 */
inline std::vector<float> flatten_broadcast_tiles(
    const std::vector<std::vector<float>>& tiles
) {
    std::vector<float> flat(tiles.size() * TILE * TILE);
    for (size_t t = 0; t < tiles.size(); t++) {
        std::copy(tiles[t].begin(), tiles[t].end(), flat.begin() + t * TILE * TILE);
    }
    return flat;
}

}  // namespace nekbone
