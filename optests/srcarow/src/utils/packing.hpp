#pragma once

#include <vector>
#include <cstdint>


namespace nekbone {

/**
 * @brief Pack one z-layer from 4 elements into a 32*32 tile
 * 
 * @param elem0 16*16 layer from element 0
 * @param elem1 16*16 layer from element 1
 * @param elem2 16*16 layer from element 2
 * @param elem3 16*16 layer from element 3
 * @return std::vector<float> 32*32 tile (1024 values)
 */
inline std::vector<float> pack_layer_to_tile(
    const double* elem0,
    const double* elem1,
    const double* elem2,
    const double* elem3
) {
    
    
    constexpr int TILE_SIZE = 32;
    constexpr int ELEM_SIZE = 16;
    std::vector<float> tile(TILE_SIZE * TILE_SIZE);
    
    // Pack into quadrants
    for (int y = 0; y < ELEM_SIZE; y++) {
        for (int x = 0; x < ELEM_SIZE; x++) {
            int src_idx = y * ELEM_SIZE + x;
            
            // Top-left: element 0
            tile[y * TILE_SIZE + x] = static_cast<float>(elem0[src_idx]);
            
            // Top-right: element 1
            tile[y * TILE_SIZE + (x + ELEM_SIZE)] = static_cast<float>(elem1[src_idx]);
            
            // Bottom-left: element 2
            tile[(y + ELEM_SIZE) * TILE_SIZE + x] = static_cast<float>(elem2[src_idx]);
            
            // Bottom-right: element 3
            tile[(y + ELEM_SIZE) * TILE_SIZE + (x + ELEM_SIZE)] = static_cast<float>(elem3[src_idx]);
        }
    }
    
    return tile;
}

/**
 * @brief Pack 4 elements into tiles (one tile per z-layer)
 * 
 * @param elements 4 elements, each 16*16*16 = 4096 points
 * @param nx Element dimension (16)
 * @param ny Element dimension (16)
 * @param nz Element dimension (16)
 * @return std::vector<float> Packed tiles
 */
inline std::vector<float> pack_elements(
    const std::vector<double>& elements,
    int nx, int ny, int nz
) {
    

    const int points_per_element = nx * ny * nz; // 16*16*16 = 4096
    const int points_per_layer = nx * ny; // 16*16 = 256
    const int tiles_per_batch = nz; // 16
    
    std::vector<float> tiles(tiles_per_batch * 1024);  // 16 tiles * 1024
    
    for (int z = 0; z < nz; z++) {
        // Extract 16*16 layers for this z from all 4 elements
        const double* elem0 = &elements[0 * points_per_element + z * points_per_layer];
        const double* elem1 = &elements[1 * points_per_element + z * points_per_layer];
        const double* elem2 = &elements[2 * points_per_element + z * points_per_layer];
        const double* elem3 = &elements[3 * points_per_element + z * points_per_layer];
        
        // Pack into one tile
        auto tile = pack_layer_to_tile(elem0, elem1, elem2, elem3);
        
        // Store in output
        std::copy(tile.begin(), tile.end(), tiles.begin() + z * 1024);
    }
    
    return tiles;
}

/**
 * @brief Unpack tiles back to 4 elements
 * 
 * @param tiles Packed tiles (16 tiles * 1024 values)
 * @param nx Element dimension (16)
 * @param ny Element dimension (16)
 * @param nz Element dimension (16)
 * @return std::vector<double> 4 elements unpacked
 */
inline std::vector<double> unpack_tiles(
    const std::vector<float>& tiles,
    int nx, int ny, int nz
) {
    
    
    constexpr int BATCH_SIZE = 4;
    constexpr int TILE_SIZE = 32;
    constexpr int ELEM_SIZE = 16;
    
    const int points_per_element = nx * ny * nz;
    const int points_per_layer = nx * ny;
    
    std::vector<double> elements(BATCH_SIZE * points_per_element);
    
    for (int z = 0; z < nz; z++) {
        const float* tile = &tiles[z * 1024];
        
        // Extract 4 quadrants
        for (int elem = 0; elem < BATCH_SIZE; elem++) {
            int offset_y = (elem / 2) * ELEM_SIZE;
            int offset_x = (elem % 2) * ELEM_SIZE;
            
            for (int y = 0; y < ELEM_SIZE; y++) {
                for (int x = 0; x < ELEM_SIZE; x++) {
                    int src_idx = (offset_y + y) * TILE_SIZE + (offset_x + x);
                    int dst_idx = elem * points_per_element + z * points_per_layer + y * nx + x;
                    
                    elements[dst_idx] = static_cast<float>(tile[src_idx]);
                }
            }
        }
    }
    
    return elements;
}

}  // namespace nekbone