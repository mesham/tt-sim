#pragma once

#include <vector>
#include <cmath>
#include <cstdio>

namespace nekbone {

/**
 * @brief Compare two vectors with tolerance
 * 
 * @param expected Expected values
 * @param actual Actual values
 * @param tolerance Comparison tolerance
 * @return true if all values match within tolerance
 */
inline bool validate_results(
    const std::vector<double>& expected,
    const std::vector<double>& actual,
    double tolerance = 1e-3
) {
    if (expected.size() != actual.size()) {
        printf("Size mismatch: expected %zu, got %zu\n", 
               expected.size(), actual.size());
        return false;
    }
    
    int mismatches = 0;
    
    for (size_t i = 0; i < expected.size(); i++) {
        double diff = std::abs(expected[i] - actual[i]);
        
        if (diff > tolerance) {
            if (mismatches < 10) {
                printf("  Mismatch at [%zu]: expected %.6f, got %.6f (diff=%.6f)\n",
                       i, expected[i], actual[i], diff);
            }
            mismatches++;
        }
    }
    
    if (mismatches > 0) {
        printf("Total mismatches: %d / %zu\n", mismatches, expected.size());
        return false;
    }
    
    return true;
}

}  // namespace nekbone