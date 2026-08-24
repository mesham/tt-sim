#include <fmt/ostream.h>
#include <chrono>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

#include <tt-metalium/tensor_accessor_args.hpp>
#include "utils/tilize_fp32.hpp"
#include "utils/broadcast.hpp"

#include "utils/packing.hpp"
#include "utils/validation.hpp"
#include "utils/matrix_utils.hpp"
#include "utils/transpose.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main() {
    bool pass = true;

    try {
        fmt::print("=== NEKBONE: CPU & Tenstorrent (True SFPU) ===\n\n");

        // Configuration
        constexpr int nx = 16;
        constexpr int ny = 16;
        constexpr int nz = 16;
        constexpr int nelt = 4;
        constexpr int batch_size = 4;
        constexpr int num_batches = nelt / batch_size;              // 4/4 = 1 batch
        constexpr int points_per_element = nx * ny * nz;            // 16*16*16 = 4096 points
        constexpr int tiles_per_batch = nz;                         // 16 tiles
        constexpr int num_tiles = num_batches * tiles_per_batch;    // 1*16 = 16 tiles
        constexpr int BCAST_TILES = 16;

        fmt::print("Configuration: {} elements -> {}*{}*{}\n", nelt, nx, ny, nz);

        // Initialize device
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        auto& cq = mesh_device->mesh_command_queue();

        Program program_xy = CreateProgram();
        Program program_z = CreateProgram();
        Program program_accum = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        // Create test data
        std::vector<double> u_elements(nelt * points_per_element, 1.0);

        // Create matrices - use scaled identity to test
        fmt::print("Creating matrices...\n");
        constexpr double scale = 2.0;
        auto D_16x16 = nekbone::create_scaled_identity_16x16(scale);
        auto Dt_16x16 = nekbone::transpose_16x16(D_16x16);

        fmt::print("  D matrix: {}*Identity\n", scale);
        fmt::print("  Expected w: {}\n\n", scale * scale * 3.0);

        auto D_32x32 = nekbone::create_block_diagonal_32x32(D_16x16);
        auto Dt_32x32 = nekbone::create_block_diagonal_32x32(Dt_16x16);
        auto G_3x3 = nekbone::create_identity_G_3x3();

        // Pack u into 32x32 tiles
        auto u_tiles = nekbone::pack_elements(u_elements, nx, ny, nz);

        // Generate static broadcast tiles for D and Dt
        fmt::print("Generating broadcast tiles...\n");
        auto D_col_bcast_tiles  = nekbone::blockdiag_col_broadcasts(D_32x32);   // 16 tiles (D as left operand)
        auto Dt_row_bcast_tiles = nekbone::blockdiag_row_broadcasts(Dt_32x32);  // 16 tiles (Dt as right operand)
        auto D_row_bcast_tiles  = nekbone::blockdiag_row_broadcasts(D_32x32);   // 16 tiles (D as right operand)
        auto Dt_col_bcast_tiles = nekbone::blockdiag_col_broadcasts(Dt_32x32);  // 16 tiles (Dt as left operand)

        // Generate per-tile u broadcast tiles for Pass 1
        std::vector<std::vector<float>> u_row_bcast_all;  // 16*num_tiles tiles
        std::vector<std::vector<float>> u_col_bcast_all;  // 16*num_tiles tiles
        for (int t = 0; t < num_tiles; t++) {
            std::vector<float> u_tile(u_tiles.begin() + t * 32 * 32,
                                       u_tiles.begin() + (t + 1) * 32 * 32);
            auto row_bcasts = nekbone::full_row_broadcasts(u_tile);
            auto col_bcasts = nekbone::full_col_broadcasts(u_tile);
            u_row_bcast_all.insert(u_row_bcast_all.end(), row_bcasts.begin(), row_bcasts.end());
            u_col_bcast_all.insert(u_col_bcast_all.end(), col_bcasts.begin(), col_bcasts.end());
        }

        // Flatten and tilize all broadcast tiles
        auto D_col_bcast_flat  = nekbone::tilize_nfaces_fp32(nekbone::flatten_broadcast_tiles(D_col_bcast_tiles), 32, 32 * BCAST_TILES);
        auto Dt_row_bcast_flat = nekbone::tilize_nfaces_fp32(nekbone::flatten_broadcast_tiles(Dt_row_bcast_tiles), 32, 32 * BCAST_TILES);
        auto D_row_bcast_flat  = nekbone::tilize_nfaces_fp32(nekbone::flatten_broadcast_tiles(D_row_bcast_tiles), 32, 32 * BCAST_TILES);
        auto Dt_col_bcast_flat = nekbone::tilize_nfaces_fp32(nekbone::flatten_broadcast_tiles(Dt_col_bcast_tiles), 32, 32 * BCAST_TILES);
        auto u_row_bcast_flat  = nekbone::tilize_nfaces_fp32(nekbone::flatten_broadcast_tiles(u_row_bcast_all), 32, 32 * BCAST_TILES * num_tiles);
        auto u_col_bcast_flat  = nekbone::tilize_nfaces_fp32(nekbone::flatten_broadcast_tiles(u_col_bcast_all), 32, 32 * BCAST_TILES * num_tiles);

        // Buffer configuration
        constexpr uint32_t tile_size_bytes = 32 * 32 * sizeof(float);   // 4096 bytes

        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = tile_size_bytes,
            .buffer_type = BufferType::DRAM
        };

        distributed::ReplicatedBufferConfig bcast_16_config{.size = tile_size_bytes * BCAST_TILES};
        distributed::ReplicatedBufferConfig bcast_per_tile_config{.size = tile_size_bytes * BCAST_TILES * num_tiles};
        distributed::ReplicatedBufferConfig output_tiles_config{.size = tile_size_bytes * num_tiles};

        // Create DRAM buffers
        auto D_col_bcast_buf  = distributed::MeshBuffer::create(bcast_16_config, dram_config, mesh_device.get());
        auto Dt_row_bcast_buf = distributed::MeshBuffer::create(bcast_16_config, dram_config, mesh_device.get());
        auto D_row_bcast_buf  = distributed::MeshBuffer::create(bcast_16_config, dram_config, mesh_device.get());
        auto Dt_col_bcast_buf = distributed::MeshBuffer::create(bcast_16_config, dram_config, mesh_device.get());
        auto u_row_bcast_buf  = distributed::MeshBuffer::create(bcast_per_tile_config, dram_config, mesh_device.get());
        auto u_col_bcast_buf  = distributed::MeshBuffer::create(bcast_per_tile_config, dram_config, mesh_device.get());
        auto ur_buffer = distributed::MeshBuffer::create(output_tiles_config, dram_config, mesh_device.get());
        auto us_buffer = distributed::MeshBuffer::create(output_tiles_config, dram_config, mesh_device.get());
        auto w_buffer  = distributed::MeshBuffer::create(output_tiles_config, dram_config, mesh_device.get());

        // Upload broadcast tiles
        distributed::EnqueueWriteMeshBuffer(cq, D_col_bcast_buf, D_col_bcast_flat, false);
        distributed::EnqueueWriteMeshBuffer(cq, Dt_row_bcast_buf, Dt_row_bcast_flat, false);
        distributed::EnqueueWriteMeshBuffer(cq, D_row_bcast_buf, D_row_bcast_flat, false);
        distributed::EnqueueWriteMeshBuffer(cq, Dt_col_bcast_buf, Dt_col_bcast_flat, false);
        distributed::EnqueueWriteMeshBuffer(cq, u_row_bcast_buf, u_row_bcast_flat, false);
        distributed::EnqueueWriteMeshBuffer(cq, u_col_bcast_buf, u_col_bcast_flat, false);

        // ====================================
        // PASS 1: Compute ur and us on device
        // ====================================

        fmt::print("=== Pass 1: Computing ur, us on device ===\n");
        auto start_pass1 = std::chrono::high_resolution_clock::now();

        // Circular buffers for Pass 1
        // c_0: D column broadcasts (16 tiles, loaded once)
        CircularBufferConfig cb_D_col_p1(tile_size_bytes * BCAST_TILES, {{CBIndex::c_0, DataFormat::Float32}});
        cb_D_col_p1.set_page_size(CBIndex::c_0, tile_size_bytes);
        CreateCircularBuffer(program_xy, core, cb_D_col_p1);

        // c_1: u row broadcasts (16 tiles per z-layer)
        CircularBufferConfig cb_u_row_p1(tile_size_bytes * BCAST_TILES, {{CBIndex::c_1, DataFormat::Float32}});
        cb_u_row_p1.set_page_size(CBIndex::c_1, tile_size_bytes);
        CreateCircularBuffer(program_xy, core, cb_u_row_p1);

        // c_2: u col broadcasts (16 tiles per z-layer)
        CircularBufferConfig cb_u_col_p1(tile_size_bytes * BCAST_TILES, {{CBIndex::c_2, DataFormat::Float32}});
        cb_u_col_p1.set_page_size(CBIndex::c_2, tile_size_bytes);
        CreateCircularBuffer(program_xy, core, cb_u_col_p1);

        // c_3: Dt row broadcasts (16 tiles, loaded once)
        CircularBufferConfig cb_Dt_row_p1(tile_size_bytes * BCAST_TILES, {{CBIndex::c_3, DataFormat::Float32}});
        cb_Dt_row_p1.set_page_size(CBIndex::c_3, tile_size_bytes);
        CreateCircularBuffer(program_xy, core, cb_Dt_row_p1);

        // c_4: accumulator scratch (1 tile)
        CircularBufferConfig cb_accum_p1(tile_size_bytes, {{CBIndex::c_4, DataFormat::Float32}});
        cb_accum_p1.set_page_size(CBIndex::c_4, tile_size_bytes);
        CreateCircularBuffer(program_xy, core, cb_accum_p1);

        // c_5: tmp scratch (1 tile)
        CircularBufferConfig cb_tmp_p1(tile_size_bytes, {{CBIndex::c_5, DataFormat::Float32}});
        cb_tmp_p1.set_page_size(CBIndex::c_5, tile_size_bytes);
        CreateCircularBuffer(program_xy, core, cb_tmp_p1);

        // c_16: ur output (1 tile)
        CircularBufferConfig cb_ur_p1(tile_size_bytes, {{CBIndex::c_16, DataFormat::Float32}});
        cb_ur_p1.set_page_size(CBIndex::c_16, tile_size_bytes);
        CreateCircularBuffer(program_xy, core, cb_ur_p1);

        // c_17: us output (1 tile)
        CircularBufferConfig cb_us_p1(tile_size_bytes, {{CBIndex::c_17, DataFormat::Float32}});
        cb_us_p1.set_page_size(CBIndex::c_17, tile_size_bytes);
        CreateCircularBuffer(program_xy, core, cb_us_p1);

        // Reader XY kernel
        std::vector<uint32_t> reader_xy_compile_args;
        TensorAccessorArgs(*D_col_bcast_buf->get_backing_buffer()).append_to(reader_xy_compile_args);
        TensorAccessorArgs(*Dt_row_bcast_buf->get_backing_buffer()).append_to(reader_xy_compile_args);
        TensorAccessorArgs(*u_row_bcast_buf->get_backing_buffer()).append_to(reader_xy_compile_args);
        TensorAccessorArgs(*u_col_bcast_buf->get_backing_buffer()).append_to(reader_xy_compile_args);

        auto reader_xy_kernel = CreateKernel(
            program_xy,
            "kernels/reader_xy.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_xy_compile_args
            }
        );

        SetRuntimeArgs(program_xy, reader_xy_kernel, core, {
            D_col_bcast_buf->address(),
            Dt_row_bcast_buf->address(),
            u_row_bcast_buf->address(),
            u_col_bcast_buf->address(),
            num_tiles
        });

        // Compute XY kernel
        auto compute_xy_kernel = CreateKernel(
            program_xy,
            "kernels/compute_xy.cpp",
            core,
            ComputeConfig{.fp32_dest_acc_en = true}
        );

        SetRuntimeArgs(program_xy, compute_xy_kernel, core, {num_tiles});

        // Writer XY kernel
        std::vector<uint32_t> writer_xy_compile_args;
        TensorAccessorArgs(*ur_buffer->get_backing_buffer()).append_to(writer_xy_compile_args);
        TensorAccessorArgs(*us_buffer->get_backing_buffer()).append_to(writer_xy_compile_args);

        auto writer_xy_kernel = CreateKernel(
            program_xy,
            "kernels/writer_xy.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_xy_compile_args
            }
        );

        SetRuntimeArgs(program_xy, writer_xy_kernel, core, {
            ur_buffer->address(),
            us_buffer->address(),
            num_tiles
        });

        // Execute Pass 1
        distributed::MeshWorkload workload_xy;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload_xy.add_program(device_range, std::move(program_xy));
        distributed::EnqueueMeshWorkload(cq, workload_xy, false);
        distributed::Finish(cq);

        auto end_pass1 = std::chrono::high_resolution_clock::now();
        auto duration_pass1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_pass1 - start_pass1);
        fmt::print("Pass 1 complete ({} ms)\n\n", duration_pass1.count());

        // =======================================
        // PASS 2: Compute ut with host transpose
        // =======================================

        fmt::print("=== Pass 2: Computing ut with transpose ===\n");
        auto start_pass2 = std::chrono::high_resolution_clock::now();

        // Transpose (z,y,x) -> (y,x,z)
        fmt::print("  Transposing (z,y,x) -> (y,x,z)...\n");
        auto u_transposed = nekbone::transpose_elements_zyx_to_yxz(u_elements, nx, ny, nz);

        // Pack transposed u and generate broadcast tiles
        fmt::print("  Generating transposed u broadcast tiles...\n");
        auto u_trans_tiles = nekbone::pack_elements(u_transposed, nx, ny, nz);

        std::vector<std::vector<float>> ut_row_bcast_all;
        for (int t = 0; t < num_tiles; t++) {
            std::vector<float> u_trans_tile(u_trans_tiles.begin() + t * 32 * 32,
                                             u_trans_tiles.begin() + (t + 1) * 32 * 32);
            auto row_bcasts = nekbone::full_row_broadcasts(u_trans_tile);
            ut_row_bcast_all.insert(ut_row_bcast_all.end(), row_bcasts.begin(), row_bcasts.end());
        }

        auto ut_row_bcast_flat = nekbone::tilize_nfaces_fp32(
            nekbone::flatten_broadcast_tiles(ut_row_bcast_all), 32, 32 * BCAST_TILES * num_tiles);

        auto ut_row_bcast_buf = distributed::MeshBuffer::create(bcast_per_tile_config, dram_config, mesh_device.get());
        distributed::EnqueueWriteMeshBuffer(cq, ut_row_bcast_buf, ut_row_bcast_flat, false);

        // Circular buffers for Pass 2
        // c_0: D column broadcasts (16 tiles, loaded once)
        CircularBufferConfig cb_D_col_p2(tile_size_bytes * BCAST_TILES, {{CBIndex::c_0, DataFormat::Float32}});
        cb_D_col_p2.set_page_size(CBIndex::c_0, tile_size_bytes);
        CreateCircularBuffer(program_z, core, cb_D_col_p2);

        // c_1: ut row broadcasts (16 tiles per z-layer)
        CircularBufferConfig cb_ut_row_p2(tile_size_bytes * BCAST_TILES, {{CBIndex::c_1, DataFormat::Float32}});
        cb_ut_row_p2.set_page_size(CBIndex::c_1, tile_size_bytes);
        CreateCircularBuffer(program_z, core, cb_ut_row_p2);

        // c_4: accumulator (1 tile)
        CircularBufferConfig cb_accum_p2(tile_size_bytes, {{CBIndex::c_4, DataFormat::Float32}});
        cb_accum_p2.set_page_size(CBIndex::c_4, tile_size_bytes);
        CreateCircularBuffer(program_z, core, cb_accum_p2);

        // c_5: tmp (1 tile)
        CircularBufferConfig cb_tmp_p2(tile_size_bytes, {{CBIndex::c_5, DataFormat::Float32}});
        cb_tmp_p2.set_page_size(CBIndex::c_5, tile_size_bytes);
        CreateCircularBuffer(program_z, core, cb_tmp_p2);

        // c_16: ut output (1 tile)
        CircularBufferConfig cb_ut_p2(tile_size_bytes, {{CBIndex::c_16, DataFormat::Float32}});
        cb_ut_p2.set_page_size(CBIndex::c_16, tile_size_bytes);
        CreateCircularBuffer(program_z, core, cb_ut_p2);

        // Reader Z kernel
        std::vector<uint32_t> reader_z_compile_args;
        TensorAccessorArgs(*D_col_bcast_buf->get_backing_buffer()).append_to(reader_z_compile_args);
        TensorAccessorArgs(*ut_row_bcast_buf->get_backing_buffer()).append_to(reader_z_compile_args);

        auto reader_z_kernel = CreateKernel(
            program_z,
            "kernels/reader_z.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_z_compile_args
            }
        );

        SetRuntimeArgs(program_z, reader_z_kernel, core, {
            D_col_bcast_buf->address(),
            ut_row_bcast_buf->address(),
            num_tiles
        });

        // Compute Z kernel
        auto compute_z_kernel = CreateKernel(
            program_z,
            "kernels/compute_z.cpp",
            core,
            ComputeConfig{.fp32_dest_acc_en = true}
        );

        SetRuntimeArgs(program_z, compute_z_kernel, core, {num_tiles});

        // Writer Z kernel
        auto ut_trans_buffer = distributed::MeshBuffer::create(output_tiles_config, dram_config, mesh_device.get());

        std::vector<uint32_t> writer_z_compile_args;
        TensorAccessorArgs(*ut_trans_buffer->get_backing_buffer()).append_to(writer_z_compile_args);

        auto writer_z_kernel = CreateKernel(
            program_z,
            "kernels/writer_z.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_z_compile_args
            }
        );

        SetRuntimeArgs(program_z, writer_z_kernel, core, {
            ut_trans_buffer->address(),
            num_tiles
        });

        // Execute Pass 2
        distributed::MeshWorkload workload_z;
        workload_z.add_program(device_range, std::move(program_z));
        distributed::EnqueueMeshWorkload(cq, workload_z, false);
        distributed::Finish(cq);

        // Download ut_transposed, transpose back
        fmt::print("  Downloading ut_transposed...\n");
        std::vector<float> ut_trans_tiles_raw(num_tiles * 1024);
        distributed::EnqueueReadMeshBuffer(cq, ut_trans_tiles_raw, ut_trans_buffer, true);

        fmt::print("  Transposing back (y,x,z) -> (z,y,x)...\n");
        auto ut_trans_tiles_rowmajor = nekbone::untilize_nfaces_fp32(ut_trans_tiles_raw, 32, 32 * num_tiles);
        auto ut_transposed = nekbone::unpack_tiles(ut_trans_tiles_rowmajor, nx, ny, nz);
        auto ut_elements = nekbone::transpose_elements_yxz_to_zyx(ut_transposed, nx, ny, nz);

        // Repack ut into tiles (row-major 32x32)
        auto ut_tiles = nekbone::pack_elements(ut_elements, nx, ny, nz);

        auto end_pass2 = std::chrono::high_resolution_clock::now();
        auto duration_pass2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_pass2 - start_pass2);
        fmt::print("Pass 2 complete ({} ms)\n\n", duration_pass2.count());

        // =====================
        // PASS 3: Accumulation
        // =====================

        fmt::print("=== Pass 3: Accumulation on device ===\n");
        auto start_pass3 = std::chrono::high_resolution_clock::now();

        // Download ur and us from device to generate broadcast tiles
        fmt::print("  Downloading ur, us from device...\n");
        std::vector<float> ur_tiles_raw(num_tiles * 1024);
        std::vector<float> us_tiles_raw(num_tiles * 1024);
        distributed::EnqueueReadMeshBuffer(cq, ur_tiles_raw, ur_buffer, true);
        distributed::EnqueueReadMeshBuffer(cq, us_tiles_raw, us_buffer, true);

        auto ur_tiles_rowmajor = nekbone::untilize_nfaces_fp32(ur_tiles_raw, 32, 32 * num_tiles);
        auto us_tiles_rowmajor = nekbone::untilize_nfaces_fp32(us_tiles_raw, 32, 32 * num_tiles);

        // Generate broadcast tiles for Pass 3:
        //   w = Dt*ur + us*D + ut*D
        //   Round 1: Dt(col) * ur(row)  -> blockdiag_col(Dt) .* full_row(ur)
        //   Round 2: us(col) * D(row)   -> full_col(us) .* blockdiag_row(D)
        //   Round 3: ut(col) * D(row)   -> full_col(ut) .* blockdiag_row(D)
        fmt::print("  Generating broadcast tiles for accumulation...\n");
        std::vector<std::vector<float>> ur_row_bcast_all;
        std::vector<std::vector<float>> us_col_bcast_all;
        std::vector<std::vector<float>> ut_col_bcast_all;

        for (int t = 0; t < num_tiles; t++) {
            std::vector<float> ur_tile(ur_tiles_rowmajor.begin() + t * 32 * 32,
                                        ur_tiles_rowmajor.begin() + (t + 1) * 32 * 32);
            auto bcasts = nekbone::full_row_broadcasts(ur_tile);
            ur_row_bcast_all.insert(ur_row_bcast_all.end(), bcasts.begin(), bcasts.end());

            std::vector<float> us_tile(us_tiles_rowmajor.begin() + t * 32 * 32,
                                        us_tiles_rowmajor.begin() + (t + 1) * 32 * 32);
            auto bcasts2 = nekbone::full_col_broadcasts(us_tile);
            us_col_bcast_all.insert(us_col_bcast_all.end(), bcasts2.begin(), bcasts2.end());

            std::vector<float> ut_tile(ut_tiles.begin() + t * 32 * 32,
                                        ut_tiles.begin() + (t + 1) * 32 * 32);
            auto bcasts3 = nekbone::full_col_broadcasts(ut_tile);
            ut_col_bcast_all.insert(ut_col_bcast_all.end(), bcasts3.begin(), bcasts3.end());
        }

        auto ur_row_bcast_flat = nekbone::tilize_nfaces_fp32(
            nekbone::flatten_broadcast_tiles(ur_row_bcast_all), 32, 32 * BCAST_TILES * num_tiles);
        auto us_col_bcast_flat = nekbone::tilize_nfaces_fp32(
            nekbone::flatten_broadcast_tiles(us_col_bcast_all), 32, 32 * BCAST_TILES * num_tiles);
        auto ut_col_bcast_flat = nekbone::tilize_nfaces_fp32(
            nekbone::flatten_broadcast_tiles(ut_col_bcast_all), 32, 32 * BCAST_TILES * num_tiles);

        auto ur_row_bcast_buf = distributed::MeshBuffer::create(bcast_per_tile_config, dram_config, mesh_device.get());
        auto us_col_bcast_buf = distributed::MeshBuffer::create(bcast_per_tile_config, dram_config, mesh_device.get());
        auto ut_col_bcast_buf = distributed::MeshBuffer::create(bcast_per_tile_config, dram_config, mesh_device.get());

        distributed::EnqueueWriteMeshBuffer(cq, ur_row_bcast_buf, ur_row_bcast_flat, false);
        distributed::EnqueueWriteMeshBuffer(cq, us_col_bcast_buf, us_col_bcast_flat, false);
        distributed::EnqueueWriteMeshBuffer(cq, ut_col_bcast_buf, ut_col_bcast_flat, false);

        // Circular buffers for Pass 3
        // c_0: left broadcasts (16 tiles)
        CircularBufferConfig cb_left_p3(tile_size_bytes * BCAST_TILES, {{CBIndex::c_0, DataFormat::Float32}});
        cb_left_p3.set_page_size(CBIndex::c_0, tile_size_bytes);
        CreateCircularBuffer(program_accum, core, cb_left_p3);

        // c_1: right broadcasts (16 tiles)
        CircularBufferConfig cb_right_p3(tile_size_bytes * BCAST_TILES, {{CBIndex::c_1, DataFormat::Float32}});
        cb_right_p3.set_page_size(CBIndex::c_1, tile_size_bytes);
        CreateCircularBuffer(program_accum, core, cb_right_p3);

        // c_4: accum (1 tile)
        CircularBufferConfig cb_accum_p3(tile_size_bytes, {{CBIndex::c_4, DataFormat::Float32}});
        cb_accum_p3.set_page_size(CBIndex::c_4, tile_size_bytes);
        CreateCircularBuffer(program_accum, core, cb_accum_p3);

        // c_5: tmp (1 tile)
        CircularBufferConfig cb_tmp_p3(tile_size_bytes, {{CBIndex::c_5, DataFormat::Float32}});
        cb_tmp_p3.set_page_size(CBIndex::c_5, tile_size_bytes);
        CreateCircularBuffer(program_accum, core, cb_tmp_p3);

        // c_6: mm_out (1 tile)
        CircularBufferConfig cb_mm_out_p3(tile_size_bytes, {{CBIndex::c_6, DataFormat::Float32}});
        cb_mm_out_p3.set_page_size(CBIndex::c_6, tile_size_bytes);
        CreateCircularBuffer(program_accum, core, cb_mm_out_p3);

        // c_7: sum (1 tile)
        CircularBufferConfig cb_sum_p3(tile_size_bytes, {{CBIndex::c_7, DataFormat::Float32}});
        cb_sum_p3.set_page_size(CBIndex::c_7, tile_size_bytes);
        CreateCircularBuffer(program_accum, core, cb_sum_p3);

        // c_16: w output (1 tile)
        CircularBufferConfig cb_w_p3(tile_size_bytes, {{CBIndex::c_16, DataFormat::Float32}});
        cb_w_p3.set_page_size(CBIndex::c_16, tile_size_bytes);
        CreateCircularBuffer(program_accum, core, cb_w_p3);

        // Reader accum kernel
        std::vector<uint32_t> reader_accum_compile_args;
        TensorAccessorArgs(*Dt_col_bcast_buf->get_backing_buffer()).append_to(reader_accum_compile_args);
        TensorAccessorArgs(*D_row_bcast_buf->get_backing_buffer()).append_to(reader_accum_compile_args);
        TensorAccessorArgs(*ur_row_bcast_buf->get_backing_buffer()).append_to(reader_accum_compile_args);
        TensorAccessorArgs(*us_col_bcast_buf->get_backing_buffer()).append_to(reader_accum_compile_args);
        TensorAccessorArgs(*ut_col_bcast_buf->get_backing_buffer()).append_to(reader_accum_compile_args);

        auto reader_accum_kernel = CreateKernel(
            program_accum,
            "kernels/reader_accum.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_accum_compile_args
            }
        );

        SetRuntimeArgs(program_accum, reader_accum_kernel, core, {
            Dt_col_bcast_buf->address(),
            D_row_bcast_buf->address(),
            ur_row_bcast_buf->address(),
            us_col_bcast_buf->address(),
            ut_col_bcast_buf->address(),
            num_tiles
        });

        // Compute accum kernel
        auto compute_accum_kernel = CreateKernel(
            program_accum,
            "kernels/compute_accum.cpp",
            core,
            ComputeConfig{.fp32_dest_acc_en = true}
        );

        SetRuntimeArgs(program_accum, compute_accum_kernel, core, {num_tiles});

        // Writer accum kernel
        std::vector<uint32_t> writer_accum_compile_args;
        TensorAccessorArgs(*w_buffer->get_backing_buffer()).append_to(writer_accum_compile_args);

        auto writer_accum_kernel = CreateKernel(
            program_accum,
            "kernels/writer_accum.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_accum_compile_args
            }
        );

        SetRuntimeArgs(program_accum, writer_accum_kernel, core, {
            w_buffer->address(),
            num_tiles
        });

        // Execute Pass 3
        distributed::MeshWorkload workload_accum;
        workload_accum.add_program(device_range, std::move(program_accum));
        distributed::EnqueueMeshWorkload(cq, workload_accum, false);
        distributed::Finish(cq);

        auto end_pass3 = std::chrono::high_resolution_clock::now();
        auto duration_pass3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_pass3 - start_pass3);
        fmt::print("Pass 3 complete ({} ms)\n\n", duration_pass3.count());

        // ============================================================
        // Validation
        // ============================================================

        fmt::print("=== Validation ===\n");

        std::vector<float> w_tiles(num_tiles * 1024);
        distributed::EnqueueReadMeshBuffer(cq, w_tiles, w_buffer, true);

        auto w_tiles_rowmajor = nekbone::untilize_nfaces_fp32(w_tiles, 32, 32 * num_tiles);
        auto w_elements = nekbone::unpack_tiles(w_tiles_rowmajor, nx, ny, nz);

        auto w_reference = nekbone::compute_ax_reference(D_16x16, Dt_16x16, G_3x3, u_elements, nx, ny, nz);

        double expected_value = scale * scale * 3.0;
        fmt::print("Expected w: {:.2f}\n\n", expected_value);

        for (int e = 0; e < nelt; e++) {
            std::vector<double> expected(
                w_reference.begin() + e * points_per_element,
                w_reference.begin() + (e + 1) * points_per_element
            );

            std::vector<double> actual(
                w_elements.begin() + e * points_per_element,
                w_elements.begin() + (e + 1) * points_per_element
            );

            if (nekbone::validate_results(expected, actual, 1e-2)) {
                fmt::print(" Element {}: All {} values match (w = {:.2f})\n",
                           e, points_per_element, expected_value);
            } else {
                fmt::print(" Element {}: Validation FAILED\n", e);
                pass = false;
            }
        }

        if (pass) {
            fmt::print("\n=== NEKBONE: CPU -> Tenstorrent (True SFPU) [SUCCESSFUL!] ===\n");
        }

        // Performance summary
        auto total_time = duration_pass1.count() + duration_pass2.count() + duration_pass3.count();
        fmt::print("\n=== Performance Summary ===\n");
        fmt::print("Pass 1 (ur,us): {} ms\n", duration_pass1.count());
        fmt::print("Pass 2 (ut):    {} ms (includes transpose)\n", duration_pass2.count());
        fmt::print("Pass 3 (accum): {} ms\n", duration_pass3.count());
        fmt::print("Total:          {} ms\n", total_time);

        // Cleanup
        mesh_device->close();

    } catch (const std::exception& e) {
        fmt::print(stderr, "Exception: {}\n", e.what());
        return 1;
    }

    fmt::print("\nResult: {}\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
