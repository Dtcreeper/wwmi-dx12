cmake_minimum_required(VERSION 3.20)
project(wwmi-dx12 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ReShade addons use the static CRT to avoid runtime dependencies
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ---------------------------------------------------------------------------
# ReShade addon (WWMI-DX32.addon64)
# ---------------------------------------------------------------------------
add_library(wwmi-dx12 SHARED
    src/addon_main.cpp
    src/ini_file.cpp
    src/mod_rules.cpp
    src/expr.cpp
    src/script_model.cpp
    src/script_runtime.cpp
    src/crc32c.cpp
    src/buffer_hash.cpp
    src/buffer_tracker.cpp
    src/texture_hash.cpp
    src/dds_loader.cpp
    src/texture_tracker.cpp
    src/override_engine.cpp
    src/pipeline_tracker.cpp
    src/skeleton_chain.cpp
    src/xxhash64.cpp
)

set_target_properties(wwmi-dx12 PROPERTIES
    OUTPUT_NAME "WWMI-DX12"
    SUFFIX ".addon64"
)

target_include_directories(wwmi-dx12 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/reshade/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(wwmi-dx12 PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

# ---------------------------------------------------------------------------
# Core unit tests (self-contained test runner, no external framework)
# ---------------------------------------------------------------------------
enable_testing()

add_executable(wwmi-tests
    tests/test_main.cpp
    tests/test_ini.cpp
    tests/test_mod_rules.cpp
    tests/test_expr.cpp
    tests/test_script_model.cpp
    tests/test_script_runtime.cpp
    tests/test_crc32c.cpp
    tests/test_buffer_hash.cpp
    tests/test_buffer_tracker.cpp
    tests/test_texture_hash.cpp
    tests/test_dds_loader.cpp
    tests/test_texture_tracker.cpp
    tests/test_override_engine.cpp
    tests/test_pipeline_tracker.cpp
    tests/test_skeleton_chain.cpp
    src/ini_file.cpp
    src/mod_rules.cpp
    src/expr.cpp
    src/script_model.cpp
    src/script_runtime.cpp
    src/crc32c.cpp
    src/buffer_hash.cpp
    src/buffer_tracker.cpp
    src/texture_hash.cpp
    src/dds_loader.cpp
    src/texture_tracker.cpp
    src/override_engine.cpp
    src/pipeline_tracker.cpp
    src/skeleton_chain.cpp
    src/xxhash64.cpp
)

target_include_directories(wwmi-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/external/reshade/include
)

target_compile_definitions(wwmi-tests PRIVATE
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
)

add_test(NAME wwmi-tests COMMAND wwmi-tests)

# Temporary sscanf semantics probe (diagnostics only)
add_executable(sscanf-probe tests/sscanf_probe.cpp)

# Temporary expr trace probe (diagnostics only)
add_executable(expr-probe tests/expr_probe.cpp src/expr.cpp)
target_include_directories(expr-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(expr-probe PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)

# Temporary runtime trace probe (diagnostics only)
add_executable(runtime-probe tests/runtime_probe.cpp
    src/script_runtime.cpp src/script_model.cpp src/expr.cpp src/ini_file.cpp
    src/mod_rules.cpp)
target_include_directories(runtime-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(runtime-probe PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)

# M6 diagnostics: brute-force 3DMigoto hash hypotheses over a view dump
add_executable(dump-probe tests/dump_probe.cpp src/crc32c.cpp)
target_include_directories(dump-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(dump-probe PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
