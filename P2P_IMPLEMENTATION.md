# Vulkan P2P GPU Communications Implementation

This document describes the P2P (peer-to-peer) GPU communications implementation for the Vulkan backend in llama.cpp.

## Overview

The P2P implementation enables direct device-to-device memory transfers between multiple Vulkan GPUs, bypassing the CPU and host memory. This provides significant performance improvements for multi-GPU workloads.

## Current Status

✅ **Phase 1: Extension Detection and Capability Querying** - COMPLETE
- Added P2P fields to `vk_device_struct` and `vk_instance_t`
- Extension detection for VK_KHR_device_group, VK_KHR_external_memory, VK_KHR_external_memory_fd
- Peer memory feature querying via `ggml_vk_detect_peer_memory_features()`
- Environment variable control: `GGML_VK_ENABLE_P2P`

✅ **Phase 2: P2P Buffer Allocation** - COMPLETE
- Extended `vk_buffer_struct` with P2P support fields
- Implemented `ggml_vk_create_p2p_buffer()` for exportable buffer creation
- Implemented `ggml_vk_import_buffer_to_peer()` for cross-device buffer sharing
- File descriptor-based memory sharing (Linux)

✅ **Phase 3: P2P Transfer Operations** - COMPLETE
- Implemented `ggml_vk_buffer_copy_p2p()` for direct device-to-device copies
- Modified `ggml_vk_buffer_copy()` to use P2P when available
- Automatic fallback to host staging when P2P unavailable
- Proper memory barriers for synchronization

✅ **Phase 4: Testing and Validation** - COMPLETE
- Added `ggml_vk_test_p2p()` test function
- Optional initialization testing via `GGML_VK_TEST_P2P` environment variable

## Features

### Platform Support
- ✅ **Linux**: Full support using POSIX file descriptors
- ⏳ **Windows**: Planned (requires VK_KHR_external_memory_win32)
- ⏳ **macOS**: Not applicable (MoltenVK limitations)

### GPU Support
- ✅ **Identical GPUs**: Full support (same vendor/model)
- ⏳ **Non-identical GPUs**: Planned future enhancement

### Performance
- **Expected bandwidth**: 20-50 GB/s (PCIe 4.0 x16)
- **Current staging bandwidth**: 10-25 GB/s
- **Expected speedup**: 1.5-3x for large tensor transfers (>10 MB)

## Usage

### Enabling P2P

Set the environment variable before running:

```bash
export GGML_VK_ENABLE_P2P=1
```

P2P is **disabled by default** for safety and compatibility.

### Optional Testing

To run P2P self-tests during initialization:

```bash
export GGML_VK_TEST_P2P=1
export GGML_VULKAN_DEBUG=1  # For detailed logging
```

### Verification

Check logs for these messages:
- `P2P mode enabled via GGML_VK_ENABLE_P2P`
- `P2P enabled: device X <-> device Y`
- `ggml_vk_buffer_copy(P2P, ...)` (not `HOST_STAGING`)

### Example

```bash
# Enable P2P and debug logging
export GGML_VK_ENABLE_P2P=1
export GGML_VULKAN_DEBUG=1

# Run inference
./llama-cli -m model.gguf -p "test prompt" --n-gpu-layers 32
```

## Implementation Details

### Key Functions

1. **`ggml_vk_detect_peer_memory_features()`**
   - Called during device initialization
   - Queries peer memory capabilities between devices
   - Populates `peer_device_indices` and `peer_memory_features`

2. **`ggml_vk_create_p2p_buffer()`**
   - Creates buffers with external memory export capability
   - Exports file descriptor for sharing (Linux)
   - Sets `p2p_exportable` flag on success

3. **`ggml_vk_import_buffer_to_peer()`**
   - Imports an exported buffer to a peer device
   - Creates peer buffer handle and memory
   - Stores in `peer_buffers` and `peer_memories` maps

4. **`ggml_vk_buffer_copy_p2p()`**
   - Direct device-to-device memory copy
   - Uses proper memory barriers for synchronization
   - Executed on source device's transfer queue

5. **`ggml_vk_buffer_copy()` (modified)**
   - Checks P2P availability between devices
   - Uses P2P when available, falls back to host staging
   - On-demand buffer import if needed

### Data Structures

**`vk_device_struct` additions:**
```cpp
bool device_group_support;
bool external_memory_support;
bool external_memory_fd_support;
std::vector<size_t> peer_device_indices;
std::map<size_t, vk::PeerMemoryFeatureFlags> peer_memory_features;
```

**`vk_buffer_struct` additions:**
```cpp
bool p2p_exportable;
int export_fd;
std::map<size_t, vk::Buffer> peer_buffers;
std::map<size_t, vk::DeviceMemory> peer_memories;
std::vector<size_t> accessible_devices;
```

## Backward Compatibility

- **Single GPU**: Completely unchanged behavior
- **P2P disabled**: Uses existing host staging path
- **P2P unavailable**: Automatic fallback to host staging
- **No API changes**: All changes are internal to Vulkan backend

## Requirements

### Vulkan Extensions
- `VK_KHR_external_memory` - External memory sharing (required)
- `VK_KHR_external_memory_fd` - File descriptor import/export (Linux, required)
- `VK_KHR_device_group` - Multi-device operations (optional, for future enhancements)

### Runtime Requirements
- Vulkan 1.2 or higher
- GPU drivers with P2P support
- 2+ identical GPUs on Linux

## Testing

### Manual Testing

1. **Enable P2P and run self-test:**
   ```bash
   export GGML_VK_ENABLE_P2P=1
   export GGML_VK_TEST_P2P=1
   export GGML_VULKAN_DEBUG=1
   ./llama-cli -m model.gguf -p "test"
   ```

2. **Check for P2P usage:**
   ```bash
   # Look for these in logs:
   # "P2P enabled: device 0 <-> device 1"
   # "ggml_vk_buffer_copy(P2P, ...)"
   ```

3. **Performance comparison:**
   ```bash
   # Without P2P
   unset GGML_VK_ENABLE_P2P
   time ./llama-cli -m model.gguf -p "test" > /dev/null
   
   # With P2P
   export GGML_VK_ENABLE_P2P=1
   time ./llama-cli -m model.gguf -p "test" > /dev/null
   ```

### Expected Behavior

**P2P Available:**
- Log: `P2P enabled: device X <-> device Y`
- Log: `P2P buffer exported with FD: N`
- Log: `ggml_vk_buffer_copy(P2P, ...)`
- Faster cross-device transfers

**P2P Unavailable (fallback):**
- Log: `P2P NOT available: device X <-> device Y`
- Log: `ggml_vk_buffer_copy(HOST_STAGING, ...)`
- Normal host staging performance

## Known Limitations

1. **Linux only**: Windows support requires additional work
2. **Identical GPUs only**: Different GPUs may have incompatible memory
3. **File descriptor limits**: Linux per-process FD limits apply
4. **Driver support**: Some older drivers may not support all features

## Future Enhancements

1. **Windows Support** (~6-8 hours)
   - Add VK_KHR_external_memory_win32
   - Use Win32 handles instead of file descriptors

2. **Split Buffer Types** (~8-12 hours)
   - Tensor row splitting across devices
   - Follow CUDA backend patterns

3. **Non-Identical GPU Support** (~10-15 hours)
   - Relax UUID matching
   - Handle different memory capabilities

4. **Collective Operations**
   - AllReduce, AllGather for multi-GPU training
   - Ring/tree topology optimization

## Code Changes Summary

**Files Modified:**
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (~500 lines added)

**Key Changes:**
- Added P2P fields to device and buffer structures
- Extension detection and enablement
- P2P buffer creation and import functions
- P2P copy implementation with fallback
- Test function for validation

**No Changes:**
- `ggml/include/ggml-vulkan.h` (public API unchanged)
- Other backend files
- Build system

## Debugging

### Enable Debug Logging
```bash
export GGML_VULKAN_DEBUG=1
```

### Common Issues

**"P2P NOT available":**
- Check if devices support `VK_KHR_device_group`
- Verify GPUs are identical (same vendor/model)
- Ensure `GGML_VK_ENABLE_P2P=1` is set

**"Buffer not exportable":**
- Check if `VK_KHR_external_memory_fd` is supported
- Verify driver support for external memory

**Falling back to host staging:**
- Normal behavior when P2P unavailable
- Check logs for specific failure reason

## Performance Tuning

- P2P works best for large transfers (>1 MB)
- Small transfers may be faster through host memory
- Monitor with `GGML_VK_PERF_LOGGER=1`

## References

- [Vulkan Specification - Device Groups](https://registry.khronos.org/vulkan/specs/1.3/html/chap4.html#fundamentals-devicegroup)
- [VK_KHR_external_memory](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory.html)
- [Implementation Plan](/Users/steforster/.claude/plans/cozy-seeking-catmull.md)
