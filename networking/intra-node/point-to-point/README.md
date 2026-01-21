# Point-to-Point Communication Profiling

Systematic characterization of GPU-to-GPU communication bandwidth and latency for simulator validation.

## Table of Contents
1. [Objective](#objective)
2. [Experimental Design](#experimental-design)
3. [Implementation Status](#implementation-status)
4. [Configuration Tracking](#configuration-tracking)
5. [Thermal Management](#thermal-management)
6. [Clock Drift Measurement](#clock-drift-measurement)
7. [Implementation Details](#implementation-details)
8. [Next Steps](#next-steps)

---

## Objective

Collect detailed performance data on intra-node GPU-to-GPU point-to-point communications to validate network simulator predictions across:
- Message sizes: 1 byte → 1 GB (latency-bound → bandwidth-bound)
- All GPU pairs in both directions
- Multiple repetitions for statistical significance

---

## Experimental Design

### Core Strategy

**What we measure:**
- Point-to-point bandwidth and latency between all GPU pairs
- Behavior across message size spectrum
- Both directions for each link (GPU 0→1 and GPU 1→0 tested separately)

**How we control:**
- Pseudo-random order (fixed seed for reproducibility)
- Warmup phase per message size
- Blocking operations (completion guarantees)
- Thermal state monitoring
- Clock drift characterization
- Profiler annotations for config tracking

### Message Size Sweep

```python
SMALL_SIZES = [2**i for i in range(0, 11)]   # 1B to 1KB (latency-bound)
MEDIUM_SIZES = [2**i for i in range(11, 20)] # 2KB to 512KB (transition)
LARGE_SIZES = [2**i for i in range(20, 31)]  # 1MB to 1GB (bandwidth-bound)
```

**31 message sizes total** spanning 10 orders of magnitude

### GPU Pair Selection

For N GPUs, test all directed pairs:
```python
pairs = [(src, dst) for src in range(N) for dst in range(N) if src != dst]
```

**For 4 GPUs:** 12 directed pairs
- Tests both directions: (0→1) and (1→0) are separate tests
- Covers all NVLink connections
- Detects asymmetric links

**Total configurations:** 12 pairs × 31 sizes = **372 configs**

### Profiling Strategy

**Communication primitive:** `torch.distributed.send()` / `recv()` with NCCL backend
- Blocking operations (natural completion guarantees)
- Leverages NVLink when available
- Fallback to PCIe

**Multi-process architecture:** `torchrun` with N processes (one per GPU)
- More representative of real distributed workloads
- Separate traces per rank
- Timestamp synchronization via barriers

**Profiling infrastructure:**
- `torch.profiler.profile()` for GPU traces
- `ExecutionTraceObserver()` for CPU traces
- Consistent with `compute/matmul/` experiments

### Fairness Considerations

1. **Warmup phase:** 5 iterations per configuration
   - Initializes NCCL buffers
   - Compiles kernels
   - Allocates memory

2. **Pseudo-random ordering:**
   - Shuffle configs with fixed seed (42)
   - Eliminates systematic biases
   - Reproducible

3. **Buffer management:**
   - Pre-allocate all message sizes during warmup
   - Reuse buffers during profiling
   - Eliminates allocation overhead

4. **Thermal state:**
   - Initial warmup to 50°C
   - Continuous temperature monitoring
   - Optional periodic re-warming if needed

5. **Clock speeds:**
   - Lock GPU clocks (if sudo access)
   - Monitor clocks throughout
   - Document in metadata

6. **System isolation:**
   - Check for running processes
   - Set `CUDA_VISIBLE_DEVICES`
   - Minimize interference

---

## Implementation Status

### Completed: Utilities

All shared utilities implemented in `networking/utils/`:

#### 1. **distributed_utils.py** (119 lines)
- `init_distributed()` - Setup NCCL with torchrun
- `cleanup_distributed()` - Teardown
- `get_rank()`, `get_world_size()` - Process info
- `barrier_all()` - Synchronization
- `print_once()` - Rank 0 printing

#### 2. **profiling_utils.py** (160 lines)
- `ProfilerContext` - Combined CPU + GPU profiling
- `configure_profiler()` - Standard schedule
- `warmup_phase()` - Pre-profiling warmup
- `torch_deterministic()` - Reproducible seeds

#### 3. **metadata_utils.py** (179 lines)
- `capture_gpu_metadata()` - GPU properties, temps, clocks, power
- `capture_topology()` - NVLink/PCIe via nvidia-smi
- `capture_nccl_info()` - NCCL version and config
- `save_metadata()` - Write to YAML

#### 4. **sync_utils.py** (180 lines)
- `SyncPoint` - CPU/GPU timestamp pair
- `create_sync_barrier()` - Barrier with timestamp capture
- `measure_clock_drift()` - GPU vs CPU clock drift (default: 30s)
- `characterize_drift_multi_window()` - Multi-window drift analysis
- `compute_alignment_offset()` - Timestamp alignment

#### 5. **launch_utils.py** (219 lines)
- `validate_environment()` - Pre-flight checks
- `lock_gpu_clocks()` / `unlock_gpu_clocks()` - Clock management
- `thermal_warmup()` - GPU thermal stabilization (60s to 50°C)
- `check_gpu_idle()` - Check for running processes
- `print_environment_summary()` - System info

#### 6. **monitoring_utils.py** (285 lines) ✨ NEW
- `TemperatureLogger` - Background temperature/power/clock logging
- `quick_warmup()` - Light 10s warmup for maintenance
- `check_temperature()` - Quick temp check

**Total:** ~1,141 lines of reusable, documented, linter-clean utilities

### ⏳ To Do: Experiment-Specific Code

Files to create in `networking/intra-node/point-to-point/`:

1. **`p2p_config.py`** - Configuration parameters
2. **`profile_p2p.py`** - Main profiling script
3. **`launch_p2p.sh`** - Convenience launcher
4. **`analysis/`** - Post-processing scripts
   - `parse_p2p_traces.py`
   - `align_timestamps.py`
   - `plot_bandwidth_latency.py`

---

## Configuration Tracking

### Challenge

Each trace file contains hundreds of communication events. Need to map each event to its configuration (src, dst, message size, repetition).

### Solution: Profiler Annotations + Execution Log

#### 1. Profiler Annotations (Primary Method)

Embed config info directly in trace event names:

```python
def run_p2p_test(rank, config, buffer_pool, repetition):
    # Create descriptive name
    name = (f"p2p_cfg{config['config_id']:04d}_"
            f"s{config['src']}d{config['dst']}_"
            f"sz{config['size']}_"
            f"r{repetition}")
    
    with torch.profiler.record_function(name):
        if rank == config['src']:
            tensor = buffer_pool[config['size']]
            dist.send(tensor, dst=config['dst'])
        elif rank == config['dst']:
            tensor = buffer_pool[config['size']]
            dist.recv(tensor, src=config['src'])
```

**In trace file:**
```json
{
  "traceEvents": [
    {
      "name": "p2p_cfg0042_s0d1_sz1048576_r0",
      "ts": 1000000,
      "dur": 50,
      ...
    }
  ]
}
```

**Parsing:**
```python
import re

def parse_config_from_name(name):
    """Parse: p2p_cfg0042_s0d1_sz1048576_r0"""
    match = re.match(r'p2p_cfg(\d+)_s(\d+)d(\d+)_sz(\d+)_r(\d+)', name)
    if match:
        return {
            'config_id': int(match.group(1)),
            'src': int(match.group(2)),
            'dst': int(match.group(3)),
            'size': int(match.group(4)),
            'repetition': int(match.group(5)),
        }
```

**Pros:**
- Direct mapping: name → config
- Easy to parse and filter
- Visible in Chrome trace viewer
- No timestamp matching needed

**Overhead:** ~0.1-1 microseconds per annotation (negligible)

#### 2. Execution Log CSV (Backup Method)

Real-time logging of execution order with CPU timestamps:

```python
execution_log = []

for rep in range(REPEAT_COUNT):
    for config in configs:
        start_time = time.perf_counter_ns()
        
        # Execute with annotation
        with torch.profiler.record_function(name):
            run_p2p_test(rank, config, buffer_pool)
        
        end_time = time.perf_counter_ns()
        
        # Log execution (rank 0 only)
        if rank == 0:
            execution_log.append({
                'config_id': config['config_id'],
                'src': config['src'],
                'dst': config['dst'],
                'size': config['size'],
                'repetition': rep,
                'timestamp_start_ns': start_time,
                'timestamp_end_ns': end_time,
                'duration_ns': end_time - start_time,
            })

# Save after experiment
save_csv(execution_log, './traces/execution_log.csv')
```

**Output CSV:**
```csv
config_id,src,dst,size,repetition,timestamp_start_ns,timestamp_end_ns,duration_ns
0,0,1,1024,0,1234567890123456,1234567890173456,50000
0,0,1,1024,1,1234567891234567,1234567891284567,50000
1,0,1,2048,0,1234567892345678,1234567892397678,52000
...
```

**Pros:**
- CPU-side timestamps (complementary to GPU traces)
- Backup if profiler annotations fail
- Execution order documentation
- Additional metadata

---

## Thermal Management

### Strategy: Monitor with Optional Re-warming

Communication workloads generate less heat than compute. GPUs will naturally cool during the experiment - this is realistic and expected.

**Approach:**
1. **Initial warmup:** Heat all GPUs to 50°C (60 seconds)
2. **Monitor continuously:** Background thread logs temp/power/clocks (1 second intervals)
3. **Analyze post-hoc:** Check if cooling affects performance
4. **If needed:** Add periodic re-warming in subsequent runs

### Implementation

#### Initial Warmup
```python
from networking.utils import thermal_warmup

thermal_warmup(device=f'cuda:{rank}', duration=60, target_temp_min=50)
```

Runs matrix multiplications until GPU reaches 50°C or 60 seconds elapsed.

#### Continuous Monitoring
```python
from networking.utils import TemperatureLogger

# Start (rank 0 only)
if rank == 0:
    temp_logger = TemperatureLogger(
        gpu_ids=list(range(world_size)),
        interval=1.0,  # Sample every second
        metrics=['temperature', 'power', 'clock_graphics']
    )
    temp_logger.start()

# ... run experiment ...

# Stop and save
if rank == 0:
    temp_logger.stop()
    temp_logger.save('./traces/temperatures.csv')
    temp_logger.print_summary()
```

**Output:**
```
Temperature logger started:
  GPUs: [0, 1, 2, 3]
  Interval: 1.0s
  Metrics: ['temperature', 'power', 'clock_graphics']

... experiment runs ...

======================================================================
TEMPERATURE LOGGER SUMMARY
======================================================================

GPU_0:
  Temperature: 52.3°C ± 2.1°C (range: 48-56°C)
  Power:       85.2W ± 12.3W
  Clock:       1410 MHz (range: 1395-1410 MHz)

GPU_1:
  Temperature: 51.8°C ± 2.0°C (range: 47-55°C)
  Power:       84.1W ± 11.8W
  Clock:       1410 MHz (range: 1395-1410 MHz)
...
======================================================================

Temperature data saved to ./traces/temperatures.csv (1247 samples)
```

**Files generated:**
- `traces/temperatures.csv` - Full time-series data
- Analysis can correlate performance with thermal state

#### Optional: Periodic Re-warming

If temperature drops significantly and performance correlates with temp:

```python
from networking.utils import quick_warmup, check_temperature

# Every 20 configs, check and re-warm if needed
if i % 20 == 0 and i > 0:
    current_temp = check_temperature(rank)
    
    if current_temp and current_temp < 45:  # Below threshold
        quick_warmup(device=f'cuda:{rank}', duration=10, target_temp_min=48)
        dist.barrier()
```

**Decision:** Start without this. Add only if post-analysis shows temperature correlation with performance variance.

---

## Clock Drift Measurement

### Challenge

Each GPU has an independent clock. Need to verify clocks are synchronized for cross-GPU timestamp correlation.

### Solution: Quick Drift Checks (30 seconds)

**For main experiment:**
- **Start:** 30s drift measurement
- **End:** 30s drift measurement
- **Total overhead:** ~60 seconds

**30 seconds provides:**
- ±0.5 ppm accuracy (0.00005% drift)
- Sufficient to detect issues
- Not overkill for regular experiments

**Implementation:**
```python
from networking.utils import measure_clock_drift

# At experiment start
drift_start = measure_clock_drift(rank, duration_seconds=30)

if rank == 0:
    drift_ppm = (drift_start['drift_ratio'] - 1.0) * 1e6
    print(f"Initial drift: {drift_ppm:.2f} ppm")

# ... run experiment ...

# At experiment end
drift_end = measure_clock_drift(rank, duration_seconds=30)

if rank == 0:
    drift_ppm_end = (drift_end['drift_ratio'] - 1.0) * 1e6
    print(f"Final drift: {drift_ppm_end:.2f} ppm")
    
    drift_change = abs(drift_end['drift_ratio'] - drift_start['drift_ratio']) * 1e6
    if drift_change > 1.0:
        print(f"⚠ Drift changed by {drift_change:.2f} ppm during experiment")
```

**Output:**
```
Rank 0: Measuring clock drift...
Initial drift: 0.15 ppm
... experiment runs (20 minutes) ...
Final drift: 0.17 ppm
✓ Drift stable (changed by 0.02 ppm)
```

**Interpretation:**
- **< 1 ppm:** Excellent, clocks well-synchronized
- **1-10 ppm:** Good, acceptable for most purposes
- **> 10 ppm:** Investigate hardware

### Optional: Precision Characterization

Separate 5-minute measurement to document hardware (run once):

```python
# characterize_drift.py - separate script
drift = measure_clock_drift(rank, duration_seconds=300)
```

This provides high-precision drift characterization but is **not part of regular experiments**.

---

## Implementation Details

### Dummy Data Creation

```python
def create_communication_buffer(message_size_bytes, dtype=torch.float32):
    """Create buffer for communication testing."""
    bytes_per_element = 4  # float32
    num_elements = message_size_bytes // bytes_per_element
    
    # Random non-zero values (prevents hardware sparse optimizations)
    buffer = torch.randn(num_elements, dtype=dtype, device='cuda')
    return buffer

# Pre-allocate buffer pool during warmup
buffer_pool = {
    size: create_communication_buffer(size)
    for size in MESSAGE_SIZES
}

# Reuse during profiling
tensor = buffer_pool[message_size]
```

**Why random non-zero data:**
- Prevents sparse hardware optimizations
- Realistic (real data is rarely all zeros)
- Tests actual bandwidth (random data can't be compressed)

### NUMA Pinning (Deferred)

**What it is:** Pinning processes to CPU sockets matching their GPU's physical location.

**Status:** Deferred for now.
- Start without it
- Check variance in initial results
- Add in second run to quantify benefit

**How to check if needed:**
```bash
nvidia-smi topo -m  # Check if multi-socket
numactl --hardware  # Check NUMA nodes
```

### Experiment Timeline

```
Start
  ↓
Initial warmup (60s) ────────────────────── Heat GPUs to 50°C
  ↓
Start temp logger (background) ──────────── Monitor throughout
  ↓
Drift check (30s) ───────────────────────── Baseline clock sync
  ↓
Allocate buffer pool ────────────────────── Pre-allocate all sizes
  ↓
Warmup phase (5 iterations × 372 configs) ─ Compile kernels, init NCCL
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Main Profiling (10 repetitions × 372 configs)
  With profiler annotations
  With execution logging
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ↓
Drift check (30s) ───────────────────────── Check stability
  ↓
Stop temp logger & save ─────────────────── Save CSV
  ↓
Save execution log ──────────────────────── Save CSV
  ↓
Save metadata ───────────────────────────── Per-rank YAML
  ↓
Done!

Total overhead: ~2 minutes
Main experiment: ~20-30 minutes
```

---

## Next Steps

### Phase 1: Core Implementation

1. **`p2p_config.py`**
   - Message size definitions
   - Config generation function
   - Profiling parameters

2. **`profile_p2p.py`**
   - Import utilities
   - Implement P2P test with annotations
   - Main execution loop with logging
   - Integrate temperature monitoring
   - Integrate drift measurement

3. **`launch_p2p.sh`**
   - Environment setup
   - Optional clock locking
   - Torchrun invocation
   - Cleanup

### Phase 2: Analysis

1. **`analysis/parse_p2p_traces.py`**
   - Parse profiler traces
   - Extract annotated events
   - Build dataframe

2. **`analysis/align_timestamps.py`**
   - Apply drift corrections
   - Align across ranks

3. **`analysis/plot_bandwidth_latency.py`**
   - Bandwidth vs message size
   - Latency measurements
   - Per-link comparisons
   - Topology correlation

### Phase 3: Validation

1. Run full experiment on real hardware
2. Analyze variance (should be < 5%)
3. Check thermal stability
4. Verify timestamp alignment
5. Compare to simulator

---

## Output Files

After running experiment:

```
traces/<gpu_type>-<driver>-<cuda>/
├── rank_0.pt.trace.json          # GPU trace (with annotated names)
├── rank_0_CPU_trace.json         # CPU trace
├── rank_1.pt.trace.json
├── rank_1_CPU_trace.json
├── rank_2.pt.trace.json
├── rank_2_CPU_trace.json
├── rank_3.pt.trace.json
├── rank_3_CPU_trace.json
├── execution_log.csv             # Config execution order & timing
├── temperatures.csv              # Temperature/power/clock time-series
├── metadata_rank_0.yml           # Per-rank metadata
├── metadata_rank_1.yml
├── metadata_rank_2.yml
└── metadata_rank_3.yml
```

---

## Expected Results

### Bandwidth vs Message Size

**Small messages (< 4KB):** Latency-bound
- Dominated by protocol overhead
- ~2-10 microseconds per operation
- Bandwidth increases linearly with size

**Medium messages (4KB - 1MB):** Transition region
- Non-linear behavior
- Bandwidth increases sub-linearly
- Protocol overhead diminishing

**Large messages (> 1MB):** Bandwidth-bound
- Approaches asymptotic bandwidth
- NVLink: ~300-400 GB/s (H100)
- PCIe Gen4 x16: ~25-30 GB/s

### Latency Floor

**NVLink direct:** ~2-5 microseconds
**NVLink multi-hop:** +1-2 µs per hop
**PCIe:** ~5-10 microseconds

### Success Criteria

**Functional:**
- ✅ All configs execute without errors
- ✅ Traces generated for all ranks
- ✅ All message sizes complete
- ✅ Reproducible with fixed seed

**Quality:**
- ✅ Low variance (CV < 5%)
- ✅ Bandwidth approaches theoretical max for large messages
- ✅ Latency stable for small messages
- ✅ Clock drift < 0.1%
- ✅ No thermal throttling

**Analysis:**
- ✅ Clear bandwidth vs size curves
- ✅ Distinct latency-bound vs bandwidth-bound regimes
- ✅ Topology effects visible (NVLink vs PCIe)
- ✅ Comparable to simulator predictions

---

## References

- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
- Similar approach: `compute/matmul/` (this repo)

