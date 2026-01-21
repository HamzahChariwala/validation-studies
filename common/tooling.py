"""
Tooling functions for parsing and linking PyTorch profiling traces.

This module provides utilities to:
1. Map operations between CPU and GPU traces
2. Extract metadata from trace operations
"""

import json
from typing import Dict, List, Any, Optional, Tuple


def op_to_kernel_tracing(
    cpu_op_id: int,
    cpu_trace: Dict[str, Any],
    gpu_trace: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Map a CPU trace operation to corresponding GPU trace operations.
    
    Args:
        cpu_op_id: The unique 'id' of the operation in the CPU trace
        cpu_trace: The loaded CPU trace dictionary
        gpu_trace: The loaded GPU trace dictionary
        
    Returns:
        List of GPU trace events that correspond to this CPU operation.
        This includes the cpu_op, cuda_runtime operations, and kernel executions.
        
    Algorithm:
        1. Find the CPU operation by id
        2. Extract its rf_id
        3. Find the cpu_op in GPU trace with matching "Record function id"
        4. Extract the "External id" from that cpu_op
        5. Return all operations in GPU trace with that "External id"
    """
    # Find the CPU operation by id
    cpu_op = None
    for node in cpu_trace.get('nodes', []):
        if node.get('id') == cpu_op_id:
            cpu_op = node
            break
    
    if cpu_op is None:
        raise ValueError(f"CPU operation with id {cpu_op_id} not found in CPU trace")
    
    # Extract rf_id from the CPU operation
    rf_id = None
    for attr in cpu_op.get('attrs', []):
        if attr.get('name') == 'rf_id':
            rf_id = attr.get('value')
            break
    
    if rf_id is None:
        raise ValueError(f"CPU operation {cpu_op_id} does not have an rf_id attribute")
    
    # Find the cpu_op in GPU trace with matching "Record function id"
    external_id = None
    for event in gpu_trace.get('traceEvents', []):
        if event.get('cat') == 'cpu_op' and \
           event.get('args', {}).get('Record function id') == rf_id:
            external_id = event.get('args', {}).get('External id')
            break
    
    if external_id is None:
        # The operation might not have launched any GPU kernels
        return []
    
    # Find all operations with the same External id
    gpu_operations = []
    for event in gpu_trace.get('traceEvents', []):
        event_external_id = event.get('args', {}).get('External id')
        if event_external_id == external_id:
            gpu_operations.append(event)
    
    return gpu_operations


def kernel_to_op_tracing(
    kernel_identifier: Tuple[int, int, int, int],  # (timestamp, duration, stream, external_id)
    gpu_trace: Dict[str, Any],
    cpu_trace: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Map a GPU trace kernel/operation back to the corresponding CPU trace operation.
    
    Args:
        kernel_identifier: Tuple of (timestamp, duration, stream, external_id) that 
                          uniquely identifies a GPU operation
        gpu_trace: The loaded GPU trace dictionary
        cpu_trace: The loaded CPU trace dictionary
        
    Returns:
        The corresponding CPU trace operation node, or None if not found.
        
    Algorithm:
        1. Find the GPU operation using the identifier
        2. Extract its "External id"
        3. Find a cpu_op in GPU trace with that "External id"
        4. Extract the "Record function id" from that cpu_op
        5. Find the CPU operation with matching rf_id
    """
    ts, dur, stream, external_id = kernel_identifier
    
    # Find the cpu_op in GPU trace with the matching External id
    record_function_id = None
    for event in gpu_trace.get('traceEvents', []):
        if event.get('cat') == 'cpu_op' and \
           event.get('args', {}).get('External id') == external_id:
            record_function_id = event.get('args', {}).get('Record function id')
            break
    
    if record_function_id is None:
        return None
    
    # Find the CPU operation with matching rf_id
    for node in cpu_trace.get('nodes', []):
        for attr in node.get('attrs', []):
            if attr.get('name') == 'rf_id' and attr.get('value') == record_function_id:
                return node
    
    return None


def operation_metadata_scrape(
    cpu_op_id: int,
    cpu_trace: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a CPU trace operation.
    
    Args:
        cpu_op_id: The unique 'id' of the operation in the CPU trace
        cpu_trace: The loaded CPU trace dictionary
        
    Returns:
        Dictionary containing extracted metadata including:
        - id: operation id
        - name: operation name
        - ctrl_deps: control dependency id
        - inputs: structured input data with tensor info
        - outputs: structured output data with tensor info
        - attrs: all attributes including rf_id, fw_parent, seq_id, etc.
    """
    # Find the CPU operation by id
    cpu_op = None
    for node in cpu_trace.get('nodes', []):
        if node.get('id') == cpu_op_id:
            cpu_op = node
            break
    
    if cpu_op is None:
        raise ValueError(f"CPU operation with id {cpu_op_id} not found in CPU trace")
    
    # Parse attributes into a more accessible format
    attrs_dict = {}
    for attr in cpu_op.get('attrs', []):
        attrs_dict[attr.get('name')] = attr.get('value')
    
    # Parse inputs - extract tensor information
    # Format: [tensor_id, storage_id, offset, numel, itemsize, device]
    inputs = cpu_op.get('inputs', {})
    parsed_inputs = []
    
    for idx, value in enumerate(inputs.get('values', [])):
        input_info = {
            'index': idx,
            'value': value,
            'shape': inputs.get('shapes', [])[idx] if idx < len(inputs.get('shapes', [])) else None,
            'type': inputs.get('types', [])[idx] if idx < len(inputs.get('types', [])) else None,
            'stride': inputs.get('strides', [])[idx] if idx < len(inputs.get('strides', [])) else None,
        }
        
        # Parse tensor info if it's in the standard format [tensor_id, storage_id, offset, numel, itemsize, device]
        if isinstance(value, list) and len(value) == 6:
            input_info['tensor_id'] = value[0]
            input_info['storage_id'] = value[1]
            input_info['offset'] = value[2]
            input_info['numel'] = value[3]
            input_info['itemsize'] = value[4]
            input_info['device'] = value[5]
        
        parsed_inputs.append(input_info)
    
    # Parse outputs - extract tensor information
    outputs = cpu_op.get('outputs', {})
    parsed_outputs = []
    
    for idx, value in enumerate(outputs.get('values', [])):
        output_info = {
            'index': idx,
            'value': value,
            'shape': outputs.get('shapes', [])[idx] if idx < len(outputs.get('shapes', [])) else None,
            'type': outputs.get('types', [])[idx] if idx < len(outputs.get('types', [])) else None,
            'stride': outputs.get('strides', [])[idx] if idx < len(outputs.get('strides', [])) else None,
        }
        
        # Parse tensor info if it's in the standard format
        if isinstance(value, list) and len(value) == 6:
            output_info['tensor_id'] = value[0]
            output_info['storage_id'] = value[1]
            output_info['offset'] = value[2]
            output_info['numel'] = value[3]
            output_info['itemsize'] = value[4]
            output_info['device'] = value[5]
        
        parsed_outputs.append(output_info)
    
    # Compile metadata
    metadata = {
        'id': cpu_op.get('id'),
        'name': cpu_op.get('name'),
        'ctrl_deps': cpu_op.get('ctrl_deps'),
        'inputs': parsed_inputs,
        'outputs': parsed_outputs,
        'rf_id': attrs_dict.get('rf_id'),
        'fw_parent': attrs_dict.get('fw_parent'),
        'seq_id': attrs_dict.get('seq_id'),
        'scope': attrs_dict.get('scope'),
        'tid': attrs_dict.get('tid'),
        'fw_tid': attrs_dict.get('fw_tid'),
        'op_schema': attrs_dict.get('op_schema'),
        'kernel_backend': attrs_dict.get('kernel_backend'),
        'kernel_file': attrs_dict.get('kernel_file'),
    }
    
    return metadata


def kernel_metadata_scrape(
    kernel_identifier: Tuple[int, int, Optional[int], int],  # (timestamp, duration, stream, external_id)
    gpu_trace: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Extract comprehensive metadata from a GPU trace operation/kernel.
    
    Args:
        kernel_identifier: Tuple of (timestamp, duration, stream, external_id) that
                          uniquely identifies a GPU operation. Stream can be None for
                          operations that don't have a stream (like cpu_op).
        gpu_trace: The loaded GPU trace dictionary
        
    Returns:
        Dictionary containing extracted metadata. The structure varies by category:
        
        For all operations:
        - ph: phase type
        - cat: category (cpu_op, kernel, cuda_runtime, etc.)
        - name: operation name
        - pid: process id
        - tid: thread id
        - ts: timestamp
        - dur: duration
        - external_id: External id linking to CPU operation
        
        For cpu_op:
        - record_function_id: Record function id
        - concrete_inputs: Input values
        - input_type: Input types
        - input_strides: Input strides
        - input_dims: Input dimensions
        - sequence_number: Sequence number (if present)
        - fwd_thread_id: Forward thread id (if present)
        
        For kernel:
        - correlation: Correlation id
        - device: Device id
        - context: Context id
        - stream: Stream id
        - queued: Queued time
        - registers_per_thread: Register usage
        - shared_memory: Shared memory usage
        - blocks_per_sm: Blocks per SM
        - warps_per_sm: Warps per SM
        - grid: Grid dimensions [x, y, z]
        - block: Block dimensions [x, y, z]
        - est_achieved_occupancy: Estimated occupancy percentage
        
        For cuda_runtime:
        - cbid: Callback id
        - correlation: Correlation id
    """
    ts, dur, stream, external_id = kernel_identifier
    
    # Find the GPU operation
    gpu_op = None
    for event in gpu_trace.get('traceEvents', []):
        event_ts = event.get('ts')
        event_dur = event.get('dur')
        event_stream = event.get('args', {}).get('stream')
        event_external_id = event.get('args', {}).get('External id')
        
        # Match based on timestamp, duration, and external_id
        # Stream matching is optional since not all operations have streams
        if event_ts == ts and event_dur == dur and event_external_id == external_id:
            if stream is None or event_stream is None or event_stream == stream:
                gpu_op = event
                break
    
    if gpu_op is None:
        return None
    
    # Extract common fields
    metadata = {
        'ph': gpu_op.get('ph'),
        'cat': gpu_op.get('cat'),
        'name': gpu_op.get('name'),
        'pid': gpu_op.get('pid'),
        'tid': gpu_op.get('tid'),
        'ts': gpu_op.get('ts'),
        'dur': gpu_op.get('dur'),
        'external_id': external_id,
    }
    
    # Extract category-specific fields from args
    args = gpu_op.get('args', {})
    
    # Common fields across categories
    if 'Record function id' in args:
        metadata['record_function_id'] = args['Record function id']
    if 'correlation' in args:
        metadata['correlation'] = args['correlation']
    if 'cbid' in args:
        metadata['cbid'] = args['cbid']
    
    # cpu_op specific fields
    if metadata['cat'] == 'cpu_op':
        metadata['concrete_inputs'] = args.get('Concrete Inputs')
        metadata['input_type'] = args.get('Input type')
        metadata['input_strides'] = args.get('Input Strides')
        metadata['input_dims'] = args.get('Input Dims')
        metadata['sequence_number'] = args.get('Sequence number')
        metadata['fwd_thread_id'] = args.get('Fwd thread id')
        metadata['ev_idx'] = args.get('Ev Idx')
    
    # kernel specific fields
    elif metadata['cat'] == 'kernel':
        metadata['device'] = args.get('device')
        metadata['context'] = args.get('context')
        metadata['stream'] = args.get('stream')
        metadata['queued'] = args.get('queued')
        metadata['registers_per_thread'] = args.get('registers per thread')
        metadata['shared_memory'] = args.get('shared memory')
        metadata['blocks_per_sm'] = args.get('blocks per SM')
        metadata['warps_per_sm'] = args.get('warps per SM')
        metadata['grid'] = args.get('grid')
        metadata['block'] = args.get('block')
        metadata['est_achieved_occupancy'] = args.get('est. achieved occupancy %')
    
    # cuda_runtime, memory, and other categories can be extended as needed
    
    return metadata


def load_trace(trace_path: str) -> Dict[str, Any]:
    """
    Helper function to load a trace file.
    
    Args:
        trace_path: Path to the trace JSON file
        
    Returns:
        Loaded trace dictionary
    """
    with open(trace_path, 'r') as f:
        return json.load(f)


def get_gpu_op_identifier(gpu_event: Dict[str, Any]) -> Tuple[int, int, Optional[int], int]:
    """
    Helper function to create a unique identifier for a GPU operation.
    
    Args:
        gpu_event: A GPU trace event dictionary
        
    Returns:
        Tuple of (timestamp, duration, stream, external_id)
    """
    ts = gpu_event.get('ts')
    dur = gpu_event.get('dur')
    stream = gpu_event.get('args', {}).get('stream')
    external_id = gpu_event.get('args', {}).get('External id')
    
    return (ts, dur, stream, external_id)
