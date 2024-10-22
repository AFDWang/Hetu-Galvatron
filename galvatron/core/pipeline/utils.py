import torch
from typing import Optional, List, Union

def listify_model(model: Union[torch.nn.Module, List[torch.nn.Module]]) -> List[torch.nn.Module]:
    if isinstance(model, list):
        return model
    return [model]

def chunk_batch(inputs, chunks):
    if inputs is None:
        return inputs

    batches = [[] for _ in range(chunks)]
    # Actual number of chunks produced
    num_chunks = -1
    for input in inputs:
        if torch.is_tensor(input):
            # Chunk only tensors.
            tensors = input.chunk(chunks)

            # Validate number of chunks equal across all inputs.
            if num_chunks != -1 and num_chunks != len(tensors):
                raise RuntimeError(f'Found different number of chunks produced for inputs: {num_chunks} and {len(tensors)}')
            num_chunks = len(tensors)

            for i, tensor in enumerate(tensors):
                batches[i].append(tensor)
        else:
            # Replicate non-tensors or tensors wrapped with 'NoChunk'.
            for i in range(chunks):
                batches[i].append(input)
            num_chunks = chunks

    # Truncate to actual number of chunks
    batches = batches[:num_chunks]

    return batches

def chunk_dict(kwargs, chunks):
    batches = [{} for _ in range(chunks)]
    num_chunks = -1
    for k,v in kwargs.items():
        if torch.is_tensor(v) and not (k.endswith("_mask") and v.shape[0] == 1):
            tensors = v.chunk(chunks)
            if num_chunks != -1 and num_chunks != len(tensors):
                raise RuntimeError(f'Found different number of chunks produced for inputs: {num_chunks} and {len(tensors)}')
            num_chunks = len(tensors)
            for i, tensor in enumerate(tensors):
                batches[i][k] = tensor
        else:
            for i in range(chunks):
                batches[i][k] = v
                
    batches = batches[:num_chunks]
    return batches