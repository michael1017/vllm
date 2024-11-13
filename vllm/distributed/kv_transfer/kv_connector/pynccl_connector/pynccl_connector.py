"""
    This file implements a simple torch distributed connector by 3 classes:
    - `TorchDistributedPipe`: a tensor transmission pipe between vllm instances,
        using `torch.distributed`
    - `TorchDistributedBuffer`: a buffer to store tensors, implemented on top 
        of `TorchDistributedPipe`
    - `TorchDistributedConnector`: a torch distributed connector between P/D 
      instance, implemented on top of `TorchDistributedBuffer`
"""
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, List, Optional, Union
from copy import deepcopy

import torch
from torch.distributed import Backend

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.pynccl_connector.pynccl_pipe \
    import PyncclPipe
from vllm.distributed.kv_transfer.kv_connector.pynccl_connector.lookup_buffer \
    import LookupBuffer
from vllm.logger import init_logger
from vllm.config import KVTransferConfig



logger = init_logger(__name__)



            
class TorchDistributedConnector(KVConnectorBase):
    
    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        config: KVTransferConfig,
    ):

        self.lookup_buffer_size = self.kv_buffer_size

        self.send_buffer: Optional[TorchDistributedBuffer] = None
        self.recv_buffer: Optional[TorchDistributedBuffer] = None
        
        device2backend = {
            "cpu": "gloo",
            "gpu": "nccl",
        }

        # In disaggregated prefill, the prefill vLLM only uses send pipe
        # and the decode vLLM only uses recv pipe
        # In remote KV cache store, vLLM will use both send pipe and recv pipe
        # So we build both send pipe and recv pipe for simplicity.
        if config.is_kv_producer:

            self.send_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                device2backend[config.kv_device],
                self.kv_buffer_size, 
            )
            self.send_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
                self.kv_buffer_size,
            )
            self.recv_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                device2backend[config.kv_device],
                self.kv_buffer_size,
            )
            self.recv_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
                self.kv_buffer_size
            )
             
        else:

            # the current vLLM instance is KV consumer, so it needs to connect
            # its recv pipe to the send pipe of KV producder

            self.recv_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                device2backend[config.kv_device],
                self.kv_buffer_size, 
            )
            self.recv_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
                self.kv_buffer_size,
            )
            self.send_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                device2backend[config.kv_device],
                self.kv_buffer_size,
            )
            self.send_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
                self.kv_buffer_size
            )

        self.send_buffer = TorchDistributedBuffer(self.send_signal_pipe,
                                                  self.send_pipe,
                                                  self.lookup_buffer_size)
        self.recv_buffer = TorchDistributedBuffer(self.recv_signal_pipe,
                                                  self.recv_pipe,
                                                  self.lookup_buffer_size)
        self.tensor_device = config.kv_device
            
            
    def select(
        self, input_tokens: Optional[torch.Tensor],
        roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

        return self.send_buffer.drop_select(input, roi)
    
    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        return self.recv_buffer.insert(
            input_tokens,
            roi,
            key,
            value,
            hidden
        )

            
            
    def build_partial_prefill_input(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        input_tokens_list: List[torch.Tensor],
        num_computed_tokens_list: List[int],
        start_pos_list: List[int],
        slot_mapping_flat: torch.Tensor,
        device: torch.device,
    ) -> "ModelInputForGPUWithSamplingMetadata":
        """
        Helper function to rebuild the model input for the current request.
        Goal: avoid running redundant prefill on those tokens that already has
        KV caches received.
        """
        rebuilt_input_tokens = []
        rebuilt_input_positions = []
        rebuilt_query_lens = []

        rebuilt_num_prefills = 0
        rebuilt_num_prefill_tokens = 0
        rebuilt_slot_mapping = []
        rebuilt_max_query_len = 0

        rebuilt_block_tables = []

        rebuilt_query_start_loc = [0]
        rebuilt_context_lens_tensor = []
        rebuilt_selected_token_indices = []

        # recounting query and context lengths
        for idx in range(len(input_tokens_list)):
            token_tensor = input_tokens_list[idx]
            num_token = len(token_tensor)
            num_computed_token = num_computed_tokens_list[idx]
            # currently attention kernel cannot handle the case where there is 0
            # query token.
            if num_computed_token == num_token:
                num_computed_token -= 1
            start_pos = start_pos_list[idx]

            rebuilt_input_tokens.append(token_tensor[num_computed_token:])
            # TODO(Jiayi): please check the correctness of next line
            rebuilt_input_positions.append(
                model_input.input_positions[start_pos +
                                            num_computed_token:start_pos +
                                            num_token])
            q_len = num_token - num_computed_token
            rebuilt_query_lens.append(q_len)

            # Attn metadata-related
            rebuilt_num_prefills += 1
            rebuilt_num_prefill_tokens += q_len
            new_slot_mapping = slot_mapping_flat[start_pos +
                                                num_computed_token:start_pos +
                                                num_token]
            rebuilt_slot_mapping.append(new_slot_mapping)
            rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)
            # TODO(Jiayi): remove hard-code (block_size=16)
            blk_size = 16
            temp_block_table = [
                slot_mapping_flat[i] // blk_size
                for i in range(start_pos, start_pos + num_token, blk_size)
            ]
            rebuilt_block_tables.append(temp_block_table)
            rebuilt_query_start_loc.append(
                rebuilt_num_prefill_tokens)  #start with 0
            rebuilt_context_lens_tensor.append(num_computed_token)

            # Sampling metadata related
            #seq_groups (use rebuilt query lens)
            rebuilt_selected_token_indices.append(rebuilt_num_prefill_tokens - 1)

        # rebuilt attn_metadata
        rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
        rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
        rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
        rebuilt_attn_metadata.slot_mapping = torch.cat(rebuilt_slot_mapping).to(
            device)
        rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len

        rebuilt_attn_metadata.block_tables = torch.tensor(
            rebuilt_block_tables,
            dtype=model_input.attn_metadata.block_tables.dtype).to(device)

        rebuilt_attn_metadata.query_start_loc = torch.tensor(
            rebuilt_query_start_loc,
            dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
        rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
            rebuilt_context_lens_tensor,
            dtype=model_input.attn_metadata.context_lens_tensor.dtype,
        ).to(device)

        rebuilt_attn_metadata._cached_prefill_metadata = None

        # rebuilt sampling_metadata
        rebuilt_sampling_metadata = deepcopy(model_input.sampling_metadata)
        for idx, q_len in enumerate(rebuilt_query_lens):
            if rebuilt_sampling_metadata.seq_groups is not None:
                rebuilt_sampling_metadata.seq_groups[idx].query_len = q_len

        rebuilt_sampling_metadata.selected_token_indices = torch.tensor(
            rebuilt_selected_token_indices,
            dtype=model_input.sampling_metadata.selected_token_indices.dtype,
        ).to(device)

        # import here to avoid circular import.
        from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
        rebuilt_model_input = ModelInputForGPUWithSamplingMetadata(
            input_tokens=torch.cat(rebuilt_input_tokens).to(device),
            input_positions=torch.cat(rebuilt_input_positions).to(device),
            seq_lens=model_input.seq_lens,
            query_lens=rebuilt_query_lens,
            lora_mapping=model_input.lora_mapping,
            lora_requests=model_input.lora_requests,
            attn_metadata=rebuilt_attn_metadata,
            prompt_adapter_mapping=model_input.prompt_adapter_mapping,
            prompt_adapter_requests=model_input.prompt_adapter_requests,
            multi_modal_kwargs=model_input.multi_modal_kwargs,
            request_ids_to_seq_ids=model_input.request_ids_to_seq_ids,
            finished_requests_ids=model_input.finished_requests_ids,
            virtual_engine=model_input.virtual_engine,
            sampling_metadata=rebuilt_sampling_metadata,
            is_prompt=model_input.is_prompt,
        )

        return rebuilt_model_input

