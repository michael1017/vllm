"""vLLM distributed KV cache transfer API.
These APIs are used in `vllm/worker/model_runner.py`.

Currently supporting TP and PP.

Workflow:
- In prefill instance, KV cache sender *buffers* the KV cache send requests
- In decode instance
    - KV cache receiver sends the hash of input tokens to sender
    - KV cache sender executes send request
    - KV cache receiver receives the KV cache
"""
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from copy import deepcopy
import time
import threading

import torch
from torch.distributed import Backend, ProcessGroup

import vllm.envs as envs
from vllm.distributed.group_coordinator import GroupCoordinator
from vllm.logger import init_logger
import vllm.distributed.parallel_state as ps
from vllm import _custom_ops as ops
from vllm.sequence import IntermediateTensors

assert envs.VLLM_DISAGG_PREFILL_ROLE in [None, "prefill", "decode", "lmc"], \
    "VLLM_DISAGG_PREFILL_ROLE can only be prefill or decode."

IS_DISTRIBUTED_KV_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE in ["prefill", "decode"])
IS_KV_PREFILL_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "prefill")
IS_KV_DECODE_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "decode")

'''Jiayi starts here'''
IS_LMC_INSTANCE: bool = (envs.VLLM_DISAGG_PREFILL_ROLE == "lmc")
'''Jiayi ends here'''

# add a tag when sending/recving input hash
DISTRIBUTED_KV_GLOO_TAG = 24857323

logger = init_logger(__name__)

import logging


class RankFilter(logging.Filter):

    def filter(self, record):
        # Only log if rank is 4
        rank = 1
        try:
            rank = torch.distributed.get_rank()
        except Exception:
            pass
        return rank % 4 == 0


for handler in logger.handlers:
    handler.addFilter(RankFilter())


class DistributedKVCoordinator(GroupCoordinator):
    """
    A class designated for distributed KV transfer
    
    Target use cases:
        1. Disaggregated prefill
        2. Remote KV cache storage
        
    """

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        # DO NOT use pynccl here
        # Pynccl send is non-blocking
        # and it's possible that the memory is freed before the data being sent
        # which may happen at high qps
        use_pynccl: bool = False,
        use_custom_allreduce: bool = False,
        use_tpu_communicator: bool = True,
        use_message_queue_broadcaster: bool = False,
        use_cpu_comm: bool = False,
    ):

        super().__init__(
            group_ranks,
            local_rank,
            torch_distributed_backend,
            use_pynccl,
            use_custom_allreduce,
            use_tpu_communicator,
            use_message_queue_broadcaster,
        )

        # if turned on, will use CPU-based communication to perform a series of sanity check.
        # but it adds ~5ms delay, so please turn it off in performance-demanding usecases (e.g. disaggregated prefill)
        self.use_cpu_comm = use_cpu_comm

        # use a threadpool to buffer send request in disaggregated prefill
        self.input_hash_to_kv_sending_requests = defaultdict(deque)
        self.kv_sending_thread = None
        self.input_hash_to_kv_sending_requests_lock = Lock()
        self.target_rank_for_send = self.ranks[(self.rank_in_group + 1) %
                                               self.world_size]
        self.target_rank_for_recv = self.ranks[(self.rank_in_group - 1) %
                                               self.world_size]

        torch.set_default_device(self.device)

    def debug_send(self,
                   tensor: torch.Tensor,
                   dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """Will send several metadata. Useful for debugging."""
        """NOTE: `dst` is the local rank of the destination rank."""

        self.send_tensor_dict(
            {
                "tensor": tensor,
                "mean": tensor.float().mean(),
                "shape": tensor.shape
            }, dst)

    def debug_recv(self,
                   size: torch.Size,
                   dtype: torch.dtype,
                   src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the src rank."""
        """NOTE: `src` is the local rank of the destination rank."""

        result = self.recv_tensor_dict(src)
        tensor = result["tensor"]
        assert torch.allclose(result["mean"], tensor.float().mean())
        assert result["shape"] == tensor.shape
        assert result[
            "shape"] == size, f"The shape sent by sender is {result['shape']} but trying to receive {size}"
        return tensor

    def kv_cache_recv(
            self,
            size: torch.Size,
            dtype: torch.dtype,
            is_hidden: bool = False,
            src: Optional[int] = None
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Receives a tensor from the src rank (blocking)."""
        """This API should be used together with `push`"""
        """NOTE: `src` is the local rank of the destination rank."""

        if self.use_cpu_comm:
            recv_func = self.debug_recv
        else:
            recv_func = self.recv

        if is_hidden and not ps.get_pp_group().is_last_rank:
            tensor = IntermediateTensors(self.recv_tensor_dict(src))
        else:
            tensor = recv_func(size, dtype, src)

        return tensor

    def send_input_hash(self, input_hash: int) -> int:

        logger.debug('[rank%d]: Sending input hash %d to rank %d',
                     torch.distributed.get_rank(), input_hash,
                     self.target_rank_for_send)

        # KV cache send go through CPU, and the original `send` only use GPU.
        # So create a new group for sending input hash.
        input_hash_tensor = torch.tensor([input_hash], device="cpu").long()
        torch.distributed.send(input_hash_tensor,
                               self.target_rank_for_send,
                               self.cpu_group,
                               tag=DISTRIBUTED_KV_GLOO_TAG)
        return_tensor = torch.tensor([0], device="cpu").long()
        torch.distributed.recv(return_tensor,
                               self.target_rank_for_recv,
                               self.cpu_group,
                               tag=DISTRIBUTED_KV_GLOO_TAG)
        return return_tensor.item()

    def recv_input_hash(self) -> Optional[int]:
        '''
            Receive an input hash, and check if it is already cached
        '''
        input_hash_tensor = torch.tensor([0], device="cpu").long()
        torch.distributed.recv(input_hash_tensor,
                               self.target_rank_for_recv,
                               self.cpu_group,
                               tag=DISTRIBUTED_KV_GLOO_TAG)
        input_hash = input_hash_tensor.item()
        # a new input hash comes in, see if it is already cached
        self.input_hash_to_kv_sending_requests_lock.acquire()
        logger.debug('Successfully received input hash %d', input_hash)
        if input_hash not in self.input_hash_to_kv_sending_requests:
            logger.warning(
            f"The KV cache of {input_hash} does not exist. "\
            f"Existing input hash: {list(self.input_hash_to_kv_sending_requests.keys())}")
            
            # 0 for fail
            x = torch.tensor([0], device="cpu").long()
            torch.distributed.send(x,
                                    self.target_rank_for_send,
                                    self.cpu_group,
                                    tag=DISTRIBUTED_KV_GLOO_TAG)
            return None
        else:
            logger.debug('Input hash %d exists, start sending', input_hash)
            
            # 1 for success
            x = torch.tensor([1], device="cpu").long()
            torch.distributed.send(x,
                                   self.target_rank_for_send,
                                   self.cpu_group,
                                   tag=DISTRIBUTED_KV_GLOO_TAG)
            return input_hash
        
    # FIXME(Jiayi): add send num_kv and token tensor
    def kv_cache_send(self,
                      input_hash: int, 
                      tensor: Union[torch.Tensor, IntermediateTensors],
                      token_ids: List[int] = [],
                      is_hidden: bool = False,
                      dst: Optional[int] = None) -> None:
        """Push the KV cache send request into the send buffer"""
        """NOTE: `dst` is the local rank of the destination rank."""

        if self.use_cpu_comm:
            send_func = self.debug_send
        else:
            send_func = self.send

        if is_hidden and not ps.get_pp_group().is_last_rank:

            assert isinstance(tensor, IntermediateTensors)

            output = deepcopy(tensor.tensors)
            for key in output:
                output[key] = output[key].contiguous()

            self.input_hash_to_kv_sending_requests[input_hash].append(
                [self.send_tensor_dict, output, dst])

        else:

            assert isinstance(tensor, torch.Tensor)

            self.input_hash_to_kv_sending_requests[input_hash].append([
                send_func,
                # use clone to make sure the tensor is contiguous
                tensor.clone(),
                token_ids,
                dst
            ])

    '''
    def kv_cache_send_loop(self):
        
        token_ids_list = self.recv_object()
        for token_ids in token_ids_list:
            logger.debug(
                '[rank%d]: Waiting for input hash from rank %d, my keys are %s',
                torch.distributed.get_rank(),
                self.target_rank_for_recv,
                list(self.input_hash_to_kv_sending_requests.keys()),
            )
            
            input_hash = hash(str(token_ids))
            #if input_hash is None:
            #    self.input_hash_to_kv_sending_requests_lock.release()
            #    continue

            # execute corresponding kv cache sending jobs in request queue
            while True:
                request = self.input_hash_to_kv_sending_requests[
                    input_hash].popleft()
                # An empty request: the KV cahe of one request are all sent
                if request == []:
                    break

                # call the send function (self.send)
                request[0](*request[1:])

            if len(self.input_hash_to_kv_sending_requests[input_hash]) == 0:
                logger.debug('Finish input hash %d, free GPU memory...',
                             input_hash)
                del self.input_hash_to_kv_sending_requests[input_hash]
            else:
                logger.debug(
                    'The buffer for input hash %d is not empty, meaning that '\
                    'there are two jobs with identical input.',
                    input_hash)

            self.input_hash_to_kv_sending_requests_lock.release()
    '''



    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors],
    ) -> None:

        #input_tokens_tuple = tuple(model_input.input_tokens.tolist())
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()

        # Assumption: current batch is all-prefill requests
        #assert torch.allclose(model_input.attn_metadata.query_start_loc,
        #                    model_input.attn_metadata.seq_start_loc)
        #assert torch.all(model_input.attn_metadata.context_lens_tensor == 0)
        
        #token_ids_list = 
        # Receive `token_ids_list` to kick off kv cache transfer
        logger.debug(f"Sender cpu group {self.ranks}")
        logger.debug(f"Receiving input token_ids_list from rank {self.target_rank_for_recv}")
        size_tensor = torch.tensor([0,0])
        torch.distributed.recv(
            size_tensor,
            self.target_rank_for_recv,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        logger.debug(f"Receiving size tensor done")
        token_ids_list_tensor = torch.empty(size_tensor.tolist(), dtype=torch.long)
        torch.distributed.recv(
            token_ids_list_tensor,
            self.target_rank_for_recv,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        logger.debug(f"Receiving token_ids_list tensor done")
        #token_ids_list = self.recv_object(src=self.target_rank_for_recv)
        #ps.get_disagg_group().input_hash_to_kv_sending_requests_lock.acquire()

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            logger.debug(f"sending request {idx}")
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            
            # TODO(Jiayi): need to be smarter (if xxx dump all)
            token_ids = input_tokens_tensor[start_pos:end_pos].tolist()
            token_tensor = torch.tensor(token_ids)
            num_toks = len(token_ids)
            #input_hash = hash(str(token_ids))
            
            # Send num_computed_toks
            num_toks_tensor = torch.tensor([num_toks])
            torch.distributed.send(
                num_toks_tensor,
                self.target_rank_for_send,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            
            # Send num_toks
            torch.distributed.send(
                num_toks_tensor,
                self.target_rank_for_send,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            
            # Send token tensor
            torch.distributed.send(
                token_tensor,
                self.target_rank_for_send,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            
            for l in range(model_executable.model.start_layer,
                        model_executable.model.end_layer):
                logger.debug(f"sending layer {l}")
                kv_cache = kv_caches[l - model_executable.model.start_layer]

                _, _, num_heads, head_size = kv_cache[0].shape

                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]
                

                # TODO(Jiayi): maybe optimize the following with one send only?
                
                # FIXME (Jiayi): need to support dynamically sending over cpu/gpu
                # Send Key cache
                torch.distributed.send(
                    key_cache[current_slot_mapping].clone().cpu(),
                    self.target_rank_for_send,
                    self.cpu_group,
                    tag=DISTRIBUTED_KV_GLOO_TAG)
                
                # Send Value cache
                torch.distributed.send(
                    value_cache[current_slot_mapping].clone().cpu(),
                    self.target_rank_for_send,
                    self.cpu_group,
                    tag=DISTRIBUTED_KV_GLOO_TAG)

            logger.debug(f"sending hidden states")
            #FIXME(Jiayi): fix pp
            torch.distributed.send(
                hidden_or_intermediate_states.clone().cpu(),
                self.target_rank_for_send,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            
        # Send end signal
        end_signal_tensor = torch.tensor([-1])
        torch.distributed.send(
            end_signal_tensor,
            self.target_rank_for_send,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
            

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())


    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool]:


        # This is disagg decode instance, during prefill state
        # Need to receive KV from the prefill instance
        token_ids_flat = model_input.input_tokens.tolist()
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        
        bypass_model_exec_list = ['no_skip']*len(seq_lens)
        
        token_ids_list = []
        slot_mapping_list = []
        token_ids_idx_mapping = {}
        start_pos = 0
        
        rebuilt_input_tokens = []
        rebuilt_input_positions= []
        rebuilt_query_lens = []
        
        rebuilt_num_prefills = 0
        rebuilt_num_prefill_tokens = 0
        rebuilt_slot_mapping = []
        rebuilt_max_query_len = 0
        
        rebuilt_block_tables = []
        
        rebuilt_query_start_loc = [0]
        rebuilt_context_lens_tensor = []
        rebuilt_selected_token_indices = []
        
        for idx, slen in enumerate(seq_lens):
            end_pos = start_pos + slen
            
            token_ids = token_ids_flat[start_pos:end_pos]
            token_ids_list.append(token_ids)
            slot_mapping_list.append(slot_mapping_flat[start_pos:end_pos])
            token_ids_idx_mapping = {hash(str(token_ids)): idx}
            
            start_pos = end_pos
            
        # Assumption: current batch is all-prefill requests
        #assert torch.allclose(model_input.attn_metadata.query_start_loc,
        #                    model_input.attn_metadata.seq_start_loc)
        #assert torch.all(model_input.attn_metadata.context_lens_tensor == 0)

        # Send `token_ids_list` to kick off kv cache transfer
        logger.debug(f"Receiver cpu group {self.ranks}")
        logger.debug(f"Sending input token_ids_list_tensor to rank {self.target_rank_for_send}")
        token_ids_list_tensor = torch.tensor(token_ids_list)
        size_tensor = torch.tensor(token_ids_list_tensor.size())
        torch.distributed.send(
            size_tensor,
            self.target_rank_for_send,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        logger.debug(f"Sending size tensor done")
        
        torch.distributed.send(
            token_ids_list_tensor,
            self.target_rank_for_send,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        logger.debug(f"Sending token_ids_list tensor done")
        
        hidden_or_intermediate_states_for_one_req = []
        
        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        # TODO(Jiayi): Note that kv caches could arrive out of order (in the future)
        # For example, req1 is long while req2 is short, req2 should start decoding earlier
        for idx in range(len(seq_lens)):
            
            logger.debug(f"receiving request {idx}")
            
            # Receive num computed token tensor
            num_computed_token_tensor = torch.tensor([0])
            torch.distributed.recv(
                num_computed_token_tensor,
                self.target_rank_for_recv,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            num_computed_token = num_computed_token_tensor.item()
            
            
            # Receive num token tensor
            num_token_tensor = torch.tensor([0])
            torch.distributed.recv(
                num_token_tensor,
                self.target_rank_for_recv,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            num_token = num_token_tensor.item()
            
            # Receive token tensor
            token_tensor = torch.empty((num_token), dtype=torch.long)
            torch.distributed.recv(
                token_tensor,
                self.target_rank_for_recv,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            
            token_ids = token_tensor.tolist()
            token_hash = hash(str(token_ids))
            
            if num_computed_token == 0:
                bypass_model_exec_list[token_ids_idx_mapping[token_hash]] = 'no_skip'
                continue

            elif num_computed_token < slen:
                bypass_model_exec_list[token_ids_idx_mapping[token_hash]] = 'prefix'
            else:
                bypass_model_exec_list[token_ids_idx_mapping[token_hash]] = 'skip'
                
            start_pos = sum(seq_lens[:token_ids_idx_mapping[token_hash]])

            # receive KV cache from disaggregated prefill instance
            for l in range(model_executable.model.start_layer,
                        model_executable.model.end_layer):

                logger.debug(f"receiving layer {l}")
                # get kv cache
                kv_cache = kv_caches[l - model_executable.model.start_layer]
                # get corresponding layer
                layer = model_executable.model.layers[l]

                
                # get kv cache shape (after sliced by tp)
                _, _, num_heads, head_size = kv_cache[0].shape
                

                # receive key tensor
                key_tensor = torch.empty((num_computed_token, num_heads, head_size),
                                        dtype=kv_cache[0].dtype)
                torch.distributed.recv(
                    key_tensor,
                    self.target_rank_for_recv,
                    self.cpu_group,
                    tag=DISTRIBUTED_KV_GLOO_TAG)
                    
                # receive value tensor
                value_tensor = torch.empty((num_computed_token, num_heads, head_size),
                                        dtype=kv_cache[0].dtype)
                torch.distributed.recv(
                    value_tensor,
                    self.target_rank_for_recv,
                    self.cpu_group,
                    tag=DISTRIBUTED_KV_GLOO_TAG)

                key_cache, value_cache = kv_cache[0], kv_cache[1]
                ops.reshape_and_cache_flash(
                    key_tensor.to(kv_cache[0].device),
                    value_tensor.to(kv_cache[0].device),
                    key_cache,
                    value_cache,
                    slot_mapping_flat[start_pos:start_pos+num_computed_token],
                    layer.self_attn.attn.kv_cache_dtype,
                    layer.self_attn.attn._k_scale,
                    layer.self_attn.attn._v_scale,
                )
                
                

            logger.debug(f"receiving hidden states")
            hidden_states = torch.empty([num_token, model_executable.config.hidden_size],
                                        dtype=kv_cache[0].dtype)
            torch.distributed.recv(
                hidden_states,
                self.target_rank_for_recv,
                self.cpu_group,
                tag=DISTRIBUTED_KV_GLOO_TAG)
            
            #TODO(Jiayi): why transfer all hidden_states?
            hidden_or_intermediate_states_for_one_req.append(hidden_states.to(kv_cache[0].device))
            
            # Modify model input data to adapt to prefix caching
            # FIXME (Jiayi): use index instead of append
            rebuilt_input_tokens.append(token_tensor[num_computed_token:])
            rebuilt_input_positions.append(model_input.input_positions[num_computed_token:num_token])
            q_len = num_token - num_computed_token
            rebuilt_query_lens.append(q_len)
            
            # Attn metadata-related
            rebuilt_num_prefills += 1
            rebuilt_num_prefill_tokens += q_len
            rebuilt_slot_mapping.append(slot_mapping_flat[start_pos+num_computed_token:start_pos+num_token])
            rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)
            # TODO(Jiayi): remove hard-code (block_size=16)
            blk_size = 16
            temp_block_table = [i//blk_size for i in range(start_pos, start_pos+num_token, blk_size)]
            rebuilt_block_tables.append(temp_block_table)
            rebuilt_query_start_loc.append(q_len) #start with 0
            rebuilt_context_lens_tensor.append(num_computed_token)
            
            # Sampleing metadata related
            #seq_groups (use rebuilt query lens)
            rebuilt_selected_token_indices.append(start_pos+q_len-1)
        
        # Receiving a null end signal
        logger.debug(f"receiving end signal")
        end_signal_tensor = torch.tensor([0])
        torch.distributed.recv(
            end_signal_tensor ,
            self.target_rank_for_recv,
            self.cpu_group,
            tag=DISTRIBUTED_KV_GLOO_TAG)
        
        if 'no_skip' in bypass_model_exec_list:
            logger.debug(f"no prefix hit in current batch")
            return [], False, model_input, False
        
        if 'prefix' in bypass_model_exec_list:
            device = kv_cache[0].device
            logger.debug(f"prefix hit in current batch")
            
            from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
            from copy import deepcopy
            # rebuilt attn_metadata
            rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
            rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
            rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
            rebuilt_attn_metadata.slot_mapping = torch.cat(rebuilt_slot_mapping).to(device)
            rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len
            
            rebuilt_attn_metadata.block_tables = torch.tensor(
                rebuilt_block_tables,
                dtype=model_input.attn_metadata.block_tables.dtype
                ).to(device)
            
            rebuilt_attn_metadata.query_start_loc = torch.tensor(
                rebuilt_query_start_loc,
                dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
            rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
                rebuilt_context_lens_tensor, 
                dtype=model_input.attn_metadata.context_lens_tensor.dtype,
                ).to(device)
            
            rebuilt_attn_metadata._cached_prefill_metadata = None
            '''
            rebuilt_attn_metadata._cached_prefill_metadata = FlashAttentionMetadata(
                num_prefills=rebuilt_attn_metadata.num_prefills,
                num_prefill_tokens=rebuilt_attn_metadata.num_prefill_tokens,
                num_decode_tokens=0,
                slot_mapping=rebuilt_attn_metadata.slot_mapping[:rebuilt_attn_metadata.num_prefill_tokens],
                seq_lens=rebuilt_attn_metadata.seq_lens[:rebuilt_attn_metadata.num_prefills],
                seq_lens_tensor=rebuilt_attn_metadata.seq_lens_tensor[:rebuilt_attn_metadata.num_prefills],
                max_query_len=rebuilt_attn_metadata.max_query_len,
                max_prefill_seq_len=rebuilt_attn_metadata.max_prefill_seq_len,
                max_decode_seq_len=0,
                query_start_loc=rebuilt_attn_metadata.query_start_loc[:rebuilt_attn_metadata.num_prefills + 1],
                seq_start_loc=rebuilt_attn_metadata.seq_start_loc[:rebuilt_attn_metadata.num_prefills + 1],
                context_lens_tensor=rebuilt_attn_metadata.context_lens_tensor[:rebuilt_attn_metadata.num_prefills],
                block_tables=rebuilt_attn_metadata.block_tables[:rebuilt_attn_metadata.num_prefills],
                use_cuda_graph=False,
            )
            '''
            # rebuilt sampling_metadata
            rebuilt_sampling_metadata = deepcopy(model_input.sampling_metadata)
            for idx, q_len in enumerate(rebuilt_query_lens):
                rebuilt_sampling_metadata.seq_groups[idx].query_len = q_len
            rebuilt_sampling_metadata.selected_token_indices = torch.tensor(
                rebuilt_selected_token_indices,
                dtype=model_input.sampling_metadata.selected_token_indices.dtype,
                ).to(device)
            
            rebuilt_model_input = ModelInputForGPUWithSamplingMetadata(
                input_tokens = torch.cat(rebuilt_input_tokens).to(device),
                input_positions = torch.cat(rebuilt_input_positions).to(device),
                seq_lens = model_input.seq_lens,
                query_lens = rebuilt_query_lens,
                lora_mapping = model_input.lora_mapping,
                lora_requests = model_input.lora_requests,
                attn_metadata = rebuilt_attn_metadata,
                prompt_adapter_mapping = model_input.prompt_adapter_mapping,
                prompt_adapter_requests = model_input.prompt_adapter_requests,
                multi_modal_kwargs = model_input.multi_modal_kwargs,
                request_ids_to_seq_ids = model_input.request_ids_to_seq_ids,
                finished_requests_ids = model_input.finished_requests_ids,
                virtual_engine = model_input.virtual_engine,
                sampling_metadata = rebuilt_sampling_metadata,
                is_prompt = model_input.is_prompt,
            )
            return [], False, rebuilt_model_input, True
        

        logger.debug(f"all skip in current batch")
        
        hidden_or_intermediate_states = torch.cat(
            hidden_or_intermediate_states_for_one_req, dim=0)

        logger.debug("[rank%d]: KV recv DONE.", torch.distributed.get_rank())
        return hidden_or_intermediate_states, True, model_input, False
