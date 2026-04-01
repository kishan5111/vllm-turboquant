# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TurboQuant capture module — bulk ingestion and ring-buffer management.

Ports the 0xSero/turboquant capture implementation.
"""

import torch
from typing import Optional

from .store import CompressedKVStore


class RingBuffer:
    """Fixed-size ring buffer for recent exact KV tokens."""

    __slots__ = (
        "capacity", "num_kv_heads", "head_dim", "device", "dtype",
        "_k", "_v", "_pos", "_total_written",
    )

    def __init__(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.capacity = capacity
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self._k = torch.zeros(
            capacity, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        self._v = torch.zeros(
            capacity, num_kv_heads, head_dim, device=device, dtype=dtype
        )
        self._pos = 0
        self._total_written = 0

    @property
    def size(self) -> int:
        return self._pos

    @property
    def is_full(self) -> bool:
        return self._pos >= self.capacity

    @property
    def total_written(self) -> int:
        return self._total_written

    def write(
        self, key: torch.Tensor, value: torch.Tensor, num_tokens: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Append tokens. Returns (overflow_k, overflow_v) if buffer overflows."""
        overflow_k_parts = []
        overflow_v_parts = []

        offset = 0
        remaining = num_tokens

        while remaining > 0:
            space = self.capacity - self._pos
            if space <= 0:
                overflow_k_parts.append(self._k[: self._pos].clone())
                overflow_v_parts.append(self._v[: self._pos].clone())
                self._pos = 0
                space = self.capacity

            n = min(remaining, space)
            self._k[self._pos : self._pos + n] = key[offset : offset + n]
            self._v[self._pos : self._pos + n] = value[offset : offset + n]
            self._pos += n
            offset += n
            remaining -= n

        self._total_written += num_tokens

        if overflow_k_parts:
            return (
                torch.cat(overflow_k_parts, dim=0),
                torch.cat(overflow_v_parts, dim=0),
            )
        return None

    def drain(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return all buffered tokens and reset."""
        if self._pos == 0:
            return None
        k = self._k[: self._pos].clone()
        v = self._v[: self._pos].clone()
        self._pos = 0
        return k, v

    def peek(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Read current buffer contents without draining."""
        if self._pos == 0:
            return None
        return self._k[: self._pos], self._v[: self._pos]

    def reset(self):
        self._pos = 0
        self._total_written = 0


class KVCaptureEngine:
    """Orchestrates capture of KV pairs into a CompressedKVStore."""

    def __init__(
        self,
        store: CompressedKVStore,
        ring_capacity: int = 128,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.store = store
        self.ring = RingBuffer(
            capacity=ring_capacity,
            num_kv_heads=store.num_kv_heads,
            head_dim=store.head_dim,
            device=device or store.device,
            dtype=dtype,
        )
        self._prefill_done = False

    @property
    def total_compressed_tokens(self) -> int:
        return self.store.num_tokens

    @property
    def total_buffered_tokens(self) -> int:
        return self.ring.size

    @property
    def total_tokens(self) -> int:
        return self.total_compressed_tokens + self.total_buffered_tokens

    def ingest_prefill(
        self, key: torch.Tensor, value: torch.Tensor, num_tokens: int
    ):
        """Bulk-capture prefill KV into the store (bypasses ring buffer)."""
        if num_tokens <= self.ring.capacity:
            self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        else:
            n_compress = num_tokens - self.ring.capacity
            self.store.append_chunk(key[:n_compress], value[:n_compress])
            self.ring.write(
                key[n_compress:num_tokens],
                value[n_compress:num_tokens],
                self.ring.capacity,
            )
        self._prefill_done = True

    def ingest_decode(
        self, key: torch.Tensor, value: torch.Tensor, num_tokens: int
    ):
        """Append decode tokens. Cheap: just writes to ring buffer."""
        overflow = self.ring.write(key[:num_tokens], value[:num_tokens], num_tokens)
        if overflow is not None:
            k_over, v_over = overflow
            self.store.append_chunk(k_over, v_over)

    def flush(self):
        """Force-flush ring buffer to compressed store."""
        data = self.ring.drain()
        if data is not None:
            k, v = data
            self.store.append_chunk(k, v)

    def reset(self):
        self.ring.reset()
        self.store.reset()
        self._prefill_done = False