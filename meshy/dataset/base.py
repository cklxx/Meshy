from typing import Any, List

import datasets

from meshy.utils.sample import Sample, SampleBuilder


class Dataset:

    def __init__(
        self,
        hf_kwargs: dict[str, Any],
        batch_size: int,
        seed: int | None = None,
        start_index: int = 0,
        wrap_epochs: bool = False,
    ):
        # Keep the unshuffled split so a stream that runs past one epoch can
        # reshuffle with a fresh per-epoch seed (``_apply_epoch``).
        self._base_dataset = datasets.load_dataset(**hf_kwargs)
        self.dataset = self._base_dataset
        self.batch_size = batch_size
        self.base_seed = seed
        # Position as (epoch, prompt-within-epoch). ``start_index`` is a global
        # prompt offset and may itself span more than one epoch.
        self.epoch = 0
        self.index = 0
        # When False (default) a batch that would cross the epoch end returns
        # [] — the original stop-at-exhaustion behaviour. Bounded continuations
        # set True to reshuffle into a new epoch instead of ending the run.
        self.wrap_epochs = bool(wrap_epochs)
        self._apply_epoch(0)
        if start_index:
            self.seek(start_index)

    def _apply_epoch(self, epoch: int) -> None:
        """Reshuffle the underlying split for zero-based ``epoch``.

        Epoch 0 uses ``base_seed`` verbatim; each later epoch uses
        ``base_seed + epoch`` so a continued run never repeats an ordering.
        Without a seed the ordering is the raw split for every epoch.
        """
        self.epoch = epoch
        if self.base_seed is not None:
            self.dataset = self._base_dataset.shuffle(seed=self.base_seed + epoch)

    def seek(self, global_prompt_offset: int) -> None:
        """Position the stream at a global prompt offset (may span epochs)."""
        n = len(self._base_dataset)
        epoch, within = divmod(global_prompt_offset, n)
        self._apply_epoch(epoch)
        self.index = within

    def next_batch(self, builder: SampleBuilder) -> List[Sample]:
        n = len(self._base_dataset)
        # Gather a full batch even if it straddles an epoch boundary.
        if self.wrap_epochs:
            rows: List[dict] = []
            while len(rows) < self.batch_size:
                take = min(self.batch_size - len(rows), n - self.index)
                sel = self.dataset.select(range(self.index, self.index + take))
                rows.extend(sel)
                self.index += take
                if self.index >= n:
                    self._apply_epoch(self.epoch + 1)
                    self.index = 0
            return [self.apply_chat_template(data, builder) for data in rows]

        if self.index + self.batch_size >= n:
            return []
        batch_data = self.dataset.select(range(self.index, self.index + self.batch_size))
        self.index += self.batch_size
        return [self.apply_chat_template(data, builder) for data in batch_data]

    def apply_chat_template(self, data: dict, builder: SampleBuilder) -> Sample:
        raise NotImplementedError
