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

    def take_prompts(
        self, builder: SampleBuilder, n: int, *, window_start: bool = False
    ) -> List[Sample]:
        """Return up to ``n`` prompts and advance the stream position by that many.

        Unlike :meth:`next_batch` this is not bounded by ``batch_size`` and does
        not interact with any subclass per-window run counter: DAPO dynamic
        sampling uses it to draw *replacement* prompts after dropping a
        zero-variance group, so a discarded prompt still advances the data
        position (no prompt is ever served twice). ``window_start`` marks the
        first draw of a window (replacement draws pass False), which a
        run-length-bounded subclass uses to count windows rather than draws.
        ``wrap_epochs`` reshuffles a short tail across the epoch boundary;
        without it the returned list is simply short when the split runs out.
        """
        if n <= 0:
            return []
        if window_start and not self.begin_window():
            # A run-length-bounded dataset declines the next window outright.
            return []
        if not self.wrap_epochs:
            total = len(self._base_dataset)
            take = min(n, total - self.index)
            if take <= 0:
                return []
            data = self.dataset.select(range(self.index, self.index + take))
            self.index += take
            return [self.apply_chat_template(row, builder) for row in data]

        rows: List[dict] = []
        total = len(self._base_dataset)
        while len(rows) < n:
            take = min(n - len(rows), total - self.index)
            rows.extend(self.dataset.select(range(self.index, self.index + take)))
            self.index += take
            if self.index >= total:
                self._apply_epoch(self.epoch + 1)
                self.index = 0
        return [self.apply_chat_template(row, builder) for row in rows]

    def begin_window(self) -> bool:
        """Reserve one output window before its first prompt draw.

        Base streams are unbounded and always accept. A subclass that bounds the
        run length (e.g. the V100 recipe's ``BoundedGSM8K``) overrides this to
        consume one window of budget and return False when the run is over, so
        the dynamic-sampling path -- which draws several times per window --
        counts windows rather than individual draws.
        """
        return True

    def apply_chat_template(self, data: dict, builder: SampleBuilder) -> Sample:
        raise NotImplementedError
