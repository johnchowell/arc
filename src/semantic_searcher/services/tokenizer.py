import logging
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from semantic_searcher.config import settings

log = logging.getLogger(__name__)

TOKENIZER_PATH = settings.tokenizer_dir / "bpe_tokenizer.json"


class BPETokenizerService:
    def __init__(self):
        self.tokenizer: Tokenizer | None = None

    async def start(self):
        if TOKENIZER_PATH.exists():
            self.tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
            log.info("Loaded BPE tokenizer from %s", TOKENIZER_PATH)
        else:
            log.info("No trained tokenizer found at %s", TOKENIZER_PATH)

    def is_trained(self) -> bool:
        return self.tokenizer is not None

    def train(self, texts: list[str], vocab_size: int = 30000):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(TOKENIZER_PATH))
        self.tokenizer = tokenizer
        log.info("Trained BPE tokenizer (vocab=%d), saved to %s", vocab_size, TOKENIZER_PATH)

    def encode(self, text: str) -> list[int]:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not trained")
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not trained")
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int | None:
        if not self.tokenizer:
            return None
        return self.tokenizer.get_vocab_size()
