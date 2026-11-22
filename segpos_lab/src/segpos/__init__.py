"""segpos: 中文分词与词性标注增量交互组件。"""
from .engine import IncrementalEngine
from .models import Config, Lexicon, ProbTable, Sentence

__all__ = ["IncrementalEngine", "Config", "Lexicon", "ProbTable", "Sentence"]
__version__ = "1.0.0"
