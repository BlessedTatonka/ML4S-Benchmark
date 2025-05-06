"""
ML4S-Benchmark Abstract Task Classes

This package contains abstract base classes for different task types in the ML4S-Benchmark.
"""

from ml4s.abstasks.AbsTask import AbsTask
from ml4s.abstasks.AbsTaskEditing import AbsTaskEditing
from ml4s.abstasks.AbsTaskUnderstanding import AbsTaskUnderstanding
from ml4s.abstasks.TaskMetadata import DescriptiveStatistics, TaskMetadata

__all__ = ["AbsTask", "AbsTaskEditing", "AbsTaskUnderstanding", "TaskMetadata", "DescriptiveStatistics"] 