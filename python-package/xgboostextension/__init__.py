# coding=utf-8
"""XGBoostExtension: XGBRanker and XGBFeature"""

from __future__ import absolute_import

import os

try:
    from .xgbranker import XGBRanker
    from .xgbfeature import XGBFeature
except ImportError:
    pass

VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')

__all__ = ['XGBRanker', 'XGBFeature']
