"""
Neuro-Evolution Module for ANERLE
Implements genotype representation, population management, and evolutionary operators.
"""
import numpy as np
import random
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import hashlib
from datetime import datetime
from firebase_admin import firestore

logger = logging.getLogger(__name__)

@dataclass
class LayerGene:
    """Gene representing a neural network layer"""
    neurons: int
    activation: str
    dropout_rate: float = 0.0
    batch_norm: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "neurons": self.neurons,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "batch_norm": self.batch_norm
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LayerGene':
        return cls(
            neurons=data["neurons"],
            activation=data["activation"],
            dropout_rate=data.get("dropout_rate", 0.0),
            batch_norm=data.get("batch_norm", False)
        )

@dataclass
class IndicatorGene:
    """Gene representing a technical indicator configuration"""
    name: str
    period: int
    parameters: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "period": self.period,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IndicatorGene':
        return cls(
            name=data["name"],
            period=data["period"],
            parameters=data.get("parameters", {})
        )

@dataclass
class TradingStrategyGenotype:
    """
    Complete genotype representing a trading strategy
    Contains neural network architecture and indicator configurations
    """
    id: str = field(default_factory=lambda: hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:16])
    layers: List[LayerGene] = field(default_factory=list)
    indicators