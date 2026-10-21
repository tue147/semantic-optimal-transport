from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from torch import Tensor
from dataclasses import dataclass

@dataclass
class GraphName:
    feature_graph: str = 'feature_graph'
    computational_graph: str = 'computational_graph'
    
class Node(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
class Index(ABC):
    @property
    @abstractmethod
    def as_index(self) -> Tuple[int|slice, ...]:
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
class Graph(ABC):
    
    @abstractmethod
    def add_node(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def add_edge(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def delete_node(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def delete_edge(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def update_node(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def update_edge(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def find_deleted_nodes(self, *args, **kwargs) -> List[Tuple[Node, Index]]:
        pass
    
    @abstractmethod
    def find_deleted_edges(self, *args, **kwargs) -> List[Tuple[Node, Index, Node, Index]]:
        pass
    
    @abstractmethod
    def iterate_nodes(self) -> List[Tuple[Node, Index]]:
        pass
    
    @abstractmethod
    def iterate_edges(self) -> List[Tuple[Node, Index, Node, Index]]:
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward_backward_gradient(self, *args, **kwargs) -> Tuple[
        Dict[Tuple[Node, Index], Any],  # node effects
        Dict[Tuple[Node, Index, Node, Index], Any], # edge effects
    ]:
        pass
    
    @abstractmethod
    def model_setup(self) -> None:
        pass
    
    @abstractmethod
    def run_model(self, *args, **kwargs) -> Tuple[Any, Any]:
        pass
    
    @abstractmethod
    def graph_type(self) -> str:
        pass
    