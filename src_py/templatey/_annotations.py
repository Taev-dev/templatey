from dataclasses import dataclass
from enum import Enum


class InterfaceAnnotationFlavor(Enum):
    SLOT = 'slot'
    VARIABLE = 'var'
    CONTENT = 'content'
    DYNAMIC = 'dynamic'


@dataclass(frozen=True)
class InterfaceAnnotation:
    flavor: InterfaceAnnotationFlavor
