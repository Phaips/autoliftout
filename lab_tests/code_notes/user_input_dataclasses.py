from dataclasses import dataclass


@dataclass
class InventoryItem:
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand

item = InventoryItem('hammers', 10.49, 12)
print(item.total_cost())

@dataclass
class LamellaTrenches:
    name: str
    ion_beam_current: float
    

@dataclass
class JcutParameters:
    name: str
    ion_beam_current: float
    jcut_angle_degrees: float
