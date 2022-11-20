from torch import Tensor
from diffusers.models.attention import GEGLU


def forward(self: GEGLU, hidden_states: Tensor) -> Tensor:
  hidden_states, gate = self.proj(hidden_states).chunk(2, dim=1)
  return hidden_states * self.gelu(gate)

def adapt_geglu(t: GEGLU) -> None:
  setattr(t, 'forward', forward.__get__(t))