from .agent_v1 import AgentV1
from .agent_v2 import AgentV2
from .agent_v3 import AgentV3
from .agent_v4 import AgentV4
from .agent_v5 import AgentV5

# 変更(v6): PyTorchを遅延読込するテンソル版を登録。
from .agent_v6 import AgentV6

# 変更(v7): PyTorchを遅延読込するテンソル版を登録。
from .agent_v7 import AgentV7

__all__ = ["AgentV1", "AgentV2", "AgentV3", "AgentV4", "AgentV5", "AgentV6", "AgentV7"]
