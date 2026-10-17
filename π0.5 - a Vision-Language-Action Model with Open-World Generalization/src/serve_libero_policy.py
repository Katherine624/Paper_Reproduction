"""Serve the official pi05_libero checkpoint on loopback only."""
import logging
import os
from pathlib import Path
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config

logging.basicConfig(level=logging.INFO, force=True)
root = Path(os.environ["PI05_ROOT"])
policy = policy_config.create_trained_policy(config.get_config("pi05_libero"), root / "checkpoints/pi05_libero")
server = websocket_policy_server.WebsocketPolicyServer(
    policy=policy, host="127.0.0.1", port=8765, metadata=policy.metadata,
)
print("OFFICIAL_LIBERO_POLICY_READY", flush=True)
server.serve_forever()
