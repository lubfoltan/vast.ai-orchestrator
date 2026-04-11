"""Wrapper around the Vast.ai Python SDK for instance management."""

import logging
import time
from typing import Any, Dict, List, Optional

from vastai import VastAI

logger = logging.getLogger(__name__)


class VastAPIError(Exception):
    """Raised when a Vast.ai API call fails."""


class VastAPI:
    """Interact with Vast.ai through the Python SDK."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            self._sdk = VastAI(api_key=api_key, raw=True, quiet=True)
        except Exception as exc:
            raise VastAPIError(f"Failed to initialise Vast.ai SDK: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search_offers(
        self,
        min_gpu_ram: float = 8.0,
        max_price: float = 1.0,
        gpu_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for available GPU instances matching criteria."""
        query = f"gpu_ram>={min_gpu_ram} dph_total<={max_price} rentable=true"
        if gpu_name:
            query += f" gpu_name={gpu_name}"

        try:
            result = self._sdk.search_offers(
                query=query,
                order="dph_total",
                type="on-demand",
            )
        except Exception as exc:
            raise VastAPIError(f"search_offers failed: {exc}") from exc

        if isinstance(result, list):
            return result
        return []

    def create_instance(
        self,
        offer_id: int,
        image: str = "pytorch/pytorch:latest",
        disk_gb: int = 30,
    ) -> Dict[str, Any]:
        """Rent an instance from an offer."""
        try:
            result = self._sdk.create_instance(
                id=offer_id,
                image=image,
                disk=float(disk_gb),
            )
        except Exception as exc:
            raise VastAPIError(f"create_instance failed: {exc}") from exc

        if isinstance(result, dict):
            return result
        return {"success": True, "raw": str(result)}

    def get_instance(self, instance_id: int) -> Optional[Dict[str, Any]]:
        """Get details for a specific instance."""
        try:
            instances = self._sdk.show_instances()
        except Exception as exc:
            raise VastAPIError(f"show_instances failed: {exc}") from exc

        if isinstance(instances, list):
            for inst in instances:
                if inst.get("id") == instance_id:
                    return inst
        return None

    def list_instances(self) -> List[Dict[str, Any]]:
        """List all current instances."""
        try:
            result = self._sdk.show_instances()
        except Exception as exc:
            raise VastAPIError(f"show_instances failed: {exc}") from exc
        if isinstance(result, list):
            return result
        return []

    def stop_instance(self, instance_id: int) -> str:
        """Stop an instance (keeps it rented, stops billing for compute)."""
        try:
            result = self._sdk.stop_instance(id=instance_id)
        except Exception as exc:
            raise VastAPIError(f"stop_instance failed: {exc}") from exc
        return str(result) if result else "Instance stopped."

    def destroy_instance(self, instance_id: int) -> str:
        """Destroy (terminate) an instance to stop billing."""
        try:
            result = self._sdk.destroy_instance(id=instance_id)
        except Exception as exc:
            raise VastAPIError(f"destroy_instance failed: {exc}") from exc
        return str(result) if result else "Instance destroyed."

    def wait_for_instance_ready(
        self,
        instance_id: int,
        timeout_seconds: int = 300,
        poll_interval: int = 10,
        callback=None,
    ) -> Dict[str, Any]:
        """Poll until an instance reaches 'running' status."""
        elapsed = 0
        while elapsed < timeout_seconds:
            info = self.get_instance(instance_id)
            status = ""
            if info:
                status = info.get("actual_status", info.get("status_msg", ""))
            if callback:
                callback(f"Instance {instance_id}: {status} ({elapsed}s)")
            if status == "running":
                return info
            time.sleep(poll_interval)
            elapsed += poll_interval

        raise VastAPIError(
            f"Instance {instance_id} did not become ready within {timeout_seconds}s"
        )
