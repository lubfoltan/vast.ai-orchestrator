"""Wrapper around the Vast.ai Python SDK for instance management."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from vastai import VastAI

logger = logging.getLogger(__name__)

LogCallback = Optional[Callable[[str], None]]


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
        min_reliability: float = 0.95,
        min_dl_speed: float = 100.0,
        min_ul_speed: float = 50.0,
        log_cb: LogCallback = None,
    ) -> List[Dict[str, Any]]:
        """Search for available GPU offers, scored by best value.

        Filters by reliability and network speed, then ranks by a composite
        ''value score'' that balances price, GPU performance, and connectivity
        rather than sorting by lowest price alone.
        """
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

        if not isinstance(result, list):
            return []

        # ── Filter unreliable / slow machines ──
        filtered = []
        for offer in result:
            reliability = offer.get("reliability2", offer.get("reliability", 1.0)) or 1.0
            dl_speed = offer.get("inet_down", 0) or 0       # Mbps
            ul_speed = offer.get("inet_up", 0) or 0         # Mbps

            if reliability < min_reliability:
                continue
            if dl_speed < min_dl_speed:
                continue
            if ul_speed < min_ul_speed:
                continue
            filtered.append(offer)

        if log_cb:
            log_cb(f"Offers: {len(result)} total → {len(filtered)} after reliability/speed filter")

        # ── Score remaining offers ──
        # Value = GPU_performance × Network_speed / Price
        # Higher is better.
        for offer in filtered:
            gpu_ram = offer.get("gpu_ram", 8) or 8
            dl = offer.get("inet_down", 100) or 100
            ul = offer.get("inet_up", 50) or 50
            price = offer.get("dph_total", 1.0) or 1.0
            reliability = offer.get("reliability2", offer.get("reliability", 0.95)) or 0.95
            dlperf = offer.get("dlperf", 5) or 5  # deep learning perf score

            # Composite score (higher = better value for a short sprint)
            offer["_value_score"] = (
                (dlperf * 2.0)           # GPU compute perf (most important)
                * (gpu_ram / 8.0)        # VRAM bonus
                * reliability            # uptime reliability
                * min(dl + ul, 2000) / 500  # network throughput factor
                / max(price, 0.01)       # cost efficiency
            )

        # Sort by value score descending (best value first)
        filtered.sort(key=lambda o: o.get("_value_score", 0), reverse=True)

        if log_cb and filtered:
            top = filtered[0]
            log_cb(
                f"Top pick: {top.get('gpu_name', '?')} "
                f"({top.get('gpu_ram', '?')} GB) — "
                f"${top.get('dph_total', '?')}/hr — "
                f"score {top['_value_score']:.1f} — "
                f"reliability {top.get('reliability2', top.get('reliability', '?')):.0%}"
            )

        return filtered

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
