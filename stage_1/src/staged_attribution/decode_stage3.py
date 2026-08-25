from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from shapely.geometry import LineString


def _single_context(context: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    route = context["route"]
    route_lengths = context["route_lengths"]
    if route.ndim == 1:
        route = route.unsqueeze(0)
    if route_lengths.ndim == 0:
        route_lengths = route_lengths.unsqueeze(0)
    return {
        "route": route.to(device),
        "route_lengths": route_lengths.to(device),
        "route_geometry": context["route_geometry"].to(device),
        "route_geometry_lengths": context["route_geometry_lengths"].to(device),
    }


@torch.no_grad()
def decode_variant_records(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, object]],
    contexts_by_id: Mapping[int, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, object]]:
    model = model.to(device)
    model.eval()
    result: list[dict[str, object]] = []
    for source in records:
        record = dict(source)
        if record.get("status") != "ready":
            result.append(record)
            continue
        traj_id = int(record["traj_id"])
        if traj_id not in contexts_by_id:
            record["status"] = "failed"
            record["failure"] = {
                "reason": "missing_decode_context",
                "message": f"no route geometry context for traj_id={traj_id}",
            }
            result.append(record)
            continue
        context = _single_context(contexts_by_id[traj_id], device)
        route_length = int(context["route_lengths"][0])
        context_route = context["route"][0, :route_length].detach().cpu().tolist()
        if context_route != [int(value) for value in record["road_seq"]]:
            record["status"] = "failed"
            record["failure"] = {
                "reason": "oracle_route_mismatch",
                "message": "variant road_seq differs from decoder oracle context",
            }
            result.append(record)
            continue
        try:
            codes = torch.as_tensor(record["codes"], dtype=torch.long, device=device)
            code_lengths = torch.tensor([len(codes)], dtype=torch.long, device=device)
            lens_type = torch.as_tensor(
                [record["lens_type"]], dtype=torch.long, device=device
            )
            percent, dxy, lengths = model.decode(codes, code_lengths, lens_type, context)
            length = int(lengths[0])
            record.update(
                {
                    "status": "decoded",
                    "pred_percent": percent[0, :length].detach().cpu().tolist(),
                    "pred_dxy": dxy[0, :length].detach().cpu().tolist(),
                    "decoded_length": length,
                    "failure": None,
                }
            )
        except Exception as error:
            record["status"] = "failed"
            record["failure"] = {"reason": "decode_error", "message": str(error)}
        result.append(record)
    return result


def _route_line(
    road_seq: Sequence[int], geometries: Mapping[int, np.ndarray]
) -> tuple[LineString, np.ndarray]:
    coordinates: list[tuple[float, float]] = []
    lengths: list[float] = []
    for road in road_seq:
        if road not in geometries:
            raise ValueError(f"missing geometry for road {road}")
        geometry = np.asarray(geometries[road], dtype=np.float64)
        if geometry.ndim != 2 or geometry.shape[0] < 2 or geometry.shape[1] != 2:
            raise ValueError(f"road {road} geometry must have shape [points>=2,2]")
        line = LineString(geometry)
        lengths.append(float(line.length))
        points = [tuple(value) for value in geometry.tolist()]
        coordinates.extend(points if not coordinates else points[1:])
    if not coordinates or sum(lengths) <= 0:
        raise ValueError("oracle route geometry has zero length")
    return LineString(coordinates), np.asarray(lengths, dtype=np.float64)


def reconstruct_gps(
    decoded: Mapping[str, object],
    road_geometries: Mapping[int, np.ndarray],
    z_score: Mapping[str, float],
) -> dict[str, object]:
    if decoded.get("status") != "decoded":
        return dict(decoded)
    result = dict(decoded)
    try:
        road_seq = np.asarray(decoded["road_seq"], dtype=np.int64)
        route, road_lengths = _route_line(road_seq.tolist(), road_geometries)
        percent = np.asarray(decoded["pred_percent"], dtype=np.float64)
        dxy = np.asarray(decoded["pred_dxy"], dtype=np.float64)
        if dxy.shape != (len(percent), 2):
            raise ValueError("pred_dxy must align with pred_percent")
        percent = percent * float(z_score["std_percent_dist"]) + float(
            z_score["mean_percent_dist"]
        )
        percent = np.clip(np.abs(np.cumsum(percent)), 0.0, 0.9999999)
        dx = dxy[:, 0] * float(z_score["std_dx"]) + float(z_score["mean_dx"])
        dy = dxy[:, 1] * float(z_score["std_dy"]) + float(z_score["mean_dy"])
        points = []
        for fraction, x_offset, y_offset in zip(percent, dx, dy):
            projected = route.interpolate(float(fraction) * route.length)
            points.append([projected.x + x_offset, projected.y + y_offset])
        cumulative = np.cumsum(road_lengths) / road_lengths.sum()
        road_index = np.searchsorted(cumulative, percent, side="right")
        road_index = np.clip(road_index, 0, len(road_seq) - 1)
        result.update(
            {
                "status": "reconstructed",
                "gps_traj": points,
                "road_traj": road_seq[road_index].astype(int).tolist(),
                "failure": None,
            }
        )
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = {"reason": "reconstruction_error", "message": str(error)}
    return result
