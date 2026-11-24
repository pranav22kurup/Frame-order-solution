#!/usr/bin/env python3
import os
import sys

import numpy as np
from PIL import Image


SIZE = 48


def frame_features(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32)

    gray = (
        rgb[:, :, 0] * 0.299
        + rgb[:, :, 1] * 0.587
        + rgb[:, :, 2] * 0.114
    )
    horizontal = np.abs(np.diff(gray, axis=1)).mean()
    vertical = np.abs(np.diff(gray, axis=0)).mean()

    histogram = []
    for channel in range(3):
        hist, _ = np.histogram(
            rgb[:, :, channel], bins=16, range=(0, 256), density=True
        )
        histogram.extend(hist)

    return (
        rgb.ravel(),
        np.float32(horizontal + vertical),
        np.asarray(histogram, dtype=np.float32),
    )


def distance_matrix(paths):
    features = [frame_features(path) for path in paths]
    rgb = np.stack([item[0] for item in features])
    edge = np.asarray([item[1] for item in features])
    hist = np.stack([item[2] for item in features])

    count = len(paths)
    distances = np.zeros((count, count), dtype=np.float64)
    for i in range(count - 1):
        pixel = np.mean(np.abs(rgb[i] - rgb[i + 1 :]), axis=1)
        histogram = np.mean(np.abs(hist[i] - hist[i + 1 :]), axis=1)
        edge_distance = np.abs(edge[i] - edge[i + 1 :])
        value = pixel / 255.0 + 0.10 * histogram + 0.08 * edge_distance
        distances[i, i + 1 :] = value
        distances[i + 1 :, i] = value
    return distances


def exact_path(distances):
    count = len(distances)
    size = 1 << count
    infinity = np.inf
    dp = np.full((size, count), infinity, dtype=np.float64)
    parent = np.full((size, count), -1, dtype=np.int16)

    for i in range(count):
        dp[1 << i, i] = 0.0

    for mask in range(size):
        remaining = (size - 1) ^ mask
        for end in range(count):
            if not (mask & (1 << end)):
                continue
            previous = dp[mask, end]
            if not np.isfinite(previous):
                continue
            candidates = remaining
            while candidates:
                bit = candidates & -candidates
                next_frame = bit.bit_length() - 1
                next_mask = mask | bit
                candidate = previous + distances[end, next_frame]
                if candidate < dp[next_mask, next_frame]:
                    dp[next_mask, next_frame] = candidate
                    parent[next_mask, next_frame] = end
                candidates ^= bit

    end = int(np.argmin(dp[size - 1]))
    path = []
    mask = size - 1
    while end != -1:
        path.append(end)
        previous = parent[mask, end]
        mask ^= 1 << end
        end = int(previous)
    path.reverse()
    return path


def cheapest_insertion(distances):
    count = len(distances)
    if count < 2:
        return list(range(count))

    flat = distances.copy()
    np.fill_diagonal(flat, np.inf)
    first, second = np.unravel_index(int(np.argmin(flat)), flat.shape)
    path = [int(first), int(second)]
    remaining = set(range(count)) - set(path)

    while remaining:
        best = None
        for node in remaining:
            for position in range(len(path)):
                left = path[position - 1] if position > 0 else None
                right = path[position] if position < len(path) else None
                increase = 0.0
                if left is not None:
                    increase += distances[left, node]
                if right is not None:
                    increase += distances[node, right]
                if left is not None and right is not None:
                    increase -= distances[left, right]
                elif left is None:
                    increase += 0.0
                elif right is None:
                    increase += 0.0

                candidate = (increase, len(path) - position, node, position)
                if best is None or candidate < best:
                    best = candidate
        _, _, node, position = best
        path.insert(position, node)
        remaining.remove(node)

    return path


def two_opt(path, distances):
    path = list(path)
    improved = True
    while improved:
        improved = False
        current = sum(distances[path[i], path[i + 1]] for i in range(len(path) - 1))
        for i in range(len(path) - 1):
            for j in range(i + 1, len(path) - 1):
                candidate = (
                    current
                    - distances[path[i], path[i + 1]]
                    - distances[path[j], path[j + 1]]
                    + distances[path[i], path[j]]
                    + distances[path[i + 1], path[j + 1]]
                )
                if candidate + 1e-9 < current:
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    current = candidate
                    improved = True
                    break
            if improved:
                break
    return path


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "/frames"
    names = sorted(
        name
        for name in os.listdir(directory)
        if name.lower().endswith(".png")
    )
    if not names:
        return

    paths = [os.path.join(directory, name) for name in names]
    distances = distance_matrix(paths)

    if len(names) <= 18:
        order = exact_path(distances)
    else:
        order = cheapest_insertion(distances)
        order = two_opt(order, distances)

    for index in order:
        print(names[index])


if __name__ == "__main__":
    main()
