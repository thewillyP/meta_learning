from typing import Iterable

import equinox as eqx
import jax

from meta_learn_lib.interface import Accessor


def create_axes[ENV](env: ENV, accessors: Iterable[Accessor[ENV, int | None]]) -> ENV:
    axes = jax.tree.map(lambda _: None, env)
    for acc in accessors:
        val = acc.get(env)
        if val is None:
            continue
        zeros_subtree = jax.tree.map(lambda _: 0, val)
        axes = acc.put(axes, zeros_subtree)
    return axes


def diff_axes[ENV](old_env: ENV, new_env: ENV, accessors: Iterable[Accessor[ENV, object]]) -> ENV:
    old_leaf_ids: set[int] = set()
    for acc in accessors:
        old_val = acc.get(old_env)
        if old_val is None:
            continue
        for a in jax.tree.leaves(old_val):
            if eqx.is_array(a):
                old_leaf_ids.add(id(a))

    axes = jax.tree.map(lambda _: None, new_env)
    for acc in accessors:
        new_val = acc.get(new_env)
        if new_val is None:
            continue
        new_arrays = [a for a in jax.tree.leaves(new_val) if eqx.is_array(a)]
        if new_arrays and not all(id(a) in old_leaf_ids for a in new_arrays):
            zeros_subtree = jax.tree.map(lambda _: 0, new_val)
            axes = acc.put(axes, zeros_subtree)
    return axes
