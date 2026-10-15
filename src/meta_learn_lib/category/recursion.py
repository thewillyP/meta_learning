from dataclasses import dataclass
from functools import reduce
from typing import Callable

from pyrsistent import PList, plist

from .lens import Lens, copy, sequence, wiring
from .lib_types import Proxy, Unit
from .mealy import Mealy
from .paralens import ParaLens


type Layer[A, X] = tuple[A, PList[X]]


def fold_value[T, A, B](
    project: Callable[[T], Layer[A, T]],
    algebra: Callable[[Layer[A, B]], B],
) -> Callable[[T], B]:
    def run(tree: T) -> B:
        payload, children = project(tree)
        return algebra((payload, plist(map(run, children))))

    return run


def sequence_shared[P, DP, X, DX, Y, DY](
    local: ParaLens[P, DP, X, DX, Y, DY],
    zero: Callable[[P], DP],
    add: Callable[[DP, DP], DP],
) -> ParaLens[P, DP, PList[X], PList[DX], PList[Y], PList[DY]]:
    def run(p_xs: tuple[P, PList[X]]) -> tuple[PList[Y], Callable[[PList[DY]], tuple[DP, PList[DX]]]]:
        p, xs = p_xs
        ys, back = sequence(Lens[X, tuple[DP, DX], Y, DY](lambda x: local.arrow.run((p, x)))).run(xs)

        def backward(dys: PList[DY]) -> tuple[DP, PList[DX]]:
            feedback = back(dys)
            return reduce(add, (dp for dp, _ in feedback), zero(p)), plist(dx for _, dx in feedback)

        return ys, backward

    return ParaLens(Lens(run))


def fold[P, DP, T, DT, A, DA, B, DB](
    project: Lens[T, DT, Layer[A, T], Layer[DA, DT]],
    algebra: ParaLens[P, DP, Layer[A, B], Layer[DA, DB], B, DB],
    zero: Callable[[P], DP],
    add: Callable[[DP, DP], DP],
) -> ParaLens[P, DP, T, DT, B, DB]:
    def run(p_tree: tuple[P, T]) -> tuple[B, Callable[[DB], tuple[DP, DT]]]:
        return body.arrow.run(p_tree)

    recursive = ParaLens(Lens(run))
    body = (
        sequence_shared(recursive, zero, add)
        .second(Proxy[tuple[A, DA]]())
        .then(algebra)
        .reparam(copy(Proxy[tuple[P, DP]](), add))
        .pre(project)
    )
    return body


@dataclass(frozen=True)
class End[S]:
    value: S


@dataclass(frozen=True)
class Item[X]:
    value: X


def list_unroll[S, X](s_xs: tuple[S, PList[X]]) -> Layer[End[S] | Item[X], tuple[S, PList[X]]]:
    s, xs = s_xs
    match xs:
        case []:
            return End(s), plist()
        case _:
            return Item(xs.first), plist(((s, xs.rest),))


def list_roll[S, X](layer: Layer[End[S] | Item[X], tuple[S, PList[X]]]) -> tuple[S, PList[X]]:
    match layer:
        case End(s), []:
            return s, plist()
        case Item(x), [(s, xs)]:
            return s, xs.cons(x)
        case _:
            raise ValueError("expected an end with no children or an item with one child")


def fold_list[P, DP, S, DS, X, DX](
    cell: ParaLens[P, DP, tuple[S, X], tuple[DS, DX], S, DS],
    zero: Callable[[P], DP],
    add: Callable[[DP, DP], DP],
) -> ParaLens[P, DP, tuple[S, PList[X]], tuple[DS, PList[DX]], S, DS]:
    def run(
        p_layer: tuple[P, Layer[End[S] | Item[X], S]],
    ) -> tuple[S, Callable[[DS], tuple[DP, Layer[End[DS] | Item[DX], DS]]]]:
        p, layer = p_layer
        match layer:
            case End(s), []:
                return s, lambda ds: (zero(p), (End(ds), plist()))
            case Item(x), [s]:
                next_s, back = cell.arrow.run((p, (s, x)))

                def backward(ds: DS) -> tuple[DP, Layer[End[DS] | Item[DX], DS]]:
                    dp, (previous_ds, dx) = back(ds)
                    return dp, (Item(dx), plist((previous_ds,)))

                return next_s, backward
            case _:
                raise ValueError("expected an end with no children or an item with one child")

    def reverse[A, B](a_bs: tuple[A, PList[B]]) -> tuple[A, PList[B]]:
        a, bs = a_bs
        return a, bs.reverse()

    project: Lens[
        tuple[S, PList[X]],
        tuple[DS, PList[DX]],
        Layer[End[S] | Item[X], tuple[S, PList[X]]],
        Layer[End[DS] | Item[DX], tuple[DS, PList[DX]]],
    ] = wiring(list_unroll, list_roll)
    return fold(project, ParaLens(Lens(run)), zero, add).pre(wiring(reverse, reverse))


def scan_list[P, DP, S, DS, X, DX, Y, DY](
    cell: Mealy[P, DP, S, DS, X, DX, Y, DY],
    zero: Callable[[P], DP],
    add: Callable[[DP, DP], DP],
) -> Mealy[P, DP, S, DS, PList[X], PList[DX], PList[Y], PList[DY]]:
    def run(
        p_syx: tuple[P, tuple[tuple[S, PList[Y]], X]],
    ) -> tuple[tuple[S, PList[Y]], Callable[[tuple[DS, PList[DY]]], tuple[DP, tuple[tuple[DS, PList[DY]], DX]]]]:
        p, ((s, ys), x) = p_syx
        (next_s, y), back = cell.step.arrow.run((p, (s, x)))

        def backward(ds_dys: tuple[DS, PList[DY]]) -> tuple[DP, tuple[tuple[DS, PList[DY]], DX]]:
            ds, dys = ds_dys
            dp, (previous_ds, dx) = back((ds, dys.first))
            return dp, ((previous_ds, dys.rest), dx)

        return (next_s, ys.cons(y)), backward

    def start(s_xs: tuple[S, PList[X]]) -> tuple[tuple[S, PList[Y]], PList[X]]:
        s, xs = s_xs
        return (s, plist()), xs

    def finish(ds_dys_dxs: tuple[tuple[DS, PList[DY]], PList[DX]]) -> tuple[DS, PList[DX]]:
        (ds, empty), dxs = ds_dys_dxs
        [] = empty
        return ds, dxs

    def reverse[A, B](a_bs: tuple[A, PList[B]]) -> tuple[A, PList[B]]:
        a, bs = a_bs
        return a, bs.reverse()

    return Mealy(fold_list(ParaLens(Lens(run)), zero, add).pre(wiring(start, finish)).post(wiring(reverse, reverse)))


@dataclass(frozen=True)
class Leaf[A]:
    value: A


@dataclass(frozen=True)
class Branch[A]:
    left: A
    right: A


type Tree[A] = Leaf[A] | Branch[Tree[A]]


def tree_unroll[A](tree: Tree[A]) -> Layer[Leaf[A] | Unit, Tree[A]]:
    match tree:
        case Leaf(value):
            return Leaf(value), plist()
        case Branch(left, right):
            return (), plist((left, right))


def tree_roll[A](layer: Layer[Leaf[A] | Unit, Tree[A]]) -> Tree[A]:
    match layer:
        case Leaf(value), []:
            return Leaf(value)
        case (), [left, right]:
            return Branch(left, right)
        case _:
            raise ValueError("expected a leaf with no children or a branch with two children")


def fold_tree[P, DP, A, DA, B, DB](
    leaf: ParaLens[P, DP, A, DA, B, DB],
    branch: ParaLens[P, DP, tuple[B, B], tuple[DB, DB], B, DB],
    zero: Callable[[P], DP],
    add: Callable[[DP, DP], DP],
) -> ParaLens[P, DP, Tree[A], Tree[DA], B, DB]:
    def run(
        p_layer: tuple[P, Layer[Leaf[A] | Unit, B]],
    ) -> tuple[B, Callable[[DB], tuple[DP, Layer[Leaf[DA] | Unit, DB]]]]:
        p, layer = p_layer
        match layer:
            case Leaf(a), []:
                b, back = leaf.arrow.run((p, a))

                def backward_leaf(db: DB) -> tuple[DP, Layer[Leaf[DA] | Unit, DB]]:
                    dp, da = back(db)
                    return dp, (Leaf(da), plist())

                return b, backward_leaf
            case (), [left, right]:
                b, back = branch.arrow.run((p, (left, right)))

                def backward_branch(db: DB) -> tuple[DP, Layer[Leaf[DA] | Unit, DB]]:
                    dp, (dl, dr) = back(db)
                    return dp, ((), plist((dl, dr)))

                return b, backward_branch
            case _:
                raise ValueError("expected a leaf with no children or a branch with two children")

    project: Lens[Tree[A], Tree[DA], Layer[Leaf[A] | Unit, Tree[A]], Layer[Leaf[DA] | Unit, Tree[DA]]] = wiring(
        tree_unroll, tree_roll
    )
    return fold(project, ParaLens(Lens(run)), zero, add)
