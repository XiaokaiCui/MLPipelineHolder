from types import CodeType
from typing import Any, Callable


def code_objects_equal(
    existing: CodeType,
    new: CodeType,
    values_equal: Callable[[Any, Any], bool],
) -> bool:
    if (
        existing.co_argcount != new.co_argcount
        or existing.co_posonlyargcount != new.co_posonlyargcount
        or existing.co_kwonlyargcount != new.co_kwonlyargcount
        or existing.co_nlocals != new.co_nlocals
        or existing.co_stacksize != new.co_stacksize
        or existing.co_flags != new.co_flags
        or existing.co_code != new.co_code
        or existing.co_names != new.co_names
        or existing.co_varnames != new.co_varnames
        or existing.co_freevars != new.co_freevars
        or existing.co_cellvars != new.co_cellvars
        or existing.co_exceptiontable != new.co_exceptiontable
        or len(existing.co_consts) != len(new.co_consts)
    ):
        return False
    for existing_constant, new_constant in zip(existing.co_consts, new.co_consts):
        if isinstance(existing_constant, CodeType) or isinstance(new_constant, CodeType):
            if not (
                isinstance(existing_constant, CodeType)
                and isinstance(new_constant, CodeType)
                and code_objects_equal(existing_constant, new_constant, values_equal)
            ):
                return False
            continue
        if not values_equal(existing_constant, new_constant):
            return False
    return True
