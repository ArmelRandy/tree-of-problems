from top.bbh.utils import BBH_TASKS


def divide(problem_name: str, l2m: bool):
    """
    Get the solve prompt given input parameters.
    Arguments
    ---------
        - problem_name: str
            Name of the problem of interest. e.g. concatenation.
        - l2m: bool,
            Boolean which indicates if we use L2M prompting.
    """
    if problem_name == "algebraic":
        if l2m:
            from top.algebraic.divide import divide_l2m as divide_fn

            return lambda prompts, n_splits: divide_fn(prompts)
        else:
            from top.algebraic.divide import divide_fn as divide_fn

            return divide_fn
    elif problem_name == "coin":
        if l2m:
            from top.coin.divide import divide_l2m as divide_fn

            return lambda prompts, n_splits: divide_fn(prompts)
        else:
            from top.coin.divide import divide_fn as divide_fn

            return divide_fn
    elif problem_name == "concatenation":
        if l2m:
            from top.concatenation.divide import divide_l2m as divide_fn

            return lambda prompts, n_splits: divide_fn(prompts)
        else:
            from top.concatenation.divide import divide_fn as divide_fn

            return divide_fn
    elif problem_name in BBH_TASKS:
        from top.bbh.divide import divide_bbh

        def f(prompts, n_splits):
            return divide_bbh(
                prompts=prompts, n_splits=n_splits, description=problem_name
            )

        return f
    elif problem_name in ["sorting", "set_intersection", "keyword_counting"]:
        from top.got.divide import divide_got

        def f(prompts, n_splits):
            return divide_got(
                prompts=prompts, n_splits=n_splits, description=problem_name
            )

        return f
    else:
        raise ValueError(f"Unsupported problem name ({problem_name})")
