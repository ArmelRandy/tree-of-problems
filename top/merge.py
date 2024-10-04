from top.bbh.utils import BBH_TASKS


def get_merge_prompt(problem_name: str, l2m: bool):
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
            from top.algebraic.merge import get_merge_prompt_l2m as merge
        else:
            from top.algebraic.merge import get_merge_prompt as merge
        return merge
    elif problem_name == "coin":
        if l2m:
            from top.coin.merge import get_merge_prompt_l2m as merge
        else:
            from top.coin.merge import get_merge_prompt as merge
        return merge
    elif problem_name == "concatenation":
        if l2m:
            from top.concatenation.merge import get_merge_prompt_l2m as merge
        else:
            from top.concatenation.merge import get_merge_prompt as merge
        return merge
    elif problem_name in BBH_TASKS:
        from top.bbh.merge import get_merge_prompt_bbh

        def f(sentences, inputs, outputs):
            return get_merge_prompt_bbh(
                sentences=sentences,
                inputs=inputs,
                outputs=outputs,
                description=problem_name,
            )

        return f
    elif problem_name in ["sorting", "set_intersection", "keyword_counting"]:
        from top.got.merge import get_merge_prompt_got

        def f(sentences, inputs, outputs):
            return get_merge_prompt_got(
                sentences=sentences,
                inputs=inputs,
                outputs=outputs,
                description=problem_name,
            )

        return f
    else:
        raise ValueError(f"Unsupported problem name ({problem_name})")
