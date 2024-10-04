from top.bbh.utils import BBH_TASKS


def get_solve_prompt(problem_name: str, description: str, k: int):
    """
    Get the solve prompt given input parameters.
    Arguments
    ---------
        - problem_name: str
            Name of the problem of interest. e.g. concatenation.
        - description: str,
            Use standard few-shot or CoT few-shot. e.g. "cot".
        - k: int,
            Number of few-shot demonstrations.
    """
    if problem_name == "algebraic":
        from top.algebraic.solve import get_solve_prompt as solve

        return lambda x: solve(prompt=x, description=description, k=k)
    elif problem_name == "coin":
        from top.coin.solve import get_solve_prompt as solve

        return lambda x: solve(prompt=x, description=description, k=k)
    elif problem_name == "concatenation":
        from top.concatenation.solve import get_solve_prompt as solve

        return lambda x: solve(prompt=x, description=description, k=k)
    elif problem_name in BBH_TASKS:
        from top.bbh.solve import get_solve_prompt_bbh

        return lambda x: get_solve_prompt_bbh(
            sentence=x, problem_name=problem_name, description=description
        )
    elif problem_name in ["sorting", "set_intersection", "keyword_counting"]:
        from top.got.solve import get_solve_prompt_got

        return lambda x: get_solve_prompt_got(
            sentence=x, problem_name=problem_name, description=description
        )
    else:
        raise ValueError(f"Unsupported problem name ({problem_name})")