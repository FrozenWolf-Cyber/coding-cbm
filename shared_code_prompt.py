from __future__ import annotations

from typing import Optional


LCB_LLAMA3_INSTRUCT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

_LCB_SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question (problem specification) "
    "and will generate a correct Python program that matches the specification and passes all tests."
)
_LCB_FORMAT_NO_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)
_LCB_FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
)


def build_lcb_user_prompt(
    problem_description: str,
    starter_code: str = "",
    language: str = "python",
) -> str:
    language = (language or "python").strip().lower()
    if starter_code:
        return (
            f"### Question:\n{problem_description}\n\n"
            f"### Format: {_LCB_FORMAT_WITH_STARTER}\n"
            f"```{language}\n{starter_code}\n```\n\n"
            f"### Answer: (use the provided format with backticks)\n\n"
        )
    return (
        f"### Question:\n{problem_description}\n\n"
        f"### Format: {_LCB_FORMAT_NO_STARTER}\n"
        f"```{language}\n# YOUR CODE HERE\n```\n\n"
        f"### Answer: (use the provided format with backticks)\n\n"
    )


def format_lcb_llama3_instruct_prompt(
    tokenizer,
    problem_description: str,
    starter_code: str = "",
    language: str = "python",
    system_prompt: Optional[str] = None,
) -> str:
    user_body = build_lcb_user_prompt(
        problem_description=problem_description,
        starter_code=starter_code,
        language=language,
    )
    messages = [
        {"role": "system", "content": system_prompt or _LCB_SYSTEM_PROMPT},
        {"role": "user", "content": user_body},
    ]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
    return f"### Problem:\n{problem_description}\n\n### Solution:\n```{language}\n"
