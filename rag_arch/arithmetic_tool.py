"""Small Decimal calculator; arithmetic is never an engineering judgement."""
from __future__ import annotations

import ast
from decimal import Context, DecimalException, Underflow, localcontext
import re


_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_PRECISION = 50


def calculate(expression: str) -> dict:
    """计算数字、加减乘除和括号；不判断单位、适用条件或工程结论。"""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression 须为非空数字四则运算表达式。")
    source = expression.strip()
    if len(source) > 1000:
        raise ValueError("表达式最多 1000 个字符。")
    if re.fullmatch(r"[0-9eE+\-*/().\s]+", source) is None:
        raise ValueError("仅支持数字、加减乘除和括号，不支持函数、属性或单位。")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("四则运算表达式语法无效。") from exc
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("表达式过于复杂，请拆成较小的计算。")

    with localcontext(Context(prec=_PRECISION)) as context:
        context.traps[Underflow] = True

        def compute(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                # Parse the original decimal literal, never Python's float.
                literal = ast.get_source_segment(source, node)
                if literal is not None and _NUMBER.fullmatch(literal):
                    return context.create_decimal(literal)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = compute(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = compute(node.left), compute(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                return left / right
            raise ValueError("仅支持数字、加减乘除和括号，不支持函数或其他运算符。")

        try:
            result = compute(tree.body)
        except DecimalException as exc:
            raise ValueError("表达式无法计算：除数为零、数值超出范围或运算无效。") from exc
    return {"expression": expression, "result": str(result), "precision": _PRECISION,
            "note": "仅执行数字四则运算，不判断单位、适用条件或工程结论。"}
