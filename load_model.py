import tomllib
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify
from dataclasses import dataclass

class Equation:
    def __init__(self, func_name, expr):
        self.func_name = func_name
        self.expr = expr
        self.free_symbols = expr.free_symbols
        self.lamb = self._lambdify()

    def __repr__(self):
        return f"{self.func_name} = {self.expr}"

    def __call__(self, **kwargs):
        return self.lamb(**kwargs)
        # return self.expr.evalf(subs=kwargs)

    def diff(self, x):
        self.expr = self.expr.diff(x)
        self.free_symbols = self.expr.free_symbols
        self.lamb = self._lambdify()
        return self.expr

    def _lambdify(self, backend='numpy'):
        return lambdify(list(self.free_symbols), self.expr, backend)

@dataclass
class Model:
    states: list
    actions: list
    post_states: list

def load_model_file(fp):
    with open(fp, "rb") as f:
        data = tomllib.load(f)
    equations = [
        Equation(func_name, parse_expr(expr)) for func_name, expr in data["equations"].items()
    ]
    model = Model(data['model']['states'], data['model']['actions'], data['model']['post_states'])
    return model, equations
