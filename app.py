# app.py
from flask import Flask, render_template, request, jsonify
import sympy as sp
import re

app = Flask(__name__)

# --- small utilities -------------------------------------------------------
def sanitize_input(text: str) -> str:
    """Normalize user math input.
    - Replace caret '^' with Python power '**'
    - Normalize unicode division '÷' to '/'
    - Strip surrounding whitespace
    """
    if not isinstance(text, str):
        return text
    return text.replace('^', '**').replace('÷', '/').strip()

# helper: parse variable (default x)
def detect_symbols(expr_str):
    # find letters a-z as symbols (simple heuristic)
    names = sorted(set(re.findall(r"[A-Za-z]+", expr_str)))
    # exclude function names like sin, cos, diff, integrate, sqrt etc.
    funcs = {"sin","cos","tan","log","exp","sqrt","diff","integrate","solve","simplify"}
    names = [n for n in names if n.lower() not in funcs]
    if not names:
        return sp.symbols('x')
    return sp.symbols(' '.join(names))

def get_primary_symbol(symbols):
    """Return a single primary symbol from possibly a tuple of symbols."""
    if isinstance(symbols, (list, tuple)):
        return symbols[0] if symbols else sp.symbols('x')
    return symbols

# Step generators for common tasks
def solve_equation_steps(equation_str):
    # handles expressions like "2*x+3=7" or "x^2-5*x+6=0"
    normalized = sanitize_input(equation_str)
    lhs_rhs = normalized.split('=')
    if len(lhs_rhs) != 2:
        return None, ["Equation must contain exactly one '=' sign."]
    lhs, rhs = lhs_rhs
    syms = detect_symbols(normalized)
    try:
        lhs_e = sp.sympify(lhs)
        rhs_e = sp.sympify(rhs)
    except Exception as e:
        return None, [f"Parsing error: {e}"]

    eq = sp.Eq(lhs_e, rhs_e)
    # Try linear/quadratic detection by degrees
    poly = sp.simplify(lhs_e - rhs_e)
    steps = [f"Original equation: {sp.pretty(eq)}", f"Simplify to one side: {sp.pretty(sp.expand(poly))} = 0"]
    # Quadratic?
    try:
        p = sp.Poly(sp.expand(poly), syms)
        degree = p.degree()
    except Exception:
        degree = None

    if degree == 1:
        # linear ax + b = 0 -> ax = -b -> x = -b/a
        coeffs = p.all_coeffs()  # [a, b]
        a, b = coeffs
        steps.append(f"Linear equation with a={a}, b={b}")
        if a == 0:
            steps.append("Coefficient a is 0 -> no or infinite solutions.")
            sol = sp.solve(eq, syms)
            return sol, steps
        sol_val = -b/a
        steps.append(f"Solve: {syms} = -b/a = {sp.nsimplify(sol_val)}")
        sol = [sp.simplify(sol_val)]
        return sol, steps
    elif degree == 2:
        a, b, c = p.all_coeffs()
        steps.append(f"Quadratic detected with a={a}, b={b}, c={c}")
        disc = b**2 - 4*a*c
        steps.append(f"Discriminant Δ = b² - 4ac = {sp.pretty(disc)}")
        if disc < 0:
            steps.append("Δ < 0 → complex roots")
        sqrt_disc = sp.sqrt(disc)
        x1 = (-b + sqrt_disc) / (2*a)
        x2 = (-b - sqrt_disc) / (2*a)
        steps.append(f"Roots: x = (-b ± √Δ) / (2a) → {sp.pretty(x1)}, {sp.pretty(x2)}")
        sol = [sp.simplify(x1), sp.simplify(x2)]
        return sol, steps
    else:
        steps.append("Using SymPy general solve")
        try:
            sol = sp.solve(eq, syms)
            steps.append(f"Solutions: {sp.pretty(sol)}")
            return sol, steps
        except Exception as e:
            return None, [f"Could not solve: {e}"]

def simplify_steps(expr_str):
    expr_str = sanitize_input(expr_str)
    syms = detect_symbols(expr_str)
    try:
        e = sp.sympify(expr_str)
    except Exception as ex:
        return None, [f"Parsing error: {ex}"]
    steps = [f"Original: {sp.pretty(e)}"]
    simplified = sp.simplify(e)
    steps.append(f"Simplified: {sp.pretty(simplified)}")
    return simplified, steps

def derivative_steps(expr_str, var=None):
    # expr_str like "diff(sin(x)*x, x)" or "sin(x)*x"
    expr_str = sanitize_input(expr_str)
    if expr_str.strip().lower().startswith('diff('):
        # try to eval inside
        try:
            inner = expr_str.strip()[5:-1]
            parts = inner.split(',')
            expr_part = parts[0]
            var = parts[1].strip() if len(parts)>1 else None
        except:
            expr_part = expr_str
    else:
        expr_part = expr_str
    if var is None:
        var = get_primary_symbol(detect_symbols(expr_part))
    try:
        f = sp.sympify(expr_part)
    except Exception as ex:
        return None, [f"Parsing error: {ex}"]
    steps = [f"Function: {sp.pretty(f)}", f"Differentiate w.r.t {var}"]
    # Basic rule attempts
    d = sp.diff(f, var)
    steps.append(f"Result: {sp.pretty(d)}")
    return d, steps

def integral_steps(expr_str, var=None):
    # expr like "integrate(x^2, x)" or "x^2"
    expr_str = sanitize_input(expr_str)
    if expr_str.strip().lower().startswith('integrate('):
        try:
            inner = expr_str.strip()[10:-1]
            parts = inner.split(',')
            expr_part = parts[0]
            var = parts[1].strip() if len(parts)>1 else None
        except:
            expr_part = expr_str
    else:
        expr_part = expr_str
    if var is None:
        var = get_primary_symbol(detect_symbols(expr_part))
    try:
        f = sp.sympify(expr_part)
    except Exception as ex:
        return None, [f"Parsing error: {ex}"]
    steps = [f"Integrand: {sp.pretty(f)}", f"Integrate w.r.t {var}"]
    try:
        I = sp.integrate(f, var)
        steps.append(f"Result: {sp.pretty(I)} + C")
        return I, steps
    except Exception as e:
        return None, [f"Could not integrate: {e}"]

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    steps = []
    query = ""
    if request.method == 'POST':
        raw_query = request.form.get('query','')
        query = sanitize_input(raw_query)
        kind = request.form.get('kind','auto')
        # auto-detect common commands
        try:
            if kind == 'auto':
                qlow = query.lower()
                if '=' in query:
                    sol, steps = solve_equation_steps(query)
                    result = sol
                elif qlow.startswith('diff') or qlow.startswith('d/d') or 'd/d' in qlow:
                    out, steps = derivative_steps(query)
                    result = out
                elif qlow.startswith('integrate') or qlow.startswith('int(') or 'integrate' in qlow:
                    out, steps = integral_steps(query)
                    result = out
                elif qlow.startswith('simplify') or '/' in query or 'simplify' in qlow:
                    out, steps = simplify_steps(query)
                    result = out
                else:
                    # try evaluate or simplify
                    try:
                        val = sp.sympify(query)
                        result = sp.N(val)
                        steps = [f"Parsed value: {sp.pretty(val)}", f"Numeric evaluation: {result}"]
                    except Exception:
                        out, steps = simplify_steps(query)
                        result = out
            elif kind == 'solve':
                sol, steps = solve_equation_steps(query)
                result = sol
            elif kind == 'diff':
                out, steps = derivative_steps(query)
                result = out
            elif kind == 'integrate':
                out, steps = integral_steps(query)
                result = out
            elif kind == 'simplify':
                out, steps = simplify_steps(query)
                result = out
        except Exception as e:
            steps = [f"Error processing: {e}"]
            result = None
    return render_template('index.html', result=result, steps=steps, query=query)

@app.route('/api/solve', methods=['POST'])
def api_solve():
    data = request.json or {}
    raw_query = data.get('query','')
    query = sanitize_input(raw_query)
    kind = data.get('kind','auto')
    # reuse logic but return json
    # (for brevity, call index logic by simulating a post)
    # We'll do a very light implementation:
    if not query:
        return jsonify({"error":"No query provided"}), 400
    try:
        if '=' in query and kind in ('auto','solve'):
            sol, steps = solve_equation_steps(query)
            return jsonify({"result": [str(s) for s in sol] if sol else None, "steps": steps})
        qlow = query.lower()
        if kind in ('auto','diff') and (qlow.startswith('diff') or 'd/d' in qlow):
            out, steps = derivative_steps(query)
            return jsonify({"result": str(out), "steps": steps})
        if kind in ('auto','integrate') and (qlow.startswith('integrate') or qlow.startswith('int(') or 'integrate' in qlow):
            out, steps = integral_steps(query)
            return jsonify({"result": str(out), "steps": steps})
        if kind in ('auto','simplify') and (qlow.startswith('simplify') or 'simplify' in qlow or '/' in query):
            out, steps = simplify_steps(query)
            return jsonify({"result": str(out), "steps": steps})
        # else evaluate
        val = sp.sympify(query)
        return jsonify({"result": str(sp.N(val)), "steps":[f"Parsed: {str(val)}"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
