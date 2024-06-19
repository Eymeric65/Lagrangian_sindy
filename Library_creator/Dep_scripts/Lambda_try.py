import sympy as sp

# Define the symbolic variables and functions
t = sp.symbols("t")
#q0 = sp.Function('q0')(t)
#q1 = sp.Function('q1')(t)
m = [sp.Function('q0')(t), sp.Function('q1')(t)]

# Create the expression
g = sp.cos(m[0]) * 2 + m[1]  # Equivalent to q0(t)*2 + q1(t)

# Substitute q0(t) with a specific value
substituted_expr = g.subs(m[0], 1).evalf()  # Substitute q0(t) with 1
print(substituted_expr)  # Should output 2*1 + q1(t)

# Substitute the remaining symbolic variable q1(t)
q1_value = 3  # Example value for q1(t)
final_result = substituted_expr.subs(m[1], q1_value)
print(final_result)  # Should print 5
