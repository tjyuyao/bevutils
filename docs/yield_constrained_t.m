%% 
clear; clc
syms r00 r01 r02 r10 r11 r12 r20 r21 r22 real
R = [r00, r01, r02; r10, r11, r12; r20, r21, r22];
syms n0 n1 n2 real
n = [n0; n1; n2];
syms p0 p1 b0 b1 real
p1 = [p0; p1; 1];
syms t0 t1 t2 real
t = [t0; t1; t2];
p2 = (R - t * n') * p1;
p2 = p2 ./ p2(3);
eqn_t0 = p2(1) == b0;
sol_t0 = solve(eqn_t0, t0);
eqn_t1 = p2(2) == b1;
sol_t1 = solve(eqn_t1, t1);

%% 
clc
idx = 0;
fmd = fopen('./solution_t.md', 'w+');
fpy = fopen('./solution_t.py', 'w+');
for sol = [sol_t0, sol_t1]
    sol_ = simplify(sol);
    str = char(sol_);
    fprintf(fmd, "$$ t_%d = %s $$\n\n", idx, latex(sol_));
    str = regexprep(str, 'p([01])', 'p[$1]');
    str = regexprep(str, 'b([01])', 'b[$1]');
    str = regexprep(str, 'r([012])([012])', 'R[:,$1,$2]');
    str = regexprep(str, 'n([012])', 'n[:,$1,0]');
    fprintf(fpy, "t%d = %s\n\n", idx, str);
    idx = idx + 1;
end
fclose('all');
