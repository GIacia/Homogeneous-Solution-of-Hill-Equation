function [x0, vx0, y0, vy0, z0, vz0] = task05(r1, v1, r2, v2, time, mode)
[r, ~] = task02(r1, v1, r2, v2, time, "True");
n = task04(r1, "True");
r = r(2000:9000,:);

A_XZ = zeros(14002,4);
A_Y = zeros(7001,2);

for i = 2000:9000
    j = i - 2000;
    A_XZ(2*j+1:2*j+2, :) = A_xz(n, time(i));
    A_Y(j+1, :) = A_y(n, time(i));
end

x = r(:,1);
y = r(:,2);
z = r(:,3);
xz = reshape([x z].', [], 1);
y = y(:);

result_xz = LSP(A_XZ) * xz;
result_y = LSP(A_Y) * y;

% result_xz = A_XZ\xz;
% result_y = A_Y\y;

x0 = result_xz(1,1);
vx0 = result_xz(2,1);
z0 = result_xz(3,1);
vz0 = result_xz(4,1);
y0 = result_y(1,1);
vy0 = result_y(2,1);

if ~isempty(mode) return; end

fprintf("< Initial Values, T = 20000 [s] >\n")
fprintf("x0\t=\t%.4f \t[m]\n",x0);
fprintf("vx0\t=\t%.4f   \t[m/s]\n",vx0);
fprintf("y0\t=\t%.4f \t[m]\n",y0);
fprintf("vy0\t=\t%.4f   \t[m/s]\n",vy0);
fprintf("z0\t=\t%.4f \t[m]\n",z0);
fprintf("vz0\t=\t%.4f   \t[m/s]\n",vz0);

end

function Y = A_y(n, t) 
Y = [cos(n*t) 1/n*sin(n*t)]; 
end

function XZ = A_xz(n, t) 
XZ = [1 4/n*sin(n*t)-3*t 6*(sin(n*t)-n*t) 2/n*(cos(n*t)-1); 0 2/n*(1-cos(n*t)) 4-3*cos(n*t) 1/n*sin(n*t)]; 
end

function lsp = LSP(A) 
lsp = (A'*A)\(A'); 
end