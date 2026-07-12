function task06(r1, v1, r2, v2, time)
[x0, vx0, y0, vy0, z0, vz0] = task05(r1, v1, r2, v2, time, "True");
n = task04(r1, "True");
[r, ~] = task02(r1, v1, r2, v2, time, "True");
R = zeros(size(r));

% Calculate the resulting positions based on the initial conditions and time
for i = 1:length(time)
    XZ = A_xz(n, time(i)) * [x0; vx0; z0; vz0];
    Y = A_y(n, time(i)) * [y0; vy0];
    R(i, :) = [XZ(1), Y, XZ(2)];
end

fprintf("position difference over 30 hours: %.4f [m]\n", norm(r(1)-R(1)-r(end)+R(end)));

figure;
plot3(r(:,1), r(:,2), r(:,3), R(:,1), R(:,2), R(:,3));
grid on;
xlabel('x.along [m]');
ylabel('y.cross [m]');
zlabel('z.radial [m]');
title('position difference between perturbed and unperturbed orbit');
legend('Perturbed', 'Hill');

figure;
plot3(r(:,1)-R(:,1), r(:,2)-R(:,2), r(:,3)-R(:,3));
grid on;
xlabel('x.along [m]');
ylabel('y.cross [m]');
zlabel('z.radial [m]');
title('position difference between perturbed and unperturbed orbit');
end

function Y = A_y(n, t) 
Y = [cos(n*t) 1/n*sin(n*t)]; 
end

function XZ = A_xz(n, t) 
XZ = [1 4/n*sin(n*t)-3*t 6*(sin(n*t)-n*t) 2/n*(cos(n*t)-1); 0 2/n*(1-cos(n*t)) 4-3*cos(n*t) 1/n*sin(n*t)]; 
end    