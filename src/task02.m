function [r, v] = task02(r1, v1, r2, v2, time, mode)
unit_z = r1 ./ vecnorm(r1, 2, 2);
y = cross(r1, v1, 2);
unit_y = y ./ vecnorm(y, 2, 2);
unit_x = cross(unit_y, unit_z, 2);
del_r = r2 - r1;
del_v = v2 - v1;
r = [dot(del_r, unit_x, 2), dot(del_r, unit_y, 2), dot(del_r, unit_z, 2)];
v = [dot(del_v, unit_x, 2), dot(del_v, unit_y, 2), dot(del_v, unit_z, 2)];

if ~isempty(mode) return; end

v_s = [dot(v1, unit_x, 2), dot(v1, unit_y, 2), dot(v1, unit_z, 2)];

figure;
subplot(3, 1, 1);
plot(time, v_s(:,1));
grid on; 
xlabel('time [s]');
ylabel('velocity x [m/s]');
title('along direction velocity in orbit-fixed frame');

subplot(3, 1, 2);
plot(time, v_s(:,2));
grid on; 
xlabel('time [s]');
ylabel('velocity y [m/s]');
title('cross direction velocity in orbit-fixed frame');

subplot(3, 1, 3);
plot(time, v_s(:,3));
grid on; 
xlabel('time [s]');
ylabel('velocity z [m/s]');
title('radial direction velocity in orbit-fixed frame');
end