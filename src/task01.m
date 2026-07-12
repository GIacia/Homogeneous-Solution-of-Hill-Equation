function task01(r1, v1, r2, v2, time)
figure;
plot3(r1(:,1), r1(:,2), r1(:,3), r2(:,1), r2(:,2), r2(:,3));
grid on;
xlabel('x [m]');
ylabel('y [m]');
zlabel('z [m]');
title('position in inertial frame');
legend('terrasar', 'tandemx');


figure;
subplot(3, 1, 1);
plot(time, r1(:,1)-r2(:,1));
grid on; 
xlabel('time [s]');
ylabel('delta_x [m]');
title('position difference of x in inertial frame');

subplot(3, 1, 2);
plot(time, r1(:,2)-r2(:,2));
grid on; 
xlabel('time [s]');
ylabel('delta_y [m]');
title('position difference of y in inertial frame');

subplot(3, 1, 3);
plot(time, r1(:,3)-r2(:,3));
grid on; 
xlabel('time [s]');
ylabel('delta_z [m]');
title('position difference of z in inertial frame');


figure;
plot(time, sqrt((r1(:,1)-r2(:,1)).^2+(r1(:,2)-r2(:,2)).^2+(r1(:,3)-r2(:,3)).^2));
grid on; 
xlabel('time [s]');
ylabel('distance [m]');
title('distance of two satellites in inertial frame');


figure;
plot(time, sqrt((v1(:,1)-v2(:,1)).^2+(v1(:,2)-v2(:,2)).^2+(v1(:,3)-v2(:,3)).^2));
grid on; 
xlabel('time [s]');
ylabel('velocity [m/s]');
title('velocity difference of two satellites in inertial frame');


figure;
subplot(3, 1, 1);
unit_r1 = r1 ./ vecnorm(r1, 2, 2);
radial_v1 = dot(v1-v2, unit_r1, 2);
plot(time, radial_v1);
grid on; 
xlabel('time [s]');
ylabel('vel diff [m/s]');
title('radial vel difference proj into radial dir of Terrasar');

subplot(3, 1, 2);
unit_r2 = r2 ./ vecnorm(r2, 2, 2);
radial_v2 = dot(v1-v2, unit_r2, 2);
plot(time, radial_v2);
grid on; 
xlabel('time [s]');
ylabel('vel diff [m/s]');
title('radial vel difference proj into radial dir of TandemX');

subplot(3, 1, 3);
plot(time, radial_v1 - radial_v2);
grid on; 
xlabel('time [s]');
ylabel('delta\_delta\_V [m/s]');
title('difference of [radial vel diff proj into radial direction] of tandemX and Terrasar');
end