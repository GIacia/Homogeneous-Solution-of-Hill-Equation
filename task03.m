function task03(r1, v1, r2, v2, time)
[r, v] = task02(r1, v1, r2, v2, time, "True");

figure;

subplot(2, 5, [1,2,3]);
plot(time, r(:,1), time, r(:,2), time, r(:,3));
title('position difference in terrasar body frame');
grid on; 
xlabel('time [s]');
ylabel('position [m]');
legend;

subplot(2, 5, [6,7,8]);
plot(time, v(:,1), time, v(:,2), time, v(:,3));
title('velocity difference in terrasar body frame');
grid on; 
xlabel('time [s]');
ylabel('velocity [m/s]');
legend;

subplot(2, 5, [4,5,9,10]);
plot3(r(:,1), r(:,2), r(:,3));
title('position difference in terrasar body frame');
grid on; 
xlabel('delta x [m]');
ylabel('delta y [m]');
zlabel('delta z [m]');
end