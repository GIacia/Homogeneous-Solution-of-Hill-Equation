function n = task04(r1, mode)
mu = 398600441800000;
r = vecnorm(r1, 2, 2);
a = mean(r);
n = sqrt(mu / a^3);
P = 2*pi/n;

if ~isempty(mode) return; end
fprintf("Mean Geocentric distance a =\t%.4f [km]\n", a/1000);
fprintf("Mean Motion n =\t\t\t%.4f [rad/s]\n", n);
fprintf("Orbital Period P =\t\t%.4f [s]\n", P);
end