clear; clc; close all;

terrasar = readmatrix("terrasar.eph", "FileType", "text");
tandemx = readmatrix("tandemx.eph", "FileType", "text");

r1 = terrasar(:,1:3);
v1 = terrasar(:,4:6);
r2 = tandemx(:,1:3);
v2 = tandemx(:,4:6);
time = linspace(0,108000,10801);

task01(r1, v1, r2, v2, time);
task02(r1, v1, r2, v2, time, []);
task03(r1, v1, r2, v2, time)
task04(r1, []);
task05(r1, v1, r2, v2, time, []);
task06(r1, v1, r2, v2, time);