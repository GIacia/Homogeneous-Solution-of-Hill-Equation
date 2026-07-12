clear; clc; close all;
cd(fileparts(mfilename("fullpath")));

terrasar = readmatrix("../data/terrasar.eph", "FileType", "text");
tandemx  = readmatrix("../data/tandemx.eph", "FileType", "text");

r1 = terrasar(:,1:3);
v1 = terrasar(:,4:6);
r2 = tandemx(:,1:3);
v2 = tandemx(:,4:6);
time = linspace(0,108000,10801);

task01(r1, v1, r2, v2, time);       % 5 figures
task02(r1, v1, r2, v2, time, []);   % 1 figure
task03(r1, v1, r2, v2, time)        % 1 figure
task04(r1, []);                     % No figure
task05(r1, v1, r2, v2, time, []);   % No figure
task06(r1, v1, r2, v2, time);       % 2 figures