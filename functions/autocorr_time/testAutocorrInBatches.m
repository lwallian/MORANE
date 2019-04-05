%
% Author: Agustin PICARD, intern @ Scalian with Valentin RESSEGUIER as
% supervisor
%

%% Test autocorrelation time in batches
clear all, close all, clc;

% data = load('C_DNS100_2Modes.mat');
data = load('C_DNS300_2Modes.mat');
cov_v = data.c;
bt = data.bt;
clear data;

autocorrTime = autocorrelationTimeInBatches(cov_v, bt);

figure, plot(autocorrTime), grid minor;
title('Autocorrelation time (DNS100 - 2 modes)');
xlabel('Period'), ylabel('\tau_{corr}');