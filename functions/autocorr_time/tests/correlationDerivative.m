%% Comparison between the correlation time estimation for the residual and its derivative
clear all, close all, clc;

% Load the residual
data = load('C_DNS100_16Modes.mat');
dt = 0.05;
% data = load('C_DNS300_16Modes.mat');
% dt = 0.25;
cov_v = data.c;
cov_v = cov_v(1 : 10000, 1 : 10000);
bt = data.bt;
clear data;

% Estimations for the residual
tau_lms = correlationTimeLMS(cov_v, bt, dt);
tau_htgen = simpleCorrelationTime(cov_v, bt, dt);
% tau_trunc = correlationTimeCut(cov_v, bt);
clear cov_v;

% Load the residual's derivative
% load('..\data\DNS300_inc3d_3D_2017_04_02_NOT_BLURRED_blocks_truncated_pre_c.mat', 'dt_c');
load('..\data\DNS100_inc3d_2D_2018_11_16_blocks_truncated_pre_c.mat', 'dt_c');
dt_c  = dt_c(1 : 10000, 1 : 10000);
db_t = diff(bt, 1, 1);

% Estimations for its derivative
dtau_lms = correlationTimeLMS(dt_c, db_t, dt);
dtau_htgen = simpleCorrelationTime(dt_c, db_t, dt);
% dtau_trunc = correlationTimeCut(dt_c, db_t);
