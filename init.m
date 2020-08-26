clear;
% close all;
clc;
dbstop if error;

current_tests=[ pwd '/current_tests' ];
fct = genpath([ pwd '/functions' ]);
mains = [ pwd '/mains'];
test_Cpp_Matlab = [ pwd '/test_Cpp_Matlab' ];
addpath(pwd)
addpath(fct)
addpath current_tests mains test_Cpp_Matlab;
clear fct current_tests mains test_Cpp_Matlab;