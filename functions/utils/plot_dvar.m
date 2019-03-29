clear all;
close all;

% We load all needed parameters from the results of the simulation
% TODO : see how results are organized and deduce K, A_tilde and D

t = ;
% Loop on each mode
for i=1:m
    Ki = ;
    first = ;
    plot(t,first)
    hold on
    
    Ai_tilde = ;
    second = 2*cov(Ai_tilde, bt_MCMC(:,i));
    plot(t, second)
    hold on
    
    Di = ;
    third = 2*cov(Di, bt_MCMC(:,i));
    plot(t, third)
    hold on
    
    dVar = first + second + third;
    plot(t, dVar)
    
    xlabel('Time')
    ylabel('Energy of the mode')
end
